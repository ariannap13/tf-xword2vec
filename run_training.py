r"""Executable for training Word2Vec models. 

Example:
  python run_training.py \
    --filenames=/PATH/TO/FILE/file1.txt,/PATH/TO/FILE/file2.txt \
    --out_dir=/PATH/TO/OUT_DIR/ \
    --batch_size=64 \
    --window_size=5 \

Learned word embeddings will be saved to /PATH/TO/OUT_DIR/embed.npy, and
vocabulary saved to /PATH/TO/OUT_DIR/vocab.txt
"""
import os
import numpy as np
import pandas as pd
pd.set_option('io.hdf.default_format','table')

import tensorflow as tf

# import project files
from dataset import Word2VecDataset
from word2vec import Word2VecModel
import data_util as du


flags = tf.app.flags

flags.DEFINE_string('arch', 'skip_gram', 'Architecture (skip_gram or cbow).')
flags.DEFINE_string('algm', 'negative_sampling', 'Training algorithm '
    '(negative_sampling or hierarchical_softmax).')
flags.DEFINE_integer('epochs', 3, 'Num of epochs to iterate training data.')
flags.DEFINE_integer('batch_size', 3000, 'Batch size.')
flags.DEFINE_integer('max_vocab_size', 0, 'Maximum vocabulary size. '
                     'If > 0, the top `max_vocab_size` most frequent words'
                     ' are kept in vocabulary.')
flags.DEFINE_integer('min_count', 5, 'Words whose counts < `min_count` are not'
                                     ' included in the vocabulary.')
flags.DEFINE_float('sample', 0.01, 'Subsampling rate.')
flags.DEFINE_integer('window_size', 6, 'Num of words on the left or right side' 
                                       ' of target word within a window.')
flags.DEFINE_integer('embed_size', 200, 'Length of word vector.')
flags.DEFINE_integer('negatives', 10, 'Num of negative words to sample.')
flags.DEFINE_float('power', 0.75, 'Distortion for negative sampling.')
flags.DEFINE_float('alpha', 0.020, 'Initial learning rate.')
flags.DEFINE_float('min_alpha', 0.003, 'Final learning rate and recommended Adam lr.')
flags.DEFINE_boolean('add_bias', True, 'Whether to add bias term to dotproduct'
                                       ' between syn0 and syn1 vectors.')
flags.DEFINE_integer('log_per_steps', 500, 'Every `log_per_steps` steps to '
                                            ' output logs.')
flags.DEFINE_list('filenames', 'data/tok_2012.pt', 'Names of comma-separated input text files.')
flags.DEFINE_string('out_dir', 'data/out', 'Output directory.')
flags.DEFINE_integer('seed', 70, 'Seed to fix sequence of random values.')
flags.DEFINE_string('optim', 'GradDesc', 'Optimization algorithm '
                            '(GradDescProx, Adam, AdaGradProx, GradDesc).')
flags.DEFINE_string('decay', 'cos', 'Polynomial (poly), cosine (cos), step or (no).')
flags.DEFINE_integer('special_tokens', 1, 'Whether to remove special tokens from'
                                       ' data files.')
flags.DEFINE_string('focus', "", 'Whether to remove special tokens from'
                                       ' data files.')

FLAGS = flags.FLAGS

# ================== main ================== 
def main(_):
  dataset = Word2VecDataset(arch=FLAGS.arch,
                            algm=FLAGS.algm,
                            batch_size=FLAGS.batch_size,
                            max_vocab_size=FLAGS.max_vocab_size,
                            min_count=FLAGS.min_count,
                            sample=FLAGS.sample,
                            window_size=FLAGS.window_size,
                            special_tokens=FLAGS.special_tokens,
                            focus=FLAGS.focus)
  
  dataset.focus = ""  
  dataset.build_vocab(FLAGS.filenames)

  word2vec = Word2VecModel(arch=FLAGS.arch,
                           algm=FLAGS.algm,
                           embed_size=FLAGS.embed_size,
                           batch_size=FLAGS.batch_size,
                           negatives=FLAGS.negatives,
                           power=FLAGS.power,
                           alpha=FLAGS.alpha,
                           min_alpha=FLAGS.min_alpha,
                           add_bias=FLAGS.add_bias,
                           random_seed=FLAGS.seed,
                           optim=FLAGS.optim,
                           decay=FLAGS.decay)
  
  to_be_run_dict = word2vec.train(dataset, FLAGS.filenames, FLAGS.epochs)
  
  datatools = du.DataFileTools(out_path=FLAGS.out_dir)

# ================== session ==================

  with tf.compat.v1.Session() as sess:
    # sess.run(dataset.iterator_initializer)
    sess.run(tf.compat.v1.tables_initializer())
    sess.run(tf.compat.v1.global_variables_initializer())
    
    print("optimizer: ", FLAGS.optim)
    print("epochs: ", FLAGS.epochs)
    print("model: ", FLAGS.arch)
    print("strategy : ", FLAGS.algm)
    print('decay : ', FLAGS.decay)
    print("word in focus: ", dataset.focus)
    list_vocab = dataset.table_words
    word_and_freq = zip(list_vocab,
                    dataset.unigram_counts,
                    dataset.keep_probs) 
    datatools.save_vocab(word_and_freq)
    
    # open log file
    log_arq = "log_" + FLAGS.optim + "_" + FLAGS.arch + "_" + FLAGS.algm \
                                   + datatools.time_sufix(".log")
    flog = open(du.path_file(log_arq, subfolder=FLAGS.out_dir), "w",
                encoding="utf-8")
    flog.write("Step\tEpoch\tAverageLoss\tLearningRate")
    
#    if FLAGS.arch == 'skip_gram' and FLAGS.algm == 'negative_sampling':
    # open H5 store file
    fname = "InputsLabels_" + FLAGS.arch + "_" + \
        {True: "ns", False: "hs"}[FLAGS.algm=="negative_sampling"] + \
        "_window" + str(FLAGS.window_size) + ".h5"
    storeh5 = pd.HDFStore(fname, mode="w", complevel=2, complib='blosc')
    df_vocab = pd.DataFrame(list_vocab, columns=["words"])
    storeh5['df_vocab'] = df_vocab
    storeh5.close()

    step = 0
    sub_step = 0
    train_epoch = 0
    average_loss = 0.
    
    #for train_epoch in range(2, 1 + FLAGS.epochs):
    sess.run(dataset.iterator_initializer)
    while True:      
      try:
          result_dict = sess.run(to_be_run_dict)
#           if train_epoch == 1:
# #            if FLAGS.arch == 'skip_gram' and FLAGS.algm == 'negative_sampling':
#             a_inputs = result_dict['inputs']
#             a_labels = result_dict['labels']
#             df_input = pd.DataFrame(a_inputs)
#             df_label = pd.DataFrame(a_labels)
#             storeh5.append("df_input", df_input, format="t", append=True) 
#             #                min_itemsize={'input': 50, 'label': 50})
#             storeh5.append("df_label", df_label, format="t", append=True) 
              
      except tf.errors.OutOfRangeError:
        break
                  
      no_log = True
      step += 1
      sub_step += 1
      op_lr = result_dict['learning_rate']
      average_loss += result_dict['loss'].mean()
      train_progress = result_dict['progress_rate']
      train_epoch = result_dict['epoch']
      if step == 1:
        # first one
        syn0_partial = sess.run(word2vec.syn0)
        np.save(os.path.join(FLAGS.out_dir, 'embed_' +
                             str(train_epoch).zfill(2) + "_step_" +
                             str(step).zfill(6)), syn0_partial)
        del syn0_partial
        print('-------------------- Epoch: ', train_epoch)
        print(' average loss:', average_loss)
        print(' learning rate:', op_lr)
        print(' progress:', round(train_progress,4))
        print('------------------------------------')
        # save first log line with the first loss and learning rate
        flog.write("\n" + str(step) + "\t" + str(train_epoch) \
                   + "\t" + str(average_loss) \
                   + "\t" + str(op_lr))          
        average_loss = 0.
        sub_step = 0
      else:
        divisor = FLAGS.log_per_steps
        if step % divisor == 0:
          print('epoch:', train_epoch, ' step:', step)
          print(' average loss:', average_loss / sub_step)
          print(' learning rate:', op_lr)
          print(' progress:', round(train_progress,4))
          print('------------------------------------')
          flog.write("\n" + str(step) + "\t" + str(train_epoch) \
                          + "\t" + str(average_loss / sub_step)
                          + "\t" + str(op_lr))
  
          syn0_partial = sess.run(word2vec.syn0)
          np.save(os.path.join(FLAGS.out_dir, 'embed_' +
                               str(train_epoch).zfill(2) + "_step_" +
                               str(step).zfill(6)), syn0_partial)
          del syn0_partial
          average_loss = 0.
          sub_step = 0
          no_log = False
    # while end
#       if train_epoch == 1:
# #        if FLAGS.arch == 'skip_gram' and FLAGS.algm == 'negative_sampling':
#         storeh5.close()
        
    # for end
    if no_log:
      if sub_step > 0:
        print('epoch:', train_epoch, ' step:', step) 
        print(' average loss:', average_loss / sub_step)
        print(' learning rate:', op_lr)
        print(' progress:', 1.)
        flog.write("\n" + str(step) + "\t" + str(train_epoch) \
                                    + "\t" + str(average_loss / sub_step) \
                                    + "\t" + str(op_lr))
    flog.close()
    
    syn0_final = sess.run(word2vec.syn0)
    np.save(os.path.join(FLAGS.out_dir, 'embed_final'), syn0_final)

    datatools.save_embed_proj(syn0_final, list_vocab, FLAGS.out_dir)
      
    sess.close()
    
  print('Word embeddings saved to', os.path.join(FLAGS.out_dir, 'embed.npy'))
  print('Vocabulary saved to', os.path.join(FLAGS.out_dir, 'vocab.txt'))

if __name__ == '__main__':
  
  tf.flags.mark_flag_as_required('filenames')

  tf.compat.v1.app.run()