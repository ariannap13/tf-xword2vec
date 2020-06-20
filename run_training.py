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
import tensorflow as tf

curr_path = os.path.dirname(os.path.abspath(__file__))

# import project files
from dataset import Word2VecDataset
from word2vec import Word2VecModel


flags = tf.app.flags

flags.DEFINE_string('arch', 'skip_gram', 'Architecture (skip_gram or cbow).')
flags.DEFINE_string('algm', 'negative_sampling', 'Training algorithm '
    '(negative_sampling or hierarchical_softmax).')
flags.DEFINE_integer('epochs', 1, 'Num of epochs to iterate training data.')
flags.DEFINE_integer('batch_size', 512, 'Batch size.')
flags.DEFINE_integer('max_vocab_size', 0, 'Maximum vocabulary size. '
                     'If > 0, the top `max_vocab_size` most frequent words'
                     ' are kept in vocabulary.')
flags.DEFINE_integer('min_count', 2, 'Words whose counts < `min_count` are not'
                                     ' included in the vocabulary.')
flags.DEFINE_float('sample', 0.0007, 'Subsampling rate.')
flags.DEFINE_integer('window_size', 6, 'Num of words on the left or right side' 
                                       ' of target word within a window.')
flags.DEFINE_integer('embed_size', 200, 'Length of word vector.')
flags.DEFINE_integer('negatives', 10, 'Num of negative words to sample.')
flags.DEFINE_float('power', 0.75, 'Distortion for negative sampling.')
flags.DEFINE_float('alpha', 0.025, 'Initial learning rate to Gradient Descent.')
flags.DEFINE_float('min_alpha', 0.004, 'Final learning rate.')
flags.DEFINE_boolean('add_bias', True, 'Whether to add bias term to dotproduct'
                                       ' between syn0 and syn1 vectors.')
flags.DEFINE_integer('log_per_steps', 1000, 'Every `log_per_steps` steps to '
                                            ' output logs.')
flags.DEFINE_list('filenames', None, 'Names of comma-separated input text files.')
flags.DEFINE_string('out_dir', 'data/out', 'Output directory.')
flags.DEFINE_integer('seed', 777, 'Seed to fix sequence of random values.')
flags.DEFINE_string('optim', 'Adam', 'Optimization algorithm '
                            '(GradDescProx, GradDesc, Adam, AdaGradProx).')
flags.DEFINE_string('decay', 'no', 'Polynomial (poly), cosine (cos) or (no).')

FLAGS = flags.FLAGS


def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)

def time_sufix(extension=".log"):
    from time import strftime
    sufix = "_" + strftime("%Y-%m-%d_%H-%M")
    sufix = sufix + extension
    return sufix

def remove_last_empty(arq):
  with open(arq, encoding='utf-8') as f_input:
    data = f_input.read().rstrip('\n')
  with open(arq, 'w', encoding='utf-8') as f_output:    
    f_output.write(data)

def save_embed_proj(array_embed, list_vocab, path_embed):
  import pandas as pd
  # create dataframe
  df_embed = pd.DataFrame(array_embed, index=list_vocab)
  # save it in HDF5
  store = pd.HDFStore(os.path.join(FLAGS.out_dir, 'embeddings.h5'))
  store['df_embed'] = df_embed
  store.close()
  # save vector to Google project, for instance
  df_embed.to_csv(os.path.join(path_embed, 'project_embed.vec'),
                  sep='\t', header=False, index=False)
  # remove last blank line in vector file
  remove_last_empty(os.path.join(path_embed, 'project_embed.vec'))
  # create label file
  with open(os.path.join(FLAGS.out_dir, 'project_embed.tsv'), 'w',
       encoding="utf-8") as fw:
    fw.write('word')
    for w in list_vocab:
      fw.write('\n' + w)
    fw.close()

# ================== main ================== 
def main(_):
  dataset = Word2VecDataset(arch=FLAGS.arch,
                            algm=FLAGS.algm,
                            epochs=FLAGS.epochs,
                            batch_size=FLAGS.batch_size,
                            max_vocab_size=FLAGS.max_vocab_size,
                            min_count=FLAGS.min_count,
                            sample=FLAGS.sample,
                            window_size=FLAGS.window_size)
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
  to_be_run_dict = word2vec.train(dataset, FLAGS.filenames)

  with tf.compat.v1.Session() as sess:
    sess.run(dataset.iterator_initializer)
    sess.run(tf.compat.v1.tables_initializer())
    sess.run(tf.compat.v1.global_variables_initializer())

    print("optimizer: ", FLAGS.optim)
    print("epochs: ", FLAGS.epochs)
    print("model: ", FLAGS.arch)
    print("strategy : ", FLAGS.algm)
    
    # open log file
    log_arq = "log_" + FLAGS.optim + "_" + FLAGS.arch + "_" + FLAGS.algm \
                                   + time_sufix(".log")
    flog = open(os.path.join(curr_path, FLAGS.out_dir, log_arq), "w", encoding="utf-8")
    flog.write("Step\tEpoch\tAverageLoss\tLearningRate")

    average_loss = 0.
    step = 0
    sub_step = 0
    
    while True:      
      try:
        result_dict = sess.run(to_be_run_dict)
      except tf.errors.OutOfRangeError:
        break
      
      no_log = True
      step += 1
      sub_step += 1
      average_loss += result_dict['loss'].mean()
      op_lr = sess.run(word2vec.lr)
      op_epoch = result_dict['op_epoch']
      train_progress = result_dict['progress_rate']
      
      if step == 1:
        # first one
        syn0_partial = sess.run(word2vec.syn0)
        np.save(os.path.join(FLAGS.out_dir, 'embed_' + 
                             str(op_epoch).zfill(2)), syn0_partial)
        del syn0_partial
        print('-------------------- Epoch: ', op_epoch)
        print(' average loss:', average_loss / sub_step)
        print(' learning rate:', op_lr)
        print(' progress:', round(train_progress,4))
        print('------------------------------------')
        # save first log line with the first loss and learning rate
        flog.write("\n" + str(step) + "\t" + str(op_epoch) \
                   + "\t" + str(average_loss / sub_step) \
                   + "\t" + str(op_lr))
        # save vocab with frequency
        ff = open(os.path.join(FLAGS.out_dir, 'vocab_freq.txt'), 'w',
                  encoding="utf-8") 
        fw = open(os.path.join(FLAGS.out_dir, 'vocab.txt'), 'w',
                 encoding="utf-8")
        list_vocab = dataset.table_words
        word_and_freq = zip(list_vocab, dataset.unigram_counts, 
                            dataset.keep_probs) 
        for i, w_f in enumerate(word_and_freq):
          if i > 0:
            fw.write('\n')
            ff.write('\n')
          fw.write(w_f[0])
          ff.write(w_f[0] + '\t' + str(w_f[1]) + '\t' + str(w_f[2]))
        fw.close()
        ff.close()
      else:
        divisor = FLAGS.log_per_steps

        if step % divisor == 0:
          print('epoch:', op_epoch, ' step:', step)
          print(' average loss:', average_loss / sub_step)
          print(' learning rate:', op_lr)
          print(' progress:', round(train_progress,4))
          print('------------------------------------')
          flog.write("\n" + str(step) + "\t" + str(op_epoch) \
                          + "\t" + str(average_loss / sub_step)
                          + "\t" + str(op_lr))
  
          syn0_partial = sess.run(word2vec.syn0)
          np.save(os.path.join(FLAGS.out_dir, 'embed_' +
                               str(op_epoch).zfill(2) + "_step_" +
                               str(step).zfill(6)), syn0_partial)
          del syn0_partial
          average_loss = 0.
          sub_step = 0
          no_log = False
      
    if no_log:
      if sub_step > 0:
        average_loss /= sub_step
        print('epoch:', op_epoch, ' step:', step) 
        print(' average loss:', average_loss)
        print(' learning rate:', op_lr)
        print(' progress:', 1.)
        flog.write("\n" + str(step) + "\t" + str(op_epoch) \
                                    + "\t" + str(average_loss/sub_step) \
                                    + "\t" + str(op_lr))
    flog.close()
    
    syn0_final = sess.run(word2vec.syn0)
    np.save(os.path.join(FLAGS.out_dir, 'embed_final'), syn0_final)

    save_embed_proj(syn0_final, list_vocab, FLAGS.out_dir)
    
    sess.close()
    
  print('Word embeddings saved to', os.path.join(FLAGS.out_dir, 'embed.npy'))
  print('Vocabulary saved to', os.path.join(FLAGS.out_dir, 'vocab.txt'))

if __name__ == '__main__':
  
  tf.flags.mark_flag_as_required('filenames')

  tf.compat.v1.app.run()