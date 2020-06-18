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
  
# del_all_flags(tf.flags.FLAGS)

flags = tf.app.flags

flags.DEFINE_string('arch', 'skip_gram', 'Architecture (skip_gram or cbow).')
flags.DEFINE_string('algm', 'negative_sampling', 'Training algorithm '
    '(negative_sampling or hierarchical_softmax).')
flags.DEFINE_integer('epochs', 5, 'Num of epochs to iterate training data.')
flags.DEFINE_integer('batch_size', 500, 'Batch size.')
flags.DEFINE_integer('max_vocab_size', 0, 'Maximum vocabulary size. '
                     'If > 0, the top `max_vocab_size` most frequent words'
                     ' are kept in vocabulary.')
flags.DEFINE_integer('min_count', 2, 'Words whose counts < `min_count` are not'
                                     ' included in the vocabulary.')
flags.DEFINE_float('sample', 0.01, 'Subsampling rate.')
flags.DEFINE_integer('window_size', 7, 'Num of words on the left or right side' 
                                       ' of target word within a window.')
flags.DEFINE_integer('embed_size', 300, 'Length of word vector.')
flags.DEFINE_integer('negatives', 5, 'Num of negative words to sample.')
flags.DEFINE_float('power', 0.75, 'Distortion for negative sampling.')
flags.DEFINE_float('alpha', 0.025, 'Initial learning rate to Gradient Descent.')
flags.DEFINE_float('min_alpha', 0.001, 'Final learning rate.')
flags.DEFINE_boolean('add_bias', True, 'Whether to add bias term to dotproduct '
                                       'between syn0 and syn1 vectors.')
flags.DEFINE_integer('log_per_steps', 1000, 'Every `log_per_steps` steps to '
                                            ' output logs.')
flags.DEFINE_list('filenames', None, 'Names of comma-separated input text files.')
flags.DEFINE_string('out_dir', 'data/out', 'Output directory.')
flags.DEFINE_integer('seed', 7, 'Seed to fix sequence of random values.')
flags.DEFINE_string('optim', 'GradDescProx', 'Optimization algorithm '
                                     '(GradDescProx, GradDesc, Adam, AdaGradProx).')

FLAGS = flags.FLAGS

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
                           optim=FLAGS.optim)
  to_be_run_dict = word2vec.train(dataset, FLAGS.filenames)

  with tf.compat.v1.Session() as sess:
    sess.run(dataset.iterator_initializer)
    sess.run(tf.compat.v1.tables_initializer())
    sess.run(tf.compat.v1.global_variables_initializer())

    print("optimizer: ", FLAGS.optim)
    print("epochs: ", FLAGS.epochs)
    
    # open log file
    log_arq = "log_" + FLAGS.optim + "_" + FLAGS.arch + time_sufix(".log")
    flog = open(os.path.join(curr_path, FLAGS.out_dir, log_arq), "w", encoding="utf-8")
    flog.write("Step\tEpoch\tLoss")

    average_loss = 0.
    step = 0
    sub_step = 0
    nepoch = 0
    
    while True:      
      try: 
        result_dict = sess.run(to_be_run_dict)
      except tf.errors.OutOfRangeError:
        break
      
      no_log = True
      step += 1
      sub_step += 1
      average_loss += result_dict['loss'].mean()
      
      if result_dict['epoch'][0] > nepoch:
        syn0_partial = sess.run(word2vec.syn0)
        np.save(os.path.join(FLAGS.out_dir, 'embed_' + 
                             str(nepoch).zfill(2)), syn0_partial)
        nepoch = result_dict['epoch'][0]
        print('-------------------- Epoch: ', nepoch)
      
      if nepoch > 1:
        divisor = FLAGS.log_per_steps / 2
      else:
        divisor = FLAGS.log_per_steps

      if step % divisor == 0:
        average_loss /= FLAGS.log_per_steps
        print('epoch:', nepoch, ' step:', step)
        print(' average loss:', average_loss)
        print(' learning rate:', result_dict['learning_rate'])
        print('------------------------------------')
        flog.write("\n" + str(step) + "\t" + str(nepoch) \
                                    + "\t" + str(average_loss))

        syn0_partial = sess.run(word2vec.syn0)
        np.save(os.path.join(FLAGS.out_dir, 'embed_' + 
                             str(nepoch).zfill(2) + "_step_" +
                             str(step).zfill(6))
                , syn0_partial)
        average_loss = 0.
        sub_step = 0
        no_log = False
      
    if no_log:
      if sub_step > 0:
        average_loss /= sub_step
        print('epoch:', nepoch, ' step:', step) 
        print(' average loss:', average_loss)
        print(' learning rate:', result_dict['learning_rate'])
        
        flog.write("\n" + str(step) + "\t" + str(nepoch) \
                                    + "\t" + str(average_loss))
    flog.close()
    
    syn0_final = sess.run(word2vec.syn0)
    np.save(os.path.join(FLAGS.out_dir, 'embed'), syn0_final)
    
    with open(os.path.join(FLAGS.out_dir, 'vocab.txt'), 'w', encoding="utf-8") as fid:
      for w in dataset.table_words:
        fid.write(w + '\n')       
      fid.close()
      
    sess.close()
    
  print('Word embeddings saved to', os.path.join(FLAGS.out_dir, 'embed.npy'))
  print('Vocabulary saved to', os.path.join(FLAGS.out_dir, 'vocab.txt'))

if __name__ == '__main__':
  
  tf.flags.mark_flag_as_required('filenames')

  tf.compat.v1.app.run()