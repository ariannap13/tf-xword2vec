"""Train a word2vec model to obtain word embedding vectors.

There are a total of four combination of architectures and training algorithms
that the model can be trained with:

architecture:
  - skip_gram
  - cbow (continuous bag-of-words)

training algorithm
  - negative_sampling
  - hierarchical_softmax
"""
import os

import tensorflow as tf
import numpy as np
from absl import app
from absl import flags

from dataset import WordTokenizer
from dataset import Word2VecDatasetBuilder
from model import Word2VecModel
from word_vectors import WordVectors

import utils

import pickle

import warnings
warnings.filterwarnings('ignore')

# resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
# tf.config.experimental_connect_to_cluster(resolver)
# This is the TPU initialization code that has to be at the beginning.
# tf.tpu.experimental.initialize_tpu_system(resolver)
# print("All devices: ", tf.config.list_logical_devices('TPU'))
# strategy = tf.distribute.TPUStrategy(resolver)

# allow GPU
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  print(
      '\n\nThis error most likely means that this notebook is not '
      'configured to use a GPU.  Change this in Notebook Settings via the '
      'command palette (cmd/ctrl-shift-P) or the Edit menu.\n\n')
  raise SystemError('GPU device not found')


flags.DEFINE_string('arch', 'skip_gram', 'Architecture (skip_gram or cbow).')
flags.DEFINE_string('algm', 'negative_sampling', 'Training algorithm '
    '(negative_sampling or hierarchical_softmax).')
flags.DEFINE_integer('epochs', 1, 'Num of epochs to iterate thru corpus.')
flags.DEFINE_integer('batch_size', 256, 'Batch size.')
flags.DEFINE_integer('max_vocab_size', 0, 'Maximum vocabulary size. If > 0, '
    'the top `max_vocab_size` most frequent words will be kept in vocabulary.')
flags.DEFINE_integer('min_count', 1, 'Words whose counts < `min_count` will '
    'not be included in the vocabulary.')
flags.DEFINE_float('sample', 1e-3, 'Subsampling rate.')
flags.DEFINE_integer('window_size', 3, 'Num of words on the left or right side'
    ' of target word within a window.')

flags.DEFINE_integer('hidden_size', 300, 'Length of word vector.')
flags.DEFINE_integer('negatives', 5, 'Num of negative words to sample.')
flags.DEFINE_float('power', 0.75, 'Distortion for negative sampling.')
flags.DEFINE_float('alpha', 0.025, 'Initial learning rate.')
flags.DEFINE_float('min_alpha', 0.0001, 'Final learning rate.')
flags.DEFINE_boolean('add_bias', True, 'Whether to add bias term to dotproduct '
    'between syn0 and syn1 vectors.')

flags.DEFINE_integer('log_per_steps', 10000, 'Every `log_per_steps` steps to '
    ' log the value of loss to be minimized.')
flags.DEFINE_list(
    'filenames', None, 'Names of comma-separated input text files.')
flags.DEFINE_string('out_dir', '/tmp/word2vec', 'Output directory.')

FLAGS = flags.FLAGS


def main(_):
  arch = FLAGS.arch
  algm = FLAGS.algm
  epochs = FLAGS.epochs
  batch_size = FLAGS.batch_size
  max_vocab_size = FLAGS.max_vocab_size
  min_count = FLAGS.min_count
  sample = FLAGS.sample
  window_size = FLAGS.window_size
  hidden_size = FLAGS.hidden_size
  negatives = FLAGS.negatives
  power = FLAGS.power
  alpha = FLAGS.alpha
  min_alpha = FLAGS.min_alpha
  add_bias = FLAGS.add_bias
  log_per_steps = FLAGS.log_per_steps
  filenames = FLAGS.filenames
  out_dir = FLAGS.out_dir

  tokenizer = WordTokenizer(
      max_vocab_size=max_vocab_size, min_count=min_count, sample=sample)
  tokenizer.build_vocab(filenames)

  builder = Word2VecDatasetBuilder(tokenizer,
                                   arch=arch,
                                   algm=algm,
                                   epochs=epochs,
                                   batch_size=batch_size,
                                   window_size=window_size)
  dataset = builder.build_dataset(filenames)
  word2vec = Word2VecModel(tokenizer.unigram_counts,
               arch=arch,
               algm=algm,
               hidden_size=hidden_size,
               batch_size=batch_size,
               negatives=negatives,
               power=power,
               alpha=alpha,
               min_alpha=min_alpha,
               add_bias=add_bias)

  train_step_signature = utils.get_train_step_signature(
      arch, algm, batch_size, window_size, builder._max_depth)
  optimizer = tf.keras.optimizers.SGD(1.0)

  @tf.function(input_signature=train_step_signature)
  def train_step(inputs, labels, progress):

    loss = word2vec(inputs, labels)
    gradients = tf.gradients(loss, word2vec.trainable_variables, unconnected_gradients='zero')

    learning_rate = tf.maximum(alpha * (1 - progress[0]) +
        min_alpha * progress[0], min_alpha)

    if hasattr(gradients[0], '_values'):
      gradients[0]._values *= learning_rate
    else:
      gradients[0] *= learning_rate

    if hasattr(gradients[1], '_values'):
      gradients[1]._values *= learning_rate
    else:
      gradients[1] *= learning_rate

    if hasattr(gradients[2], '_values'):
      gradients[2]._values *= learning_rate
    else:
      gradients[2] *= learning_rate

    optimizer.apply_gradients(
        zip(gradients, word2vec.trainable_variables))

    return loss, learning_rate


  average_loss = 0.
 # list_trpoints = []
  # training su corpus completo
  for step, (inputs, labels, progress, nsent) in enumerate(dataset):
    if inputs != labels:
        #print(tokenizer._table_words[inputs.numpy()[0]], tokenizer._table_words[labels.numpy()[0]], nsent.numpy()[0])
        with tf.device('/device:GPU:0'): # forse da cambiare con with  with tf.device('/cpu:0'): per tornare alla velocità che aveva con la cpu ????
            loss, learning_rate = train_step(inputs, labels, progress) # vector of losses for each element of the set [(target,c1), (target,neg1), (target,neg2),...]

        average_loss += loss.numpy().mean() # this is the mean loss per training instance [(target,c1), (target,neg1), (target,neg2),...], I add it to total avg loss
        if step % log_per_steps == 0:
            if step > 0:
                average_loss /= log_per_steps
            print('step:', step, 'average_loss:', average_loss,
            'learning_rate:', learning_rate.numpy())
            average_loss = 0.

#    list_trpoints.append((inputs.ref(), labels.ref(), nsent.ref()))
  print("Training completed")

 # set_trpoints = set(list_trpoints)
  # re-build dataset only with one epoch (each tuple (target, context_word) appears only once)
  builder = Word2VecDatasetBuilder(tokenizer,
                                     arch=arch,
                                     algm=algm,
                                     epochs=1, # un'unica iterazione su totale frasi
                                     batch_size=1, # un'unico esempio trattato alla volta
                                     window_size=window_size)
  dataset = builder.build_dataset(filenames)
  print("Dataset re-built")
# decido di calcolare la loss senza negative sampling, completa

  # liste_termini_weat
  S = ["technology"]
  T = ["art"]
  A = ["man", "boy", "brother", "he", "him", "his"]
  B = ["woman", "she", "her"]

  list_index_weat = []
  for word in S+T+A+B:
      list_index_weat.append(tokenizer._vocab[word])

  diz_gradients = {} # dictionary like {(inputs, labels, nsent): gradient}
  hessian_diz = {} # dictionary like {inputs: hessian}


  tot_count = 0
  for step, (inputs, labels, progress, nsent) in enumerate(dataset): # forse si può rendere più efficiente ed evitare di iterare su tutto
    if inputs.numpy()[0] in list_index_weat:
        if inputs != labels:
            tot_count+=1
  print("Tot. number of inputs to consider: ", tot_count)

  vocab_len = word2vec._vocab_size

  i=0
  for step, (inputs, labels, progress, nsent) in enumerate(dataset):
      if inputs.numpy()[0] in list_index_weat:
          if inputs != labels:
              i+=1
              with tf.device('/device:GPU:0'):
                  with tf.GradientTape(persistent=True) as tape:
                      loss = word2vec._full_loss(inputs, labels, vocab_len)
                      tape.watch(loss)
                      gradients = tape.gradient(loss, word2vec.trainable_variables, unconnected_gradients='zero')
                  jacobian = tape.batch_jacobian(gradients[0], word2vec.trainable_variables[0])
              print("Progress: ", i/tot_count, "Number: ", i)

              grad_loss = gradients[0]._values.numpy()[0]
              #hessian_loss = jacobian[inputs.numpy()[0]].numpy()
              # potrei già direttamente salvare la matrice invertita con
              hessian_loss = tf.linalg.inv(jacobian[inputs.numpy()[0]]).numpy()
              diz_key = (tokenizer._table_words[inputs.numpy()[0]], tokenizer._table_words[labels.numpy()[0]], str(nsent.numpy()[0]))

              if diz_key not in diz_gradients:
                  diz_gradients[diz_key] = grad_loss.copy()
              else:
                  diz_gradients[diz_key] += grad_loss.copy()

              if tokenizer._table_words[inputs.numpy()[0]] not in hessian_diz:
                  hessian_diz[tokenizer._table_words[inputs.numpy()[0]]] = hessian_loss.copy()
              else:
                  hessian_diz[tokenizer._table_words[inputs.numpy()[0]]] += hessian_loss.copy()

  # Store data (serialize)
  with open(os.path.join(FLAGS.out_dir, 'dict_gradients.pickle'), 'wb') as handle_grad:
      pickle.dump(diz_gradients, handle_grad, protocol=pickle.HIGHEST_PROTOCOL)

  with open(os.path.join(FLAGS.out_dir, 'dict_hessians.pickle'), 'wb') as handle_hes:
      pickle.dump(hessian_diz, handle_hes, protocol=pickle.HIGHEST_PROTOCOL)

  #print(hessian_diz)
  syn0_final = word2vec.weights[0].numpy()
  np.save(os.path.join(FLAGS.out_dir, 'syn0_final'), syn0_final)
  with tf.io.gfile.GFile(os.path.join(FLAGS.out_dir, 'vocab.txt'), 'w') as f:
    for w in tokenizer.table_words:
      f.write(w + '\n')
  print('Word embeddings saved to',
      os.path.join(FLAGS.out_dir, 'syn0_final.npy'))
  print('Vocabulary saved to', os.path.join(FLAGS.out_dir, 'vocab.txt'))


if __name__ == '__main__':
  flags.mark_flag_as_required('filenames')
  # training and saving of useful elements
  app.run(main)
