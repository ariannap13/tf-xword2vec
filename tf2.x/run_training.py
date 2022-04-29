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

from collections import Counter

import warnings
warnings.filterwarnings('ignore')

resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
tf.config.experimental_connect_to_cluster(resolver)
# This is the TPU initialization code that has to be at the beginning.
tf.tpu.experimental.initialize_tpu_system(resolver)
print("All devices: ", tf.config.list_logical_devices('TPU'))
strategy = tf.distribute.TPUStrategy(resolver)


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
                                     epochs=1,
                                     batch_size=batch_size,
                                     window_size=window_size)
  dataset = builder.build_dataset(filenames)
  print("Dataset re-built")
# decido di calcolare la loss senza negative sampling, completa

  # liste_termini_weat
  S = ["science", "technology"]
  T = ["art"]
  A = ["man", "boy", "brother", "he", "him", "his"]
  B = ["woman", "she", "her"]

  list_index_weat = []
  for word in S+T+A+B:
      list_index_weat.append(tokenizer._vocab[word])

  diz_gradients = {} # dictionary like {(inputs, labels, nsent): gradient}
  hessian_diz = {} # dictionary like {inputs: hessian}

  for step, (inputs, labels, progress, nsent) in enumerate(dataset):
      if inputs.numpy()[0] in list_index_weat:


  # DA SISTEMARE per rendere più efficiente
  # devo calcolare total_expsum per ciascun target (solo parole WEAT)
#  diz_expsum_input = {}

  # forse questo potrebbe velocizzare
 # keys = [elem[0] for elem in set_trpoints if tokenizer._table_words[elem[0].deref().numpy()[0]] in S+T+A+B]
 # diz_expsum_input = {key: None for key in keys} # così posso aggiungere direttamente elementi senza dover cercare in tutta la lista di chiavi

 # for tr_point in set_trpoints:
#      inputs = tr_point[0].deref()
#      labels = tr_point[1].deref()
#      nsent = tr_point[2].deref()
#      if inputs.numpy()[0] in list_index_weat:
#          if inputs.ref() not in diz_expsum_input:
#              diz_expsum_input[inputs.ref()] = word2vec.exp_sum(inputs)


#  for step, (inputs, labels, progress, nsent) in enumerate(dataset): # forse si può rendere più efficiente ed evitare di iterare su tutto
#    if inputs.numpy()[0] in list_index_weat:
#        print("one of the weat words")
#        if inputs.numpy()[0] not in diz_expsum_input:
#            diz_expsum_input[inputs.numpy()[0]] = word2vec.exp_sum(inputs)

#  print("Exponential sum computed")
  vocab_len = word2vec._vocab_size
  for step, (inputs, labels, progress, nsent) in enumerate(dataset):
      if inputs.numpy()[0] in list_index_weat:
          with strategy.scope():
              with tf.GradientTape(persistent=True) as tape:
                 # expsum = diz_expsum_input[inputs.numpy()[0]]
                  loss = word2vec.normal_loss(inputs, labels, vocab_len)
                  print(loss)
                  tape.watch(loss)
                  gradients = tape.gradient(loss, word2vec.trainable_variables, unconnected_gradients='zero')
              jacobian = tape.batch_jacobian(gradients[0], word2vec.trainable_variables[0])
            #  del tape

          grad_loss = gradients[0]._values.numpy()[0]
          hessian_loss = jacobian[inputs.numpy()[0]].numpy()
          diz_key = (tokenizer._table_words[inputs.numpy()[0]], tokenizer._table_words[labels.numpy()[0]], str(nsent.numpy()[0]))
          if diz_key not in diz_gradients:
              diz_gradients[diz_key] = grad_loss
          else:
              diz_gradients[diz_key] += grad_loss

          if tokenizer._table_words[inputs.numpy()[0]] not in hessian_diz:
              hessian_diz[tokenizer._table_words[inputs.numpy()[0]]] = hessian_loss
          else:
              hessian_diz[tokenizer._table_words[inputs.numpy()[0]]] += hessian_loss

  # Store data (serialize)
  with open('dict_gradients.pickle', 'wb') as handle_grad:
      pickle.dump(diz_gradients, handle_grad, protocol=pickle.HIGHEST_PROTOCOL)

  with open('dict_hessians.pickle', 'wb') as handle_hes:
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
