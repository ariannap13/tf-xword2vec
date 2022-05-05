"""Defines word2vec model using tf.keras API.
"""
import tensorflow as tf

from dataset import WordTokenizer
from dataset import Word2VecDatasetBuilder


class Word2VecModel(tf.keras.Model):
    """Word2Vec model."""

    def __init__(self,
                 unigram_counts,
                 arch='skip_gram',
                 algm='negative_sampling',
                 hidden_size=300,
                 batch_size=256,
                 negatives=5,
                 power=0.75,
                 alpha=0.025,
                 min_alpha=0.0001,
                 add_bias=True,
                 random_seed=0):
        """Constructor.

        Args:
          unigram_counts: a list of ints, the counts of word tokens in the corpus.
          arch: string scalar, architecture ('skip_gram' or 'cbow').
          algm: string scalar, training algorithm ('negative_sampling' or
            'hierarchical_softmax').
          hidden_size: int scalar, length of word vector.
          batch_size: int scalar, batch size.
          negatives: int scalar, num of negative words to sample.
          power: float scalar, distortion for negative sampling.
          alpha: float scalar, initial learning rate.
          min_alpha: float scalar, final learning rate.
          add_bias: bool scalar, whether to add bias term to dotproduct
            between syn0 and syn1 vectors.
          random_seed: int scalar, random_seed.
        """
        super(Word2VecModel, self).__init__()
        self._unigram_counts = unigram_counts
        self._arch = arch
        self._algm = algm
        self._hidden_size = hidden_size
        self._vocab_size = len(unigram_counts)
        self._batch_size = batch_size
        self._negatives = negatives
        self._power = power
        self._alpha = alpha
        self._min_alpha = min_alpha
        self._add_bias = add_bias
        self._random_seed = random_seed
        self._all_losses = {}

        self._input_size = (self._vocab_size if self._algm == 'negative_sampling'
                            else self._vocab_size - 1)

        self.add_weight('syn0',
                        shape=[self._vocab_size, self._hidden_size],
                        initializer=tf.keras.initializers.RandomUniform(
                            minval=-0.5/self._hidden_size,
                            maxval=0.5/self._hidden_size,
                            seed=self._random_seed))

        self.add_weight('syn1',
                        shape=[self._input_size, self._hidden_size],
                        initializer=tf.keras.initializers.RandomUniform(
                            minval=-0.1, maxval=0.1,
                            seed=self._random_seed))

        self.add_weight('biases',
                        shape=[self._input_size],
                        initializer=tf.keras.initializers.Zeros())

    def call(self, inputs, labels):
        """Runs the forward pass to compute loss.

        Args:
          inputs: int tensor of shape [batch_size] (skip_gram) or
            [batch_size, 2*window_size+1] (cbow)
          labels: int tensor of shape [batch_size] (negative_sampling) or
            [batch_size, 2*max_depth+1] (hierarchical_softmax)

        Returns:
          loss: float tensor, cross entropy loss.
        """
        if self._algm == 'negative_sampling':
            loss = self._negative_sampling_loss(inputs, labels)
        elif self._algm == 'hierarchical_softmax':
            loss = self._hierarchical_softmax_loss(inputs, labels)
        return loss

    def _full_loss(self, inputs, labels, vocab_len, get_context=False):
        """Builds the full loss.

        Args:
          inputs: int tensor of shape [batch_size] (skip_gram) or
            [batch_size, 2*window_size+1] (cbow)
          labels: int tensor of shape [batch_size]

        Returns:
          loss: float tensor of shape [batch_size, vocab_size].
        """

        losses = []
        _, syn1, biases = self.weights

        contexts = []
        for b_i in range(self._batch_size):
            vocab_tmp = vocab.copy()
            # remove target and label - problema si ha dal momento in cui target e label sono lo stesso termine: avremo vettori di loss di lunghezza diversa
            vocab_tmp.remove(inputs[0].numpy())
#        if inputs[b_i] != labels[b_i]: # se per sfiga mi capita sia come input che come label
#            vocab_tmp.remove(labels[b_i].numpy())
#            ntoremove = 2
#        else:
#            ntoremove = 1
            vocab_tmp.remove(labels[0].numpy())
            ntoremove = 2
            contexts.append(tf.constant(list(vocab_tmp),
                                        shape=(1, vocab_len - ntoremove)))

        context_mat = tf.concat(contexts, axis=0)
        inputs_syn0 = self._get_inputs_syn0(inputs)
        true_syn1 = tf.gather(syn1, labels)
        context_syn1 = tf.gather(syn1, context_mat)

        true_logits = tf.reduce_sum(tf.multiply(inputs_syn0, true_syn1), 1)

        context_logits = tf.einsum('ijk,ikl->il', tf.expand_dims(inputs_syn0, 1),
                                   tf.transpose(context_syn1, (0, 2, 1)))

        if self._add_bias:

            true_logits += tf.gather(biases, labels)

            context_logits += tf.gather(biases, context_mat)

        true_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(true_logits), logits=true_logits)

        context_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(context_logits), logits=context_logits)

        losses.append(tf.expand_dims(true_cross_entropy, 1))

        if (get_context):
            # returns loss for every word in vocab
            vocab = set(range(vocab_len))

            contexts = []
            for b_i in range(self._batch_size):
                vocab_tmp = vocab.copy()
                # remove target
                vocab_tmp.remove(inputs[b_i].numpy())
                contexts.append(tf.constant(
                    list(vocab_tmp), shape=(1, vocab_len - 1)))

            context_mat = tf.concat(contexts, axis=0)
            context_syn1 = tf.gather(syn1, context_mat)
            # [batch_size, vocab_len -1]
            context_logits = tf.einsum('ijk,ikl->il', tf.expand_dims(inputs_syn0, 1),
                                       tf.transpose(context_syn1, (0, 2, 1)))
            # [batch_size, vocab_len -1]
            context_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(context_logits), logits=context_logits)
            losses.append(context_cross_entropy)

        loss = tf.concat(losses, 1)
        return loss

    def normal_loss_opt(self, inputs, labels, vocab_len):
        """Builds the loss.

        Args:
          vocab_len: int length of vocabolary

        Returns:
          loss: dictionary of every targets and his losses of real label
          & losses of context label.
        """
        res = []
        for b_i in range(self._batch_size):

            if (inputs[b_i].numpy() not in self._all_losses):
                # already exists, return
                # res.append(self._all_losses[inputs[b_i].numpy()])
                # calcola e restituisce

                # tensor of shape [1, vocab_length] made of vocab indexes
                vocab = tf.constant(list(range(vocab_len)),
                                    shape=(1, vocab_len))

                _, syn1, biases = self.weights

                # calc real contexts of given target
                inputs_syn0 = self._get_inputs_syn0(
                    tf.expand_dims(inputs[b_i], 0))  # [batch_size, hidden_size

                # real
                true_labels_syn1 = tf.gather(syn1, vocab)
                # [batch_size, vocab_len]
                true_labels_logits = tf.einsum('ijk,ikl->il', tf.expand_dims(inputs_syn0, 1),
                                               tf.transpose(true_labels_syn1, (0, 2, 1)))
                # [batch_size, vocab_len]
                real_losses = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(true_labels_logits), logits=true_labels_logits)

                # context
                context_syn1 = tf.gather(syn1, vocab)
                # [batch_size, vocab_len -1]
                context_logits = tf.einsum('ijk,ikl->il', tf.expand_dims(inputs_syn0, 1),
                                           tf.transpose(context_syn1, (0, 2, 1)))
                # [batch_size, vocab_len -1]
                context_losses = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.zeros_like(context_logits), logits=context_logits)

                # self._all_losses[inputs[b_i].numpy()] = {
                #   'real': real_losses,  # includes target itself
                #   'context': context_losses
                # }
            else:
                real_losses = self._all_losses[inputs[b_i].numpy()]['real']
                context_losses = self._all_losses[inputs[b_i].numpy(
                )]['context']
            # create concat tensor TRUE LABEL | ...OTHER CONTEXTS
            # res.append(tf.concat([
            #   tf.expand_dims(
            #     self._all_losses[inputs[b_i].numpy()]['real'][0, labels[b_i].numpy()],
            #     0),
            #   self._remove_index(
            #     self._all_losses[inputs[b_i].numpy()]['context'][0],
            #     labels[b_i].numpy()
            #     )
            #   ], axis = 0))

            res.append([
                inputs[b_i].numpy(),
                real_losses,
                context_losses
            ])

        return res

    def _concat_loss(self, _input, label):

        return tf.concat([
            tf.expand_dims(
                self._all_losses[_input]['real'][0, label],
                0),
            self._remove_index(
                self._all_losses[_input]['context'][0],
                label
            )
        ], axis=0)

    def _remove_index(self, tensor, index, replace=False):
        if replace:
            return tf.concat([tensor[0:index], replace, tensor[index+1:len(tensor)]], axis=0)
        return tf.concat([tensor[0:index], tensor[index+1:len(tensor)]], axis=0)

    def _true_loss(self, inputs, labels):
        """Builds the loss for true context.

        Args:
          inputs: int tensor of shape [batch_size] (skip_gram) or
            [batch_size, 2*window_size+1] (cbow)
          labels: int tensor of shape [batch_size]

        Returns:
          loss: float tensor of shape [batch_size, 1].
        """
        _, syn1, biases = self.weights

        inputs_syn0 = self._get_inputs_syn0(
            inputs)  # [batch_size, hidden_size]
        true_syn1 = tf.gather(syn1, labels)  # [batch_size, hidden_size]

        # [batch_size]
        true_logits = tf.reduce_sum(tf.multiply(inputs_syn0, true_syn1), 1)

        # [batch_size]
        true_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(true_logits), logits=true_logits)

        loss = tf.expand_dims(true_cross_entropy, 1)
        return loss

    def _negative_sampling_loss(self, inputs, labels):
        """Builds the loss for negative sampling.

        Args:
          inputs: int tensor of shape [batch_size] (skip_gram) or
            [batch_size, 2*window_size+1] (cbow)
          labels: int tensor of shape [batch_size]

        Returns:
          loss: float tensor of shape [batch_size, negatives + 1].
        """
        _, syn1, biases = self.weights

        sampled_values = tf.random.fixed_unigram_candidate_sampler(
            true_classes=tf.expand_dims(labels, 1),
            num_true=1,
            num_sampled=self._batch_size*self._negatives,
            unique=True,
            range_max=len(self._unigram_counts),
            distortion=self._power,
            unigrams=self._unigram_counts)

        sampled = sampled_values.sampled_candidates
        sampled_mat = tf.reshape(sampled, [self._batch_size, self._negatives])
        inputs_syn0 = self._get_inputs_syn0(
            inputs)  # [batch_size, hidden_size]
        true_syn1 = tf.gather(syn1, labels)  # [batch_size, hidden_size]
        # [batch_size, negatives, hidden_size]
        sampled_syn1 = tf.gather(syn1, sampled_mat)
        # [batch_size]
        true_logits = tf.reduce_sum(tf.multiply(inputs_syn0, true_syn1), 1)
        # [batch_size, negatives]
        sampled_logits = tf.einsum('ijk,ikl->il', tf.expand_dims(inputs_syn0, 1),
                                   tf.transpose(sampled_syn1, (0, 2, 1)))

        if self._add_bias:
            # [batch_size]
            true_logits += tf.gather(biases, labels)
            # [batch_size, negatives]
            sampled_logits += tf.gather(biases, sampled_mat)

        # [batch_size]
        true_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(true_logits), logits=true_logits)
        # [batch_size, negatives]
        sampled_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(sampled_logits), logits=sampled_logits)

        loss = tf.concat(
            [tf.expand_dims(true_cross_entropy, 1), sampled_cross_entropy], 1)
        return loss

    def _hierarchical_softmax_loss(self, inputs, labels):
        """Builds the loss for hierarchical softmax.

        Args:
          inputs: int tensor of shape [batch_size] (skip_gram) or
            [batch_size, 2*window_size+1] (cbow)
          labels: int tensor of shape [batch_size, 2*max_depth+1]

        Returns:
          loss: float tensor of shape [sum_of_code_len]
        """
        _, syn1, biases = self.weights

        inputs_syn0_list = tf.unstack(self._get_inputs_syn0(inputs))
        codes_points_list = tf.unstack(labels)
        max_depth = (labels.shape.as_list()[1] - 1) // 2
        loss = []
        for i in range(self._batch_size):
            inputs_syn0 = inputs_syn0_list[i]  # [hidden_size]
            codes_points = codes_points_list[i]  # [2*max_depth+1]
            true_size = codes_points[-1]

            codes = codes_points[:true_size]
            points = codes_points[max_depth:max_depth+true_size]
            logits = tf.reduce_sum(
                tf.multiply(inputs_syn0, tf.gather(syn1, points)), 1)
            if self._add_bias:
                logits += tf.gather(biases, points)

            # [true_size]
            loss.append(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.cast(codes, 'float32'), logits=logits))
        loss = tf.concat(loss, axis=0)
        return loss

    def _get_inputs_syn0(self, inputs):
        """Builds the activations of hidden layer given input words embeddings
        `syn0` and input word indices.

        Args:
          inputs: int tensor of shape [batch_size] (skip_gram) or
            [batch_size, 2*window_size+1] (cbow)

        Returns:
          inputs_syn0: [batch_size, hidden_size]
        """
        # syn0: [vocab_size, hidden_size]
        syn0, _, _ = self.weights
        if self._arch == 'skip_gram':
            inputs_syn0 = tf.gather(syn0, inputs)  # [batch_size, hidden_size]
        else:
            inputs_syn0 = []
            contexts_list = tf.unstack(inputs)
            for i in range(self._batch_size):
                contexts = contexts_list[i]
                context_words = contexts[:-1]
                true_size = contexts[-1]
                inputs_syn0.append(
                    tf.reduce_mean(tf.gather(syn0, context_words[:true_size]), axis=0))
            inputs_syn0 = tf.stack(inputs_syn0)

        return inputs_syn0
