import numpy as np
import torch
import torch.utils.data
from typing import List, \
    Union
from dataset_torch import split_given_size

def fixed_unigram_candidate_sampler(
    true_classes: Union[np.array, torch.Tensor],
    num_samples: int,
    unigrams: List[Union[int, float]],
    distortion: float = 1.):

    if isinstance(true_classes, torch.Tensor):
        true_classes = true_classes.detach().cpu().numpy()
    if true_classes.shape[0] != num_samples:
        raise ValueError('true_classes must be a 2D matrix with shape (num_samples, num_true)')
    unigrams = np.array(unigrams)
    if distortion != 1.:
        unigrams = unigrams.astype(np.float64) ** distortion
    # print('unigrams:', unigrams)
    indices = np.arange(num_samples)
    result = np.zeros(num_samples, dtype=np.int64)
    while len(indices) > 0:
        # print('len(indices):', len(indices))
        sampler = torch.utils.data.WeightedRandomSampler(unigrams, len(indices))
        candidates = np.array(list(sampler))
        candidates = np.reshape(candidates, (len(indices), 1))
        # print('candidates:', candidates)
        # print('true_classes:', true_classes[indices, :])
        result[indices] = candidates.T
        mask = (candidates == true_classes[indices, :])
        mask = mask.sum(1).astype(np.bool)
        # print('mask:', mask)
        indices = indices[mask]
    return result


class Word2VecModel(torch.nn.Module):

    def __init__(self,
                unigram_counts,
                hidden_size=300,
                batch_size=256,
                negatives=5,
                power=0.75,
                alpha=0.002,
                random_seed=0):

      super(Word2VecModel, self).__init__()
      self._unigram_counts = unigram_counts
      self._hidden_size = hidden_size
      self._vocab_size = len(unigram_counts)
      self._batch_size = batch_size
      self._negatives = negatives
      self._power = power
      self._alpha = alpha
      self._random_seed = random_seed
      self._all_losses = {}

      self._input_size = self._vocab_size

      # syn0
      torch.manual_seed(self._random_seed)
      syn0 = torch.empty(self._vocab_size, self._hidden_size)
      self.syn0 = torch.nn.Parameter(data=torch.nn.init.uniform_(syn0, a=(-0.5/self._hidden_size), b=(0.5/self._hidden_size)), requires_grad=True)

      # syn1
      torch.manual_seed(self._random_seed)
      syn1 = torch.empty(self._input_size, self._hidden_size)
      self.syn1 = torch.nn.Parameter(data=torch.nn.init.uniform_(syn1, a=-0.1, b=0.1), requires_grad=True)


    def forward(self, inputs, labels, batch_size, unigram_counts, negatives, weights):
      loss = self._negative_sampling_loss_torch(inputs, labels, batch_size, unigram_counts, negatives, weights)
      return loss

    def _negative_sampling_loss_torch(self, inputs, labels, batch_size, unigram_counts, negatives, weights=None):
      """Builds the negative sampling loss.

      Args:
        input: int of shape [batch_size] => 1 (skip_gram)
        label: int of shape [batch_size] => 1
        batch_size: batch size chosen
        unigram_counts: list of sorted counts of words in vocab
        negatives: number of negatives to consider
        weights: list of weights, syn0 and syn1 matrices

      Returns:
        loss: float tensor of shape [batch_size, vocab_size].
      """
      if not weights:
        syn0 = self.syn0
        syn1 = self.syn1
      else:
        syn0 = weights[0]
        syn1 = weights[1]

      torch.manual_seed(np.array(inputs).mean())
      true_classes_array = torch.unsqueeze(torch.tensor(np.repeat(labels, negatives)), 1)
      #print(true_classes_array.shape)
      sampled_values = fixed_unigram_candidate_sampler(true_classes = true_classes_array,
                                                        num_samples = negatives*batch_size,
                                                        unigrams = unigram_counts,
                                                        distortion = 0.75)
      sampled_values = split_given_size(sampled_values, batch_size)
      sampled_values = np.array([x for x in sampled_values if len(x)==batch_size])
      sampled_values = torch.from_numpy(sampled_values)
      #sampled_mat = torch.reshape(sampled_values, (batch_size, negatives))

      inputs_syn0 = torch.index_select(syn0, 0, torch.from_numpy(np.array(inputs)))
      true_syn1 = torch.index_select(syn1, 0, torch.from_numpy(np.array(labels)))

      #sampled_syn1 = syn1[sampled_values]
      list_sampled_syn1 = []
      for batch in sampled_values:
        sampled_syn1_batch = torch.index_select(syn1, 0, batch)
        list_sampled_syn1.append(sampled_syn1_batch)

      sampled_syn1 = torch.stack(list_sampled_syn1, 0)

      true_logits = torch.sum(torch.multiply(inputs_syn0, true_syn1), dim=1)
      true_logits.requires_grad_()

      sampled_logits = torch.einsum('ijk,ikl->il', inputs_syn0.unsqueeze(1),
                                    sampled_syn1.permute(1, 2, 0))
      sampled_logits.requires_grad_()

      loss = torch.nn.BCEWithLogitsLoss(reduction='none')

      true_cross_entropy = loss(true_logits, torch.ones_like(true_logits))
      sampled_cross_entropy = loss(sampled_logits, torch.zeros_like(sampled_logits))

      loss = torch.concat(
          [true_cross_entropy.unsqueeze(1), sampled_cross_entropy], dim=1)
      return loss
