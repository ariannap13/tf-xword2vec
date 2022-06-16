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


def _full_loss_torch(input, label, vocab_len, weights=None):
  """Builds the full loss. The batch_size is forced as 1 since we are post-training.

  Args:
    input: int of shape [batch_size] => 1 (skip_gram)
    label: int of shape [batch_size] => 1

  Returns:
    loss: float tensor of shape [batch_size, vocab_size].
  """
  vocab = set(range(vocab_len))

  syn0 = weights[0]
  syn1 = weights[1]

  contexts = []

  vocab_tmp = vocab.copy()
  vocab_tmp.remove(label) # remove label which is already true label from other labels
  contexts.append(np.array(list(vocab_tmp)))

  inputs_syn0 = torch.index_select(syn0, 0, torch.from_numpy(np.array(input)))
  context_syn1 = torch.index_select(syn1, 0, torch.from_numpy(contexts[0]))
  true_syn1 = torch.index_select(syn1, 0, torch.from_numpy(np.array(label)))

  true_logits = torch.sum(torch.multiply(inputs_syn0, true_syn1), dim=1)
  true_logits.requires_grad_()

  context_logits = torch.einsum('ijk,ikl->il', inputs_syn0.unsqueeze(1),
      context_syn1.unsqueeze(0).permute(0, 2, 1))
  context_logits.requires_grad_()

  loss = torch.nn.BCEWithLogitsLoss(reduction='none')

  true_cross_entropy = loss(true_logits, torch.ones_like(true_logits))
  #true_cross_entropy.backward()
  context_cross_entropy = loss(context_logits, torch.zeros_like(context_logits))
  #context_cross_entropy.backward()

  loss = torch.concat(
      [true_cross_entropy.unsqueeze(0), context_cross_entropy], dim=1)
  return loss


def _true_loss_torch(input, label, vocab_len, weights=None):
  """Builds the true loss. The batch_size is forced as 1 since we are post-training.

  Args:
    input: int of shape [batch_size] => 1 (skip_gram)
    label: int of shape [batch_size] => 1

  Returns:
    loss: float tensor of shape [batch_size, 1].
  """
  vocab = set(range(vocab_len))

  syn0 = weights[0]
  syn1 = weights[1]

  inputs_syn0 = torch.index_select(syn0, 0, torch.from_numpy(np.array(input)))
  true_syn1 = torch.index_select(syn1, 0, torch.from_numpy(np.array(label)))

  true_logits = torch.sum(torch.multiply(inputs_syn0, true_syn1), dim=1)
  true_logits.requires_grad_()

  loss = torch.nn.BCEWithLogitsLoss(reduction='none')

  true_cross_entropy = loss(true_logits, torch.ones_like(true_logits))
  #true_cross_entropy.backward()

  loss = true_cross_entropy.unsqueeze(0)
  return loss


def _negative_sampling_loss_torch(inputs, labels, batch_size, unigram_counts, negatives, weights):
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

def get_key(val):
  for key, value in vocab.items():
    if val == value:
      return key


def post_training(k, vocab, dataset, list_index_weat, loss, weights, diz_gradients, hessian_diz):
  vocab_len = len(vocab)
  i = 0
  for step, training_point in enumerate(dataset):
    input = training_point[0][0]
    if input in list_index_weat:
      input = training_point[0][0]
      label = training_point[0][1]
      nsent = training_point[0][2]
      true_loss = _true_loss_torch(input, label, vocab_len, weights)
      true_loss.sum().backward(inputs=weights[0], create_graph=True, retain_graph=True)
      grad = torch.autograd.grad(true_loss.sum(), weights[0], create_graph=True, retain_graph=True)[0]

      hessian = []
      for w in grad[input]:
        hess_i = torch.autograd.grad(w, weights[0], retain_graph=True)[0]
        hessian.append(hess_i[input])

      hess = torch.stack(hessian)

      hess_target = hess.numpy()
      grad_target = grad.detach().numpy()[input]

      diz_key = (inv_vocab[input], inv_vocab[label], nsent)

      if diz_key not in diz_gradients:
        diz_gradients[diz_key] = grad_target.copy()
      else:
        diz_gradients[diz_key] += grad_target.copy()

      if inv_vocab[input] not in hessian_diz:
        hessian_diz[inv_vocab[input]] = hess_target.copy()
      else:
        hessian_diz[inv_vocab[input]] += hess_target.copy()

      i+=1

      if i%10==0:
        print(i) # progress

  with open(os.path.join('dict_gradients_'+str(k)+'.pickle'), 'wb') as handle_grad:
    pickle.dump(diz_gradients, handle_grad, protocol=pickle.HIGHEST_PROTOCOL)

  with open(os.path.join('dict_hessians_'+str(k)+'.pickle'), 'wb') as handle_hes:
    pickle.dump(hessian_diz, handle_hes, protocol=pickle.HIGHEST_PROTOCOL)


def compute_g(A, B, c, get_emb):
  """
  compute g measure which is part of effect size.

  A: set of words of a given pole (attribute set)
  B: set of words of a the opposite pole (attribute set)
  c: word of which the embedding is compared to word embeddings of terms in A and B
  get_emb: function allowint to get the w2v embedding given a word

  return: g measure
  """
  mean_cos_a = 0
  mean_cos_b = 0
  for word_a in A:
    mean_cos_a += 1 - spatial.distance.cosine(get_emb(c), get_emb(word_a))
  mean_cos_a /= len(A)
  for word_b in B:
    mean_cos_b += 1 - spatial.distance.cosine(get_emb(c), get_emb(word_b))
  mean_cos_b /= len(B)
  g = mean_cos_a - mean_cos_b
  return g


def effect_size(S, T, A, B, get_emb):
  """
  Compute effect size.

  S: first target set
  T: second target set
  A: first attribute set
  B: second attribute set
  get_emb: function allowing to get the w2v embedding given a word

  return: effect size measure
  """
  mean_g_s = 0
  mean_g_t = 0
  for word_s in S:
    mean_g_s += compute_g(A, B, word_s, get_emb)
  mean_g_s /= len(S)
  for word_t in T:
    mean_g_t += compute_g(A, B, word_t, get_emb)
  mean_g_t /= len(T)
  list_g_x = []
  for word_x in S+T:
    g_x = compute_g(A, B, word_x, get_emb)
    list_g_x.append(g_x)
  std = statistics.stdev(list_g_x)
  effect_size = (mean_g_s-mean_g_t)/std
  return effect_size


def get_emb(wv, word):
    """
    Function allowing to get the embedding of a given word.

    word: word of interest
    return: word embedding
    """
    return wv._syn0_final[wv._rev_vocab[word]]


def get_sent_grad(diz_gradients, sent_id):
  """
  Computes the gradient for the sentence to be removed by aggregating the values in diz_gradients.

  diz_gradients: dictionary of type {(target, context_word, nsent): gradient}
  sent_id: id of sentence in the corpus
  return: dictionary of sentence gradient by target word
  """
  diz_grad_sent = {}
  for key in diz_gradients:
    target = key[0]
    nsent = key[2]
    if int(nsent) == sent_id: # sum over the gradients of instances which are in the sentence I'm removing
      if target not in diz_grad_sent:
        diz_grad_sent[target] = diz_gradients[key].copy()
      else:
        diz_grad_sent[target] += diz_gradients[key].copy()

  return diz_grad_sent


def get_full_grad(diz_gradients):
  """
  Computes the full gradient by aggregating the values in diz_gradients.

  diz_gradients: dictionary of type {(target, context_word, nsent): gradient}
  return: dictionary of full gradient by target word
  """
  diz_grad_full = {}
  for key in diz_gradients:
    target = key[0]
    if target not in diz_grad_full:
      diz_grad_full[target] = diz_gradients[key].copy()
    else:
      diz_grad_full[target] += diz_gradients[key].copy()

  return diz_grad_full


def get_perturbed_emb_sent(wv, sent_text, diz_gradients, hessian_diz, sent_id):
  """
  Function allowing to get the approximate version of embedding of a given word
  though influence functions.

  wv: WordVector class object
	sent_text: sentence of interest
  diz_gradients: dictionary of gradients {(target, context_word, nsent): gradient}
  sent_id: id of sentence in the corpus
	return: approximated word embedding dictionary
  """
  diz_grad_sent = get_sent_grad(diz_gradients, sent_id)

  perturbed_emb = {} # dictionary {word: emb}
  for word in S+T+A+B+most_common_words:
    emb = get_emb(word)
    if ' '+word+' ' in sent_text: # single entity
      V = len(wv._vocab)
      if word in diz_grad_sent:
        grad_sent = diz_grad_sent[word] # gradient for the word of interest
        hessian = np.linalg.inv(hessian_diz[word])
      else: # in case the term is appearing only in the sentence to be removed
        grad_sent = np.zeros(300) # 300 is the hidden size
        hessian = np.zeros((300,300))
      # decide whether grad_sent should be multiplied by V or not - technically yes to fully follow Brunet et al.
      emb = emb + (1/V)*np.dot(hessian, grad_sent)
    perturbed_emb[word] = emb

  return perturbed_emb


def get_emb_pert(word):
  """
  Apply dictionary to given word to get the embedding.

  word: word of interest
  return: word embedding
  """
  return perturbed_emb[word]


def get_sim_matrix(S,T,A,B,most_common_words,inv_vocab,get_emb):
  """
  Get similarity matrix for target and attribute sets.

  S,T: target sets
  A,B: attribute sets
  get_emb: method to get word embedding given a word
  return: similarity matrix
  """
  sim_matrix_list = [] # create empty similarity list

  for target_word in S+T:
    list_cossim = []
    for attr_word in A+B+most_common_words:
      if attr_word in inv_vocab and target_word in inv_vocab:
        list_cossim.append(1 - spatial.distance.cosine(get_emb(target_word), get_emb(attr_word)))
      else:
        list_cossim.append(np.nan)
    sim_matrix_list.append(list_cossim)

  sim_matrix = pd.DataFrame(sim_matrix_list, columns = [attr_word for attr_word in A+B+most_common_words])
  sim_matrix.index = [target_word for target_word in S+T]

  return sim_matrix


def get_sim_perturbed(k, data, S, T, A, B, most_common_words, wv, diz_gradients, diz_hessian, get_emb_pert):

  list_sim_sent = []
  print('k value: ', str(k))
  for sent_id in range(len(data)):
    sent_text = data[sent_id][0]+' '

    # check my sentence contains at least one WEAT word, otherwise there is no contribution to the perturbed embedding
    first=True
    for word in S+T+A+B:
      if ' '+word+' ' in sent_text: # word as single occurrence (not part of another word), eg. otherwise "the" is selected because it contains "he"
        if first==True: # I consider a sentence if it has at least one occurrence of WEAT words
          # define pertubed embedding
          perturbed_emb = get_perturbed_emb_sent(wv, sent_text, diz_gradients_0, hessian_diz_0, sent_id)
          # effect size perturbed
          sim_matrix_pert = get_sim_matrix(S, T, A, B, most_common_words, get_emb_pert)

          print("Sentence: ", sent_id, sent_text)
          print("Similarity matrix perturbed corpus:\n", sim_matrix_pert)

          complete_list = []
          for target in S+T:
              list_target = sim_matrix_pert.loc[target].values.flatten().tolist()
              complete_list.append(list_target)

          list_sim_sent.append([sent_id, complete_list])

          first=False

  return list_sim_sent


def compute_corr(k, target_set, list_sim_sent, list_sim_sent_retrain):
  print("Num. of k: ", str(k))
  mean_list = []
  for target in target_set:
     mean_list.append(0)
  mean_list = np.array(mean_list)
  for sent in list_sim_sent:
    sent_id = sent[0]
    for sent_retrain in list_sim_sent_retrain:
      if sent_id == sent_retrain[0]
        corr = []
        for i in range(len(target_set)):
            corr.append(scipy.stats.pearsonr(np.array(sent[1][i]),
                                            np.array(sent_retrain[1][i]))[0])
        mean_list += np.array(corr)
        print('num. sent: ', sent_id)
        for i in range(len(target_set)):
            print('correlation of '+target_set[i]+': ', corr[i])

  mean_list = mean_list/len(list_sim_sent)

  for i in range(len(target_set)):
      print('mean correlation of '+target_set[i]+': ', mean_list[i])


def get_variation_sim_matrix(k, data, S, T, A, B, wv, diz_gradients, hessian_diz):
  print('k value: ', str(k))
  for sent_id in range(len(data)):
    sent_text = data[sent_id][0]+' '

    # check my sentence contains at least one WEAT word, otherwise there is no contribution to the perturbed embedding
    first=True
    for word in S+T+A+B:
      if ' '+word+' ' in sent_text: # word as single occurrence (not part of another word), eg. otherwise "the" is selected because it contains "he"
        if first==True: # I consider a sentence if it has at least one occurrence of WEAT words
          # define pertubed embedding
          perturbed_emb = get_perturbed_emb_sent(wv, sent_text, diz_gradients, hessian_diz, sent_id)

          # effect size perturbed
          sim_matrix_pert = get_sim_matrix(S, T, A, B, [], get_emb_pert)
          #print(sim_matrix_pert)

          print("Sentence: ", sent_id, sent_text)
          print("Diff. similarity matrices:\n", sim_matrix_pert - sim_matrix_full)

          first=False
