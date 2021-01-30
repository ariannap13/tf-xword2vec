import heapq
import itertools
import collections
import string
import re
import os
import numpy as np
import tensorflow as tf
from functools import partial

# out of vocabulary
OOV_ID = -1

# regex or special tokens
protect_ini = u"\uFF5F"         # "｟"  # U+FF5F
protect_end = u"\uFF60"         # "｠"  # U+FF60
connector = u"\uffed"           # "￭"
spacer = u"\u2581"              # "▁"
regex_code = protect_ini + "[^" + protect_end + "]*" + protect_end
#partial_end = " ￭ . ｟mrk_case_modifier_c｠ "
regex_partial_end = r"( ￭ \. ｟mrk_case_modifier_c｠)|( ￭ \. ｟mrk_begin_case_region_u｠)"

class Word2VecDataset(object):
  """Dataset for generating matrices holding word indices to train Word2Vec 
  models.
  """
  def __init__(self,
               arch='skip_gram',
               algm='negative_sampling',
               batch_size=500,
               max_vocab_size=0,
               min_count=30,
               sample=1e-3,
               window_size=6,
               special_tokens=0,
               focus="",
               out_dir="data/out"):
    """Constructor.
    Args:
      arch: string scalar, architecture ('skip_gram' or 'cbow').
      algm: string scalar: training algorithm ('negative_sampling' or
        'hierarchical_softmax').
      batch_size: int scalar, the returned tensors in `get_tensor_dict` have
        shapes [batch_size, :]. 
      max_vocab_size: int scalar, maximum vocabulary size. If > 0, the top 
        `max_vocab_size` most frequent words are kept in vocabulary.
      min_count: int scalar, words whose counts < `min_count` are not included
        in the vocabulary.
      sample: float scalar, subsampling rate.
      window_size: int scalar, num of words on the left or right side of
        target word within a window.
      special_tokens: int scalar, that is converted to boolean. If zero, there is
        not any special token in the data. Else, the special tokens will be removed,
        considering the special tokens used in pyonmttok tokenizer.
      focus: string scalar, focus word to research. 
        Only lines with this word will be used.
      out_dir: string scalar, contains the output folder. 
        Combined filenames will be created in the same folder.
    """
    self._arch = arch
    self._algm = algm
    self._batch_size = batch_size
    self._max_vocab_size = max_vocab_size
    self._min_count = min_count
    self._sample = sample
    self._window_size = window_size
    self._special_tokens = True if special_tokens else False

    self._iterator_initializer = None
    self._table_words = None
    self._unigram_counts = None
    self._keep_probs = None
    self._corpus_size = None
    self._max_depth = None
    
    self._focus = str(focus)
    self._out_dir = out_dir
    self._comb_filenames = ""

    
  @property
  def iterator_initializer(self):
    return self._iterator_initializer

  @property
  def table_words(self):
    return self._table_words

  @property
  def unigram_counts(self):
    return self._unigram_counts

  @property
  def keep_probs(self):
    return self._keep_probs

  @property
  def focus(self):
    return self._focus
  
  @focus.setter
  def focus(self, v):
    self._focus = str(v)
    
  @property
  def comb_filenames(self):
    return self._comb_filenames

  def remove_tokens(self, dline):
    dline = dline.replace("｟mrk_case_modifier_c｠ ", "")
    dline = dline.replace(" ｟mrk_end_case_region_u｠", "")
    dline = dline.replace("｟mrk_begin_case_region_u｠ ", "")
    dline = dline.replace(" ￭ - ￭ ", "-")
    dline = re.sub(regex_code, "", dline)
    dline = dline.replace('" ', "")
    dline = dline.replace(" " + connector, "")
    dline = dline.replace(spacer, "")
    dline = dline.replace(" " * 2, " ")
    # remove numbers
    #dline = re.sub(r'\d+', '', dline)
    dline = re.sub(r'[…\”»«@\'\(\)\[\]\{\}]', '', dline)
    dline = re.sub(r'[^\x00-\x7F\x80-\xFF\u0100-\u017F\u0180-\u024F\u1E00-\u1EFF]', '', dline) 
    dline = dline.replace(" " * 2, " ")
    if not set(string.ascii_lowercase).intersection(dline.lower()):
      dline = ""
    return dline
 
  
  def set_comb_filenames(self):
    self._focus = self._focus.strip()
    if self._focus == "":
      if self._special_tokens:
        comb_filenames = "clean_filenames.txt"
      else: 
        comb_filenames = None
    else:
      if self._special_tokens:
        comb_filenames = "clean_filenames_focus_" +  self._focus + ".txt"
      else:
        comb_filenames = "full_filenames_focus_" +  self._focus + ".txt"
    return os.path.join(self._out_dir, comb_filenames)
    

  def _build_raw_vocab(self, filenames):
    """Builds raw vocabulary.
    Args:
      filenames: list of strings, holding names of text files.
    Returns: 
      raw_vocab: a list of 2-tuples holding the word (string) and count (int),
        sorted in descending order of word count. 
    """
    self._comb_filenames = self.set_comb_filenames()
    map_open = partial(open, encoding="utf-8")
    lines = itertools.chain(*map(map_open, filenames))
    raw_vocab = collections.Counter()
    if self._special_tokens or len(self._focus) > 0:
      fcomb = open(self._comb_filenames,"w", encoding="utf-8")
    for line in lines:
        #paragraph = line.split(partial_end)
        paragraph = re.split(regex_partial_end, line)
        # broken paragraph
        for sline in paragraph:
            if sline:
              if self._special_tokens:
                sline = self.remove_tokens(sline)
                sline = sline.strip()
              if len(sline) == 0: 
                continue
              if len(self._focus) == 0:
                fcomb.write(sline)
              elif not self._focus in sline:
                continue
              else:
                fcomb.write(sline)
              # update vocab
              raw_vocab.update(sline.strip().split())
            #else:
            #  print("None")
              
    if self._special_tokens or len(self._focus) > 0: 
      fcomb.close()
    raw_vocab = raw_vocab.most_common()
    if self._max_vocab_size > 0:
      raw_vocab = raw_vocab[:self._max_vocab_size]
    return raw_vocab

  def build_vocab(self, filenames):
    """Builds vocabulary.
    Has the side effect of setting the following attributes:   
    - table_words: list of string, holding the list of vocabulary words. Index
        of each entry is the same as the word index into the vocabulary.
    - unigram_counts: list of int, holding word counts. Index of each entry
        is the same as the word index into the vocabulary.
    - keep_probs: list of float, holding words' keep prob for subsampling. 
        Index of each entry is the same as the word index into the vocabulary.
    - corpus_size: int scalar, effective corpus size.
    Args:
      filenames: list of strings, holding names of text files.
    """
    raw_vocab = self._build_raw_vocab(filenames)
    raw_vocab = [(w, c) for w, c in raw_vocab if c >= self._min_count]
    self._corpus_size = sum(list(zip(*raw_vocab))[1])

    self._table_words = []
    self._unigram_counts = []
    self._keep_probs = []
    for word, count in raw_vocab:
      frac = count / float(self._corpus_size)
      keep_prob = (np.sqrt(frac / self._sample) + 1) * (self._sample / frac)
      keep_prob = np.minimum(keep_prob, 1.0)
      self._table_words.append(word)
      self._unigram_counts.append(count)
      self._keep_probs.append(keep_prob)

  def _build_binary_tree(self, unigram_counts):
    """Builds a Huffman tree for hierarchical softmax. Has the side effect
    of setting `max_depth`.
    Args:
      unigram_counts: list of int, holding word counts. Index of each entry
        is the same as the word index into the vocabulary.
    Returns:
      codes_points: an int numpy array of shape [vocab_size, 2*max_depth+1]
        where each row holds the codes (0-1 binary values) padded to
        `max_depth`, and points (non-leaf node indices) padded to `max_depth`,
        of each vocabulary word. The last entry is the true length of code
        and point (<= `max_depth`).
    """
    vocab_size = len(unigram_counts)
    heap = [[unigram_counts[i], i] for i in range(vocab_size)]
    heapq.heapify(heap)
    for i in range(vocab_size - 1):
      min1, min2 = heapq.heappop(heap), heapq.heappop(heap)
      heapq.heappush(heap, [min1[0] + min2[0], i + vocab_size, min1, min2])

    node_list = []
    max_depth, stack = 0, [[heap[0], [], []]]
    while stack:
      node, code, point = stack.pop()
      if node[1] < vocab_size:
        node.extend([code, point, len(point)])
        max_depth = np.maximum(len(code), max_depth)
        node_list.append(node)
      else:
        point = np.array(list(point) + [node[1]-vocab_size])
        stack.append([node[2], np.array(list(code)+[0]), point])
        stack.append([node[3], np.array(list(code)+[1]), point])

    node_list = sorted(node_list, key=lambda items: items[1])
    codes_points = np.zeros([vocab_size, max_depth*2+1], dtype=np.int32)
    for i in range(len(node_list)):
      length = node_list[i][4] # length of code or point
      codes_points[i, -1] = length
      codes_points[i, :length] = node_list[i][2] # code
      codes_points[i, max_depth:max_depth+length] = node_list[i][3] # point
    self._max_depth = max_depth
    return codes_points

  def _prepare_inputs_labels(self, tensor):
    """Set shape of `tensor` according to architecture and training algorithm,
    and split `tensor` into `inputs` and `labels`.
    Args:
      tensor: rank-2 int tensor, holding word indices for prediction inputs
        and prediction labels, returned by `generate_instances`.
    Returns:
      inputs: rank-2 int tensor, holding word indices for prediction inputs. 
      labels: rank-2 int tensor, holding word indices for prediction labels.
    """
    if self._arch == 'skip_gram':
      if self._algm == 'negative_sampling':
        tensor.set_shape([self._batch_size, 2])
      else:
        # hierarchical_softmax
        tensor.set_shape([self._batch_size, 2 * self._max_depth + 2])
      inputs = tensor[:, :1]
      labels = tensor[:, 1:]
    else:
      # cbow
      if self._algm == 'negative_sampling':
        tensor.set_shape([self._batch_size, 2 * self._window_size + 2])
      else:
        # hierarchical_softmax
        tensor.set_shape([self._batch_size, 
            2 * self._window_size + 2 * self._max_depth + 2])
      inputs = tensor[:, : 2 * self._window_size + 1]
      labels = tensor[:, 2 * self._window_size + 1:]
    return inputs, labels

  def get_tensor_dict(self, filenames, epochs):
    """Generates tensor dict mapping from tensor names to tensors.
    Args:
      filenames: list of strings, holding names of text files.
      epochs: int scalar, num times the dataset is iterated. 
      
    Returns:
      tensor_dict: a dict mapping from tensor names to tensors with shape being:
        when arch=='skip_gram', algm=='negative_sampling'
          inputs: [N],                    labels: [N]
        when arch=='cbow', algm=='negative_sampling'
          inputs: [N, 2*window_size+1],   labels: [N]
        when arch=='skip_gram', algm=='hierarchical_softmax'
          inputs: [N],                    labels: [N, 2*max_depth+1]
        when arch=='cbow', algm=='hierarchical_softmax'
          inputs: [N, 2*window_size+1],   labels: [N, 2*max_depth+1]
        progress: [N], the percentage of sentences covered so far to one epoch.
                  Used to compute learning rate.
        epoch: [N], the epoch associated to the sentence.
    """
    table_words = self._table_words
    unigram_counts = self._unigram_counts
    keep_probs = self._keep_probs
    if not table_words or not unigram_counts or not keep_probs:
      raise ValueError('`table_words`, `unigram_counts`, and `keep_probs` must',
          'be set by calling `build_vocab()`')

    if self._algm == 'hierarchical_softmax':
      codes_points = tf.constant(self._build_binary_tree(unigram_counts))
    elif self._algm == 'negative_sampling':
      codes_points = None
    else:
      raise ValueError('algm must be hierarchical_softmax or negative_sampling')
   
    table_words = tf.contrib.lookup.index_table_from_tensor(
        tf.constant(table_words), default_value=OOV_ID)
    keep_probs = tf.constant(keep_probs)

    if self._special_tokens or self._focus != "":
      # name was resolved building vocabulary
      comb_file = [self._comb_filenames]
    else:
      comb_file = filenames
      
    num_sents = sum([len(list(open(fn, encoding="utf-8"))) for fn in comb_file])
    
    assert num_sents > 1
    # tried to remove epoch but interaction failed.
    tot_sents = epochs * num_sents
    
    a_zip = tf.data.TextLineDataset(comb_file).repeat(epochs)
    b_zip = tf.range(1, 1 + tot_sents) / tot_sents
    c_zip = tf.repeat(tf.range(1, 1 + epochs), num_sents)
    
    dataset = tf.data.Dataset.zip((a_zip,
                                   tf.data.Dataset.from_tensor_slices(b_zip),
                                   tf.data.Dataset.from_tensor_slices(c_zip)))
        
    dataset = dataset.map(lambda sent, progress, epoch: 
        (get_word_indices(sent, table_words), progress, epoch))
      
    dataset = dataset.map(lambda indices, progress, epoch: 
        (subsample(indices, keep_probs), progress, epoch))
      
    dataset = dataset.filter(lambda indices, progress, epoch: 
        tf.greater(tf.size(indices), 1))

    dataset = dataset.map(lambda indices, progress, epoch: (
      generate_instances(indices, self._arch, self._window_size, codes_points),
                         progress, epoch))
    
    dataset = dataset.map(lambda instances, progress, epoch: (
        instances, tf.fill(tf.shape(instances)[:1], progress),
                   tf.fill(tf.shape(instances)[:1], epoch)))

    dataset = dataset.flat_map(lambda instances, progress, epoch: 
        tf.data.Dataset.from_tensor_slices((instances, progress, epoch)))
      
    dataset = dataset.batch(self._batch_size, drop_remainder=True)
    
    iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
    
    self._iterator_initializer = iterator.initializer
    self._iterator_next = iterator.get_next()
    tensor, progress, epoch = self._iterator_next
    progress.set_shape([self._batch_size])
    epoch.set_shape([self._batch_size])

    inputs, labels = self._prepare_inputs_labels(tensor)
    if self._arch == 'skip_gram':
      inputs = tf.squeeze(inputs, axis=1)
    if self._algm == 'negative_sampling':
      labels = tf.squeeze(labels, axis=1)
            
    return {'inputs': inputs, 'labels': labels,
            'progress': progress, 'epoch': epoch} 
  

def get_word_indices(sent, table_words):
  """Converts a sentence into a list of word indices.
  Args:
    sent: a scalar string tensor, a sentence where words are space-delimited.
    table_words: a `HashTable` mapping from words (string tensor) to word 
      indices (int tensor).
  
  Returns:
    indices: rank-1 int tensor, the word indices within a sentence.
  """
  words = tf.string_split([sent]).values
  indices = tf.cast(table_words.lookup(words), tf.int32) # 32
  return indices


def subsample(indices, keep_probs):
  """Filters out-of-vocabulary words and then applies subsampling on words in a 
  sentence. Words with high frequencies have lower keep probs.
  Args:
    indices: rank-1 int tensor, the word indices within a sentence.
    keep_probs: rank-1 float tensor, the prob to drop the each vocabulary word. 
  Returns:
    indices: rank-1 int tensor, the word indices within a sentence after 
      subsampling.
  """
  indices = tf.boolean_mask(indices, tf.not_equal(indices, OOV_ID))
  keep_probs = tf.gather(keep_probs, indices)
  randvars = tf.random.uniform(tf.shape(keep_probs), 0, 1)
  indices = tf.boolean_mask(indices, tf.less(randvars, keep_probs))
  return indices


def generate_instances(indices, arch, window_size, codes_points=None):
  """Generates matrices holding word indices to be passed to Word2Vec models 
  for each sentence. The shape and contents of output matrices depends on the 
  architecture ('skip_gram', 'cbow') and training algorithm ('negative_sampling'
  , 'hierarchical_softmax').
  It takes as input a list of word indices in a subsampled-sentence, where each
  word is a target word, and their context words are those within the window 
  centered at a target word. For skip gram architecture, `num_context_words` 
  instances are generated for a target word, and for cbow architecture, a single
  instance is generated for a target word.
  If `codes_points` is not None ('hierarchical softmax'), the word to be 
  predicted (context word for 'skip_gram', and target word for 'cbow') are 
  represented by their 'codes' and 'points' in the Huffman tree (See 
  `_build_binary_tree`). 
  Args:
    indices: rank-1 int tensor, the word indices within a sentence after
      subsampling.
    arch: scalar string, architecture ('skip_gram' or 'cbow').
    window_size: int scalar, num of words on the left or right side of
      target word within a window.
    codes_points: None, or an int tensor of shape [vocab_size, 2*max_depth+1] 
      where each row holds the codes (0-1 binary values) padded to `max_depth`, 
      and points (non-leaf node indices) padded to `max_depth`, of each 
      vocabulary word. The last entry is the true length of code and point 
      (<= `max_depth`).
    
  Returns:
    instances: an int tensor holding word indices, with shape being
      when arch=='skip_gram', algm=='negative_sampling'
        shape: [N, 2]
      when arch=='cbow', algm=='negative_sampling'
        shape: [N, 2*window_size+2]
      when arch=='skip_gram', algm=='hierarchical_softmax'
        shape: [N, 2*max_depth+2]
      when arch=='cbow', algm='hierarchical_softmax'
        shape: [N, 2*window_size+2*max_depth+2]
  """
  def per_target_fn(index, init_array):
    reduced_size = tf.random.uniform([], maxval=window_size, dtype=tf.int32)
    left = tf.range(tf.maximum(index - window_size + reduced_size, 0), index)
    right = tf.range(index + 1, 
        tf.minimum(index + 1 + window_size - reduced_size, tf.size(indices)))
    context = tf.concat([left, right], axis=0)
    context = tf.gather(indices, context)

    if arch == 'skip_gram':
      window = tf.stack([tf.fill(tf.shape(context), indices[index]), 
                        context], axis=1)
    elif arch == 'cbow':
      true_size = tf.size(context)
      window = tf.concat([tf.pad(context, [[0, 2*window_size-true_size]]), 
                          [true_size, indices[index]]], axis=0)
      window = tf.expand_dims(window, axis=0)
    else:
      raise ValueError('architecture must be skip_gram or cbow.')

    if codes_points is not None:
      window = tf.concat([window[:, :-1], 
                          tf.gather(codes_points, window[:, -1])], axis=1)
    return index + 1, init_array.write(index, window)

  size = tf.size(indices)
  init_array = tf.TensorArray(tf.int32, size=size, infer_shape=False)
  _, result_array = tf.while_loop(lambda i, ta: i < size,
                                  per_target_fn, 
                                  [0, init_array], back_prop=False)

  instances = tf.cast(result_array.concat(), tf.int64)  # 64
  return instances
