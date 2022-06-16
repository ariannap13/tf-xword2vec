import nltk
nltk.download('punkt')
import os
import sys
import pandas as pd
import torch
import numpy as np
import pickle
import collections
from scipy import spatial
import statistics
import collections
from sklearn.feature_extraction.text import CountVectorizer


def get_vocab(text):
  raw_vocab = collections.Counter()
  for line in text:
    raw_vocab.update(line)
  raw_vocab = dict(raw_vocab.most_common())
  return raw_vocab


def subsampling(vocab, whitelist=[], rate=0.0001):

  # A storage to store tokens after subsampling
  new_tokens = whitelist.copy()
  # tokens is a list of word indexes from original text
  np.random.seed(0)
  for word in vocab.keys():
      frac = vocab[word]/len(vocab.keys())
      prob = (np.sqrt(frac/rate) + 1) * (rate/frac)
      if np.random.random() < prob:
          new_tokens.append(word)
  return list(set(new_tokens))


def low_freq(vocab, whitelist=[], min_freq=1):
  """
  keep words with frequency >= min_freq
  """
  return list(set([x for x in vocab.keys() if vocab[x] >= min_freq] + whitelist.copy()))


def apply_reduction(text, vocab, whitelist, min_freq, sampling_rate):

  high_freq = low_freq(vocab, whitelist.copy(), min_freq)

  new_text = [[x for x in sent if x in high_freq] for sent in text]
  new_vocab = get_vocab(new_text)

  subsamples = subsampling(new_vocab, whitelist.copy(), sampling_rate)

  new_new_text = [[x for x in sent if x in subsamples] for sent in new_text]

  return new_text, new_new_text


def split_given_size(a, size):
    return np.split(a, np.arange(size,len(a),size))


def create_skipgram(text, window, whitelist=[], min_freq=1, sampling_rate=1e-3, epochs=1, batch_size=1):
  """
  returns: list of [target, context, sentence number], vocab
  """
  data = []
  vocab = get_vocab(text)
  text, new_text = apply_reduction(text, vocab, whitelist.copy(), min_freq, sampling_rate)
  vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b", analyzer=lambda x: x)
  vectorizer.fit_transform(text)
  for nsent, sentence in enumerate(new_text):
    for i, t in enumerate(sentence):
      contexts = list(range(i-window, i + window+1))
      contexts = [c for c in contexts if c >= 0 and c != i and c < len(sentence)]
      for c in contexts:
        data.append([vectorizer.vocabulary_[t], vectorizer.vocabulary_[sentence[c]], nsent])

  data = split_given_size(data, batch_size)
  data = [x for x in data if len(x)==batch_size]
  data = data * epochs

  vcount = get_vocab(text)
  vocab = dict(sorted(vectorizer.vocabulary_.items(), key=lambda item: item[1], reverse=False))
  unigram_counts = [vcount[x] for x in vocab]

  return data, unigram_counts, vocab, {v: k for k, v in vocab.items()}
