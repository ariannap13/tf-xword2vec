import numpy as np
import torch
import torch.utils.data
from typing import List, \
    Union
from dataset_torch import split_given_size

import nltk
nltk.download('punkt')
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["figure.figsize"] = (20,22)
sns.set_theme(rc={'figure.figsize': (20,22)})
import os
import sys
import pandas as pd

import torch

import numpy as np
import pickle
import collections

from dataset_torch import create_skipgram
from model_torch import Word2VecModel

BATCH_SIZE = 5
NEGATIVES = 5
EPOCHS = 2
SAMPLING_RATE = 1E-3
MIN_FREQ = 1
WINDOW_SIZE = 3
LEARNING_RATE = 1E-3
# liste_termini_weat
S = ["technology"]
T = ["art"]
A = ["man", "he", "him", "his"]
B = ["woman", "she", "her"]

WEATLIST = S+T+A+B

text = []
with open('/content/nyt_articles_31.txt', 'r') as f:
  text = [nltk.tokenize.word_tokenize(x.strip()) for x in f.readlines()]


data, unigram_counts, vocab, inv_vocab = create_skipgram(text, WINDOW_SIZE, WEATLIST.copy(), MIN_FREQ, SAMPLING_RATE, EPOCHS, BATCH_SIZE)

word2vec = Word2VecModel(unigram_counts,
              hidden_size=300,
              batch_size=BATCH_SIZE,
              negatives=NEGATIVES,
              power=0.75,
              alpha=LEARNING_RATE)
log_per_steps= 100


def train_step(model, opt, inputs, labels):

  opt.zero_grad()
  neg_loss = model._negative_sampling_loss_torch(inputs, labels, word2vec._batch_size, word2vec._unigram_counts, word2vec._negatives)
  neg_loss.sum().backward(create_graph = True, retain_graph = True)

  opt.step()

  return neg_loss
  

average_loss = 0.
# optimizer
optimizer = torch.optim.SGD(word2vec.parameters(), lr=word2vec._alpha)

for step, training_point in enumerate(data):
  inputs = [x[0] for x in training_point]
  labels = [x[1] for x in training_point]
  nsent = [x[2] for x in training_point]
  loss = train_step(word2vec, optimizer, inputs, labels)

print("Training completed")

weights = []
for param in word2vec.parameters():
  weights.append(param)

syn0_final = word2vec.syn0.detach().numpy()
syn1_final = word2vec.syn1.detach().numpy()

np.save('syn0_final_torch', syn0_final)
np.save('syn1_final_torch', syn1_final)
