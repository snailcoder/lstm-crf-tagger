#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : data_utils.py
# Author            : Yan <yanwong@126.com>
# Date              : 31.03.2020
# Last Modified Date: 09.04.2020
# Last Modified By  : Yan <yanwong@126.com>

import logging
import glob
import collections

import numpy as np
import tensorflow as tf

logging.basicConfig(level=logging.INFO)

def load_w2v(fname, vocab):
  """ Loads pre-trained word vectors.

  Args:
    fname: The pre-trained word vector file.
           Created by word2vec in non-binary mode.
    vocab: The dict of words appearing in training corpus.

  Returns:
    A dict of pre-trained word vectors for each word in vocab.
  """
  word_vecs = {}
  with open(fname, "r") as f:
    header = f.readline()
    vocab_size, layer1_size = list(map(int, header.split()))
    for line in f:
      toks = line.split()
      assert(len(toks) == layer1_size + 1)
      word = toks[0]
      if word in vocab:
        word_vecs[word] = np.array(toks[1:]).astype(np.float)

  return word_vecs

def load_vocab(vocab_file):
  """ Load vocab as an ordered dict.

  Args:
    vocab_file: The vocab file in which each line is a single word.

  Returns:
    An ordered dict of which key is the word and value is id.

  """
  vocab = collections.OrderedDict()

  with tf.io.gfile.GFile(vocab_file, 'r') as f:
    lines = f.readlines()
    for i in range(len(lines)):
      word = lines[i].strip()
      vocab[word] = i
  return vocab

def load_vocab_embeddings(word_vecs, vocab, emb_dim):
  """Load pre-trained word embeddings for words in vocab. For the word that's
     in vocab but there's no corresponding pre-trained embedding, generate a
     embedding randomly for it.

  Args:
    word_vecs: A dict contains pre-trained word embedding. Each word in this
      dict is also in the vocab.
    vocab: An ordered dict of which key is the word and value is id.
    emb_dim: The dimension of word embeddings.

  Returns:
    A word embedding list contains all words in vocab. In addition, it contains
    PAD and UNK embeddings, too.
  """
  embeddings = []
  for word in vocab:
    emb = word_vecs.get(word, None)
    if emb is None:
      emb = np.random.uniform(-0.25, 0.25, emb_dim)
    embeddings.append(emb)

  return np.array(embeddings)

def create_dataset(file_pattern, batch_size):
  """Fetches string values from disk into tf.data.Dataset.

  Args:
    file_pattern: Comma-separated list of file patterns (e.g.
      "/tmp/train_data-?????-of-00100", where '?' acts as a wildcard that
      matches any character).
    batch_size: Batch size.

  Returns:
    A dataset read from TFRecord files.
  """
  data_files = []
  for pattern in file_pattern.split(','):
    data_files.extend(glob.glob(pattern))
  if not data_files:
    logging.fatal('Found no input files matching %s', file_pattern)
  else:
    logging.info('Prefetching values from %d files matching %s',
                 len(data_files), file_pattern)

  dataset = tf.data.TFRecordDataset(data_files)

  def _parse_record(record):
    features = {
        'sentence': tf.io.VarLenFeature(dtype=tf.int64),
        'tags': tf.io.VarLenFeature(dtype=tf.int64)
        }
    parsed_features = tf.io.parse_single_example(record, features)

    sent = tf.sparse.to_dense(parsed_features['sentence'])
    tags = tf.sparse.to_dense(parsed_features['tags'])
    return sent, tags

  dataset = dataset.map(_parse_record)
  dataset = dataset.shuffle(buffer_size=100000, seed=42)
  dataset = dataset.padded_batch(
      batch_size,
      padded_shapes=([-1], [-1]))

  return dataset

