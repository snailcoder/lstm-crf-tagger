#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : build_tfrecords.py
# Author            : Yan <yanwong@126.com>
# Date              : 07.04.2020
# Last Modified Date: 26.04.2020
# Last Modified By  : Yan <yanwong@126.com>

import os
import collections
import argparse
import logging
import json

import numpy as np
import tensorflow as tf

import special_words

logging.basicConfig(level=logging.INFO)

def _build_vocab(input_file, output_dir):
  """ Load the vocab file created by word2vec to build the vocab dict.

  Args:
    input_file: The processed corpus. Each line contains a pair of word and tag.

  Returns:
    An ordered dict mapping each character to its Id.
  """
  word_cnt = collections.Counter()

  with tf.io.gfile.GFile(input_file, mode='r') as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      word_cnt.update(line.split()[0])

  sorted_items = word_cnt.most_common()

  vocab = collections.OrderedDict()
  vocab[special_words.PAD] = special_words.PAD_ID
  vocab[special_words.UNK] = special_words.UNK_ID

  for i, item in enumerate(sorted_items):
    vocab[item[0]] = i + 2  # 0: PAD, 1: UNK
  
  logging.info('Create vocab with %d words.', len(vocab))

  vocab_file = os.path.join(output_dir, 'vocab.txt')
  with tf.io.gfile.GFile(vocab_file, mode='w') as f:
    f.write('\n'.join(vocab.keys()))

  logging.info('Wrote vocab file to %s', vocab_file)

  return vocab

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(
      int64_list=tf.train.Int64List(value=[int(v) for v in value]))

def _sentence_to_ids(sent, vocab):
  """Helper for converting a sentence (list of words) to a list of ids."""
  ids = [vocab.get(w, special_words.UNK_ID) for w in sent]
  return ids

def _create_serialized_example(sent, tags, vocab):
  """Helper for creating a serialized Example proto."""
  example = tf.train.Example(features=tf.train.Features(feature={
    "sentence": _int64_feature(_sentence_to_ids(sent, vocab)),
    "tags": _int64_feature(tags)
    }))
  return example.SerializeToString()

def _build_dataset(filename, vocab, tag_dict):
  """ Build dataset from the file in Co-NLL NER task 2002 format
      (check corpus/pku_training.txt).

  Args:
    filename: The file contains sentences of which each character has been tagged.
    vocab: A dict mapping each character to Id.
    tag_dict: A dict mapping each tag to Id.

  Returns:
    A list containing serialized examples.

  """
  serialized = []

  with tf.io.gfile.GFile(filename, 'r') as f:
    sent = []
    tags = []
    # tag_dict = {'S': 0, 'B': 1, 'M': 2, 'E': 3}

    for line in f:
      line = line.strip()
      if not line:
        if sent and tags:
          serialized.append(
              _create_serialized_example(sent, tags, vocab))
          sent = []
          tags = []
      else:
        toks = line.split()
        assert(len(toks) >= 2 and toks[1] in tag_dict)
        sent.append(toks[0])
        tags.append(tag_dict[toks[1]])

  return serialized

def _write_shard(filename, dataset, indices):
  """Writes a TFRecord shard."""
  with tf.io.TFRecordWriter(filename) as writer:
    for j in indices:
      writer.write(dataset[j])

def _write_dataset(name, dataset, indices, num_shards, output_dir):
  """Writes a sharded TFRecord dataset.

  Args:
    name: Name of the dataset (e.g. "train").
    dataset: List of serialized Example protos.
    indices: List of indices of 'dataset' to be written.
    num_shards: The number of output shards.
  """
  borders = np.int32(np.linspace(0, len(indices), num_shards + 1))
  for i in range(num_shards):
    filename = os.path.join(
        output_dir, '%s-%.5d-of-%.5d' % (name, i, num_shards))
    shard_indices = indices[borders[i]:borders[i + 1]]
    _write_shard(filename, dataset, shard_indices)
    logging.info('Wrote dataset indices [%d, %d) to output shard %s',
                 borders[i], borders[i + 1], filename)

def main():
  parser = argparse.ArgumentParser(
      description='Make processed corpus datasets.')

  parser.add_argument(
      'input_file',
      help='Each character and if tag appear on their own line.')
  parser.add_argument('output_dir', help='The output directory.')
  parser.add_argument('tag_dict',
                      help='Json-format tag dict file. For example,'
                          ' for word segmentation task, the dict'
                          ' contains "S", "B", "M", "E"; for NER task,'
                          ' the dict contains "O", "B-PER", "I-PER",'
                          ' "B-ORG", "I-ORG", "B-LOC", "I-LOC".'
                          ' Check corpus/seg_tags.json and'
                          ' corpus/ner_tags.json.')
  parser.add_argument(
      '-validation_percentage', type=float, default=0.1,
      help='Percentage of the training data used for validation.')
  parser.add_argument('-train_shards', type=int, default=100,
                      help='Number of output shards for the training set.')
  parser.add_argument('-validation_shards', type=int, default=1,
                      help='Number of output shards for the validation set.')

  args = parser.parse_args()

  if not tf.io.gfile.isdir(args.output_dir):
    tf.io.gfile.makedirs(args.output_dir)

  with open(args.tag_dict, 'r') as f:
    tag_dict = json.load(f)

  vocab = _build_vocab(args.input_file, args.output_dir)
  dataset = _build_dataset(args.input_file, vocab, tag_dict)

  logging.info('Shuffling dataset.')
  np.random.seed(123)
  shuffled_indices = np.random.permutation(len(dataset))
  num_validation_sentences = int(args.validation_percentage * len(dataset))

  val_indices = shuffled_indices[:num_validation_sentences]
  train_indices = shuffled_indices[num_validation_sentences:]

  _write_dataset('train', dataset, train_indices,
                 args.train_shards, args.output_dir)
  _write_dataset('valid', dataset, val_indices,
                 args.validation_shards, args.output_dir)

if __name__ == '__main__':
  main()

