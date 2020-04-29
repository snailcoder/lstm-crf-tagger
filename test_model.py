#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : test_model.py
# Author            : Yan <yanwong@126.com>
# Date              : 27.04.2020
# Last Modified Date: 29.04.2020
# Last Modified By  : Yan <yanwong@126.com>

import argparse
import logging
import time
import os
import json

import tensorflow as tf
import numpy as np

import model
import losses
import metrics
import data_utils
import configuration
import special_words

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description='Test the LSTM-CRF tagger.')

parser.add_argument('input_file',
                    help='Text file containing test data.')
# parser.add_argument('output_file',
#                     help='Text file in Co-NLL format for saving predictions.')
parser.add_argument('vocab',
                    help='The vocabulary file (the same with training).')
parser.add_argument('tag_dict',
                    help='Json-format tag dict file. For example,'
                         ' for word segmentation task, the dict'
                         ' contains "S", "B", "M", "E"; for NER task,'
                         ' the dict contains "O", "B-PER", "I-PER",'
                         ' "B-ORG", "I-ORG", "B-LOC", "I-LOC".'
                         ' Check corpus/seg_tags.json and'
                         ' corpus/ner_tags.json.')

parser.add_argument('ckpt_dir',
                    help='Directory for loadding checkpoints.')
parser.add_argument('batch_size', type=int, default=20,
                    help='Batch size for testing.')

args = parser.parse_args()

vocab = data_utils.load_vocab(args.vocab)
with open(args.tag_dict, 'r') as f:
  tag_dict = json.load(f)

all_inp = []
all_tar = []
with open(args.input_file, 'r') as f:
  chs = []
  ids = []
  for line in f:
    line = line.strip()
    if not line:
      if chs and ids:
        all_inp.append(chs)
        all_tar.append(ids)
        chs = []
        ids = []
      continue
    toks = line.split()
    assert len(toks) == 2
    chs.append(vocab.get(toks[0], special_words.UNK_ID))
    ids.append(tag_dict[toks[1]])


def gen():
  for chs, ids in zip(all_inp, all_tar):
    yield(chs, ids)

dataset = tf.data.Dataset.from_generator(gen, (tf.int64, tf.int64))
dataset = dataset.padded_batch(args.batch_size,
                               padded_shapes=([-1], [-1]))
# for item in dataset.take(3):
#   print(item[0], item[1])

model_config = configuration.ModelConfig()
tagger = model.Model(None, len(vocab), model_config)
ckpt = tf.train.Checkpoint(tagger=tagger)
ckpt.restore(tf.train.latest_checkpoint(args.ckpt_dir))

accuracy = tf.keras.metrics.Accuracy(name='accuracy')
metric = metrics.TaggerMetric(model_config.n_tags)

def test_step(inp, tar):
  # inp.shape == (batch_size, max_seq_len)
  # tar.shape == (batch_size, max_seq_len)
  padding_mask = data_utils.create_padding_mask(inp)
  
  pred, potentials = tagger(inp, False, padding_mask)
  accuracy(tar, pred, padding_mask)
  metric(tar, pred, padding_mask)
  
  return pred, padding_mask

def classification_report(metric):
  metric_res = metric.result().numpy()
  # print('Classification report:\n')
  print('\taccuracy: ', metric_res[0])
  print('\tprecison: ', metric_res[1])
  print('\trecall: ', metric_res[2])
  print('\tF1 score: ', metric_res[3])

for batch, (inp, tar) in enumerate(dataset):
  pred, mask = test_step(inp, tar)

  if batch % 50 == 0:
    print('Batch {} Accuracy {:.4f}'.format(batch, accuracy.result()))

print('Accuracy {:.4f}'.format(accuracy.result()))
classification_report(metric)

