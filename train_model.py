#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : train_model.py
# Author            : Yan <yanwong@126.com>
# Date              : 08.04.2020
# Last Modified Date: 10.04.2020
# Last Modified By  : Yan <yanwong@126.com>

import argparse
import collections
import logging
import time

import tensorflow as tf

import model
import losses
import metrics
import data_utils
import configuration

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(
    description='Train the LSTM-CRF tagger for word segmentation.')

parser.add_argument('train_pattern',
                    help='File pattern of sharded TFRecord files containing '
                    'tf.Example protos for training.')
parser.add_argument('dev_pattern',
                    help='File pattern of sharded TFRecord files containing '
                    'tf.Example protos for validation.')
parser.add_argument('w2v',
                    help='The pre-trained word vector.')
parser.add_argument('vocab',
                    help='The vocabulary file containing all words')
parser.add_argument('save_dir',
                    help='Directory for saving and loading checkpoints.')


args = parser.parse_args()

model_config = configuration.ModelConfig()
train_config = configuration.TrainingConfig()

train_dataset = data_utils.create_dataset(args.train_pattern,
                                          train_config.batch_size)
dev_dataset = data_utils.create_dataset(args.dev_pattern,
                                        train_config.batch_size)

vocab = data_utils.load_vocab(args.vocab)
w2v = data_utils.load_w2v(args.w2v, vocab)
embeddings = data_utils.load_vocab_embeddings(w2v, vocab, model_config.d_word)

tagger = model.Model(embeddings, len(vocab), model_config)

optimizer = tf.keras.optimizers.SGD(lr=train_config.learning_rate,
                                    clipvalue=train_config.clip_gradients)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Accuracy(name='train_accuracy')
train_metric = metrics.TaggerMetric(model_config.n_tags)
train_precision = tf.keras.metrics.Precision(name='train_precision')
train_recall = tf.keras.metrics.Recall(name='train_recall')

def train_step(inp, tar):
  # inp.shape == (batch_size, max_seq_len)
  # tar.shape == (batch_size, max_seq_len)
  padding_mask = data_utils.create_padding_mask(inp)
  
  with tf.GradientTape() as tape:
    pred, potentials = tagger(inp, True, padding_mask)  # (batch_size, max_seq_len)
    loss = losses.loss_function(tar, potentials, padding_mask, 
                                tagger.crf_layer.trans_params)

  gradients = tape.gradient(loss, tagger.trainable_variables)
  optimizer.apply_gradients(zip(gradients, tagger.trainable_variables))

  train_loss(loss)
  train_accuracy(tar, pred, padding_mask)
  train_metric(tar, pred, padding_mask)
  # train_precision(tar, pred)
  # train_recall(tar, pred)

for epoch in range(train_config.n_epochs):
  start = time.time()

  train_loss.reset_states()
  train_accuracy.reset_states()
  train_metric.reset_states()
  # train_precision.reset_states()
  # train_recall.reset_states()

  for batch, (inp, tar) in enumerate(train_dataset):
    train_step(inp, tar)

    if batch % 50 == 0:
      print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
        epoch + 1, batch, train_loss.result(), train_accuracy.result()))
      print('Tagger metric: ', train_metric.result())

  train_f1_score = 2 * train_precision.result() * train_recall.result() / (
      train_precision.result() + train_recall.result()) 

  print('Epoch {} Loss {:.4f} Accuracy {:.4f} F1 {:.4f}'.format(
    epoch + 1, train_loss.result(), train_accuracy.result()))
  print('Tagger metric: ', train_metric.result())



