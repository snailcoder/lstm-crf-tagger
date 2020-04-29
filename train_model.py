#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : train_model.py
# Author            : Yan <yanwong@126.com>
# Date              : 08.04.2020
# Last Modified Date: 27.04.2020
# Last Modified By  : Yan <yanwong@126.com>

import argparse
import collections
import logging
import time
import os

import tensorflow as tf
import numpy as np

import model
import losses
import metrics
import data_utils
import configuration

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description='Train the LSTM-CRF tagger.')

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
parser.add_argument('ckpt_dir',
                    help='Directory for saving and loading checkpoints.')
# parser.add_argument('model_dir', default='lstm_crf_tagger',
#                     help='Directory for saving and loading model.')
# parser.add_argument('model_version', default='0001',
#                     help='Version of saved model.')


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

dev_accuracy = tf.keras.metrics.Accuracy(name='dev_accuracy')
dev_metric = metrics.TaggerMetric(model_config.n_tags)

ckpt = tf.train.Checkpoint(tagger=tagger, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, args.ckpt_dir, max_to_keep=50)

# If you want to deploy model, put SavedModel here.
# model_path = os.path.join(args.model_dir, args.model_version)

if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print('Latest checkpoint restored!')

def classification_report(metric):
  metric_res = metric.result().numpy()
  # print('Classification report:\n')
  print('\taccuracy: ', metric_res[0])
  print('\tprecison: ', metric_res[1])
  print('\trecall: ', metric_res[2])
  print('\tF1 score: ', metric_res[3])

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

def eval_step(inp, tar):
  # inp.shape == (batch_size, max_seq_len)
  # tar.shape == (batch_size, max_seq_len)
  padding_mask = data_utils.create_padding_mask(inp)
  
  pred, potentials = tagger(inp, False, padding_mask)  # (batch_size, max_seq_len)
  loss = losses.loss_function(tar, potentials, padding_mask,
                              tagger.crf_layer.trans_params)

  dev_accuracy(tar, pred, padding_mask)
  dev_metric(tar, pred, padding_mask)

def evaluate(dataset):
  dev_accuracy.reset_states()
  dev_metric.reset_states()

  for batch, (inp, tar) in enumerate(dataset):
    eval_step(inp, tar)

  classification_report(dev_metric)

best_dev = -np.inf

for epoch in range(train_config.n_epochs):
  start = time.time()

  train_loss.reset_states()
  train_accuracy.reset_states()
  train_metric.reset_states()

  dev_accuracy.reset_states()
  dev_metric.reset_states()

  for batch, (inp, tar) in enumerate(train_dataset):
    train_step(inp, tar)

    if batch % 50 == 0:
      print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
        epoch + 1, batch, train_loss.result(), train_accuracy.result()))
      classification_report(train_metric)

    if batch % train_config.freq_eval == 0:
      evaluate(dev_dataset)

      macro_f1 = tf.math.reduce_mean(dev_metric.result()[3])
      print('Eval accuracy {:.4f} Macro F1 {:.4f}'.format(
        dev_accuracy.result(), macro_f1))
      classification_report(dev_metric)

      if macro_f1 > best_dev:
        best_dev = macro_f1
        print('New best dev F1 {:.4f}'.format(best_dev))

        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}.'.format(
          epoch + 1, ckpt_save_path))

        # Save model for deploying.
        # Remember to use tagger.call() instead of tagger() in above inferences.
        # tf.saved_model(tagger, model_path, signatures={'call': tagger.call})
        # print('Saved model for epoch {} at {}.').format(epoch + 1, model_path)

      print('Until now, best dev F1 {:.4f}'.format(best_dev))

  print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(
    epoch + 1, train_loss.result(), train_accuracy.result()))
  classification_report(train_metric)

print('Best dev F1 score {:.4f}'.format(best_dev))


