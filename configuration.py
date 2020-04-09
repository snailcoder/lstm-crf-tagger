#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : configuration.py
# Author            : Yan <yanwong@126.com>
# Date              : 08.04.2020
# Last Modified Date: 09.04.2020
# Last Modified By  : Yan <yanwong@126.com>

"""Default configuration for model architecture and training."""

class ModelConfig(object):
  """Wrapper class for model hyperparameters."""
  def __init__(self):
    self.d_word = 100  # word embedding dimension
    self.d_word_lstm = 100  # word LSTM hidden layer size
    self.emb_dropout = 0.5  # droupout on the input (0 = no dropout)
    self.lstm_dropout = 0.
    self.l2_lambda = 0.0001
    self.n_tags = 4

class TrainingConfig(object):
  """Wrapper class for model hyperparameters."""
  def __init__(self):
    self.learning_rate = 0.01
    self.clip_gradients = 5.0
    self.n_epochs = 100  # number of epochs over the training set
    self.freq_eval = 1000  # evaluate on dev every freq_eval steps
    self.batch_size = 64
