#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : losses.py
# Author            : Yan <yanwong@126.com>
# Date              : 08.04.2020
# Last Modified Date: 09.04.2020
# Last Modified By  : Yan <yanwong@126.com>

import tensorflow as tf
import tensorflow_addons as tfa

def loss_function(real, pred, mask, trans_params):
  # real.shape == (batch_size, max_seq_len)
  # pred.shape == (batch_size, max_seq_len, num_tags)
  # mask.shape == (batch_size, max_seq_len)
  true_seq_len = tf.math.reduce_sum(mask, axis=1)
  true_seq_len = tf.cast(true_seq_len, tf.int32)
  real = tf.cast(real, tf.int32)
  log_likelihood, _ = tfa.text.crf_log_likelihood(
      pred, real, true_seq_len, trans_params)
  loss = tf.math.reduce_sum(-log_likelihood)

  return loss

