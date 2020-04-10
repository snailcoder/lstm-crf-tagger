#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : metrics.py
# Author            : Yan <yanwong@126.com>
# Date              : 10.04.2020
# Last Modified Date: 10.04.2020
# Last Modified By  : Yan <yanwong@126.com>

import tensorflow as tf

class TaggerMetric(tf.keras.metrics.Metric):
  def __init__(self, n_classes, name='tagger_metric', **kwargs):
    super(TaggerMetric, self).__init__(name=name, **kwargs)

    self.n_classes = n_classes
    self.true_positives = self.add_weight(
        name='tp', shape=(n_classes,), initializer='zeros')
    self.false_positives = self.add_weight(
        name='fp', shape=(n_classes,), initializer='zeros')
    self.true_negatives = self.add_weight(
        name='tn', shape=(n_classes,), initializer='zeros')
    self.false_negatives = self.add_weight(
        name='fn', shape=(n_classes,), initializer='zeros')
    self.total = self.add_weight(
        name='total', initializer='zeros')

  def update_state(self, y_true, y_pred, sample_weight=None):
    # sample_weight.shape == (batch_size, max_seq_len)

    if sample_weight is None:
      sample_weight = tf.ones_like(y_true, dtype=tf.float32)

    true_seq_len = tf.math.reduce_sum(sample_weight)
    self.total.assign_add(true_seq_len)

    for i in range(self.n_classes):
      binary_true_label = tf.cast(tf.math.equal(y_true, i), tf.float32)
      binary_pred_label = tf.cast(tf.math.equal(y_pred, i), tf.float32)

      true_pos = tf.math.count_nonzero(
          binary_pred_label * binary_true_label * sample_weight)
      true_pos = tf.cast(true_pos, tf.float32)
      # true_positives[i] is a Tensor which has no add_assign method.
      self.true_positives[i].assign(self.true_positives[i] + true_pos)

      true_neg = tf.math.count_nonzero(
          (binary_true_label - 1) * (binary_pred_label - 1) * sample_weight)
      true_neg = tf.cast(true_neg, tf.float32)
      self.true_negatives[i].assign(self.true_negatives[i] + true_neg)

      false_pos = tf.math.count_nonzero(
          binary_pred_label * (binary_true_label - 1) * sample_weight)
      false_pos = tf.cast(false_pos, tf.float32)
      self.false_positives[i].assign(self.false_positives[i] + false_pos)

      false_neg = tf.math.count_nonzero(
          (binary_pred_label - 1) * binary_true_label * sample_weight)
      false_neg = tf.cast(false_neg, tf.float32)
      self.false_negatives[i].assign(self.false_negatives[i] + false_neg)

  def result(self):
    accuracy = tf.math.divide_no_nan(
        self.true_positives + self.true_negatives,
        self.total)
    precision = tf.math.divide_no_nan(
        self.true_positives,
        self.true_positives + self.false_positives)
    recall = tf.math.divide_no_nan(
        self.true_positives,
        self.true_positives + self.false_negatives)
    f1_score = tf.math.divide_no_nan(2 * precision * recall,
                                     precision + recall)
    
    # This is the same accuracy with tf.keras.metrics.Accuracy,
    # so it is unnecessary to reimplement it again.
    # accuracy = tf.math.divide_no_nan(
    #     tf.math.reduce_sum(self.true_positives),
    #     self.total)
 
    return accuracy, precision, recall, f1_score

  def reset_states(self):
    self.true_positives.assign(tf.zeros_like(self.true_positives))
    self.false_positives.assign(tf.zeros_like(self.false_positives))
    self.true_negatives.assign(tf.zeros_like(self.true_negatives))
    self.false_negatives.assign(tf.zeros_like(self.false_negatives))
    self.total.assign(tf.zeros_like(self.total))

