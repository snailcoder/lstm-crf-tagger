#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : model.py
# Author            : Yan <yanwong@126.com>
# Date              : 30.03.2020
# Last Modified Date: 10.04.2020
# Last Modified By  : Yan <yanwong@126.com>

import tensorflow as tf
import tensorflow_addons as tfa

# class PretrainedEmbedding(tf.keras.layers.Layer):
#   def __init__(self, embeddings, rate=0.1):
#     super(PretrainedEmbedding, self).__init__()
# 
#     self.embeddings = tf.constant(embeddings)
#     self.dropout = tf.keras.layers.Dropout(rate=rate)
# 
#   def call(self, inputs, training=None):
#     output = tf.nn.embedding_lookup(self.embeddings, inputs)
#     return self.dropout(output, training=training)

class CRFLayer(tf.keras.layers.Layer):
  def __init__(self, num_tags):
    super(CRFLayer, self).__init__()

    self.num_tags = num_tags

  def build(self, input_shape):
    self.trans_params = self.add_weight(
        name='trans_params',
        shape=(self.num_tags, self.num_tags))
    self.build = True

  def call(self, x, seq_len):
    # x.shape == (batch_size, max_seq_len, num_tags)

    tags, scores = tfa.text.crf_decode(x, self.trans_params, seq_len)

    return tags, scores

class Model(tf.keras.Model):
  def __init__(self, embeddings, vocab_size, config):
    super(Model, self).__init__()

    self.rnn_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
        config.d_word_lstm,
        return_sequences=True,
        recurrent_dropout=config.lstm_dropout,
        kernel_regularizer=tf.keras.regularizers.l2(config.l2_lambda),
        recurrent_regularizer=tf.keras.regularizers.l2(config.l2_lambda)))
    self.hidden_layer = tf.keras.layers.Dense(
        config.d_word_lstm, activation='tanh')
    self.final_layer = tf.keras.layers.Dense(config.n_tags)
    self.crf_layer = CRFLayer(config.n_tags)
    self.embedding_layer = tf.keras.layers.Embedding(
        vocab_size, config.d_word, trainable=False)
    self.embedding_dropout = tf.keras.layers.Dropout(rate=config.emb_dropout)

    if embeddings is not None:
      self.embedding_layer.build((None, vocab_size))
      self.embedding_layer.set_weights([embeddings])

  def call(self, x, training, padding_mask):
    # x.shape == (batch_size, max_seq_len)
    # padding_mask.shape == (batch_size, max_seq_len)

    x = self.embedding_layer(x)  # (batch_size, max_seq_len, d_word)
    x = self.embedding_dropout(x, training=training)
    x = self.rnn_layer(x, mask=padding_mask, training=training)
    x = self.hidden_layer(x)
    logits = self.final_layer(x)

    true_seq_len = tf.cast(tf.math.reduce_sum(padding_mask, axis=1), tf.int32)
    tags, scores = self.crf_layer(logits, true_seq_len)

    return tags, logits

