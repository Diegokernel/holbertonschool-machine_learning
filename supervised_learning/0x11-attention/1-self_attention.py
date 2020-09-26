#!/usr/bin/env python3
"""
Class SelfAttention
"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
    Self Attention Class
    """
    def __init__(self, units):
        """
        Initiailize variables
        :param units: int representing num of hidden units
        """
        super(SelfAttention, self).__init__()

        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
        Public instance method
        :param s_prev: tensor shape(batch, units)
        """
        decW = tf.expand_dims(s_prev, 1)

        decW = self.W(decW)
        encU = self.U(hidden_states)
        outV = self.V((tf.nn.tanh(decW + encU)))
        weights = tf.nn.softmax(outV, axis=1)
        context = tf.reduce_sum((weights * hidden_states), axis=1)

        return context, weights
