#!/usr/bin/evn python3
"""
Class RNNEncoder
inherits from tensorflow.keras.layers.Layer to encode
machine translators
"""

import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """
    RNNcoder Class
    """
    def __init__(self, vocab, embedding, units, batch):
        """
        class constructor
            return full sequence of outputs
            return hidden state
            recurrent weights intialized with glorot_uniform
        """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units=units,
                                       recurrent_initializer="glorot_uniform",
                                       return_sequences=True,
                                       return_state=True)

    def initialize_hidden_state(self):
        """
        Public instance method
        Initiailizes the hidden states for RNN cell to a tensor of zeros
        :return: a tensor shape(batch, units)
                contains: initialized hidden states
        """
        initializer = tf.keras.initializers.Zeros()
        rnnten = initializer(shape=(self.batch, self.units))
        return rnnten

    def call(self, x, initial):
        """
        Public instanc method
        """
        outputs, hidden = self.gru(self.embedding(x), initial_state=initial)
        return outputs, hidden
