#!/usr/bin/env python3
"""DeepNeuralNetwork"""
import numpy as np


class DeepNeuralNetwork:
    """deep neural network with one hidden layer"""

    def __init__(self, nx, layers):
        """constructor"""
        if type(nx) != int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(layers) != list or len(layers) < 1:
            raise TypeError('layers must be a list of positive integers')

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.L):
            if type(layers[i]) != int or layers[i] < 0:
                raise TypeError('layers must be a list of positive integers')
            kb = "b" + str(i + 1)
            kW = "W" + str(i + 1)

            self.weights[kb] = np.zeros(layers[i]).reshape(layers[i], 1)
            if i > 0:
                val = layers[i-1]
            else:
                val = nx
            self.weights[kW] = np.random.randn(layers[i], val) * np.sqrt(2/val)

    @property
    def L(self):
        """property to retrieve L"""
        return self.__L

    @property
    def cache(self):
        """property to retrieve b1"""
        return self.__cache

    @property
    def weights(self):
        """property to retrieve A1"""
        return self.__weights

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        self.__cache['A0'] = X

        for l in range(self.__L):
            W_key = "W{}".format(l + 1)
            b_key = "b{}".format(l + 1)
            A_key_prev = "A{}".format(l)
            A_key_forw = "A{}".format(l + 1)

            Z = np.matmul(self.__weights[W_key], self.__cache[A_key_prev]) \
                + self.__weights[b_key]
            self.__cache[A_key_forw] = 1 / (1 + np.exp(-Z))

        return self.__cache[A_key_forw], self.__cache
