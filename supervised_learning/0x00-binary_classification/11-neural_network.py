#!/usr/bin/env python3
"""Contains the NeuralNetwork"""

import numpy as np


class NeuralNetwork():
    """ neural network"""

    def __init__(self, nx, nodes):
        """ constructor"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.W1 = np.random.randn(nx, nodes).reshape(nodes, nx)
        self.b1 = np.zeros(nodes).reshape(nodes, 1)
        self.A1 = 0
        self.W2 = np.random.randn(nodes).reshape(1, nodes)
        self.b2 = 0
        self.A2 = 0

        @property
        def W1(self):
                """property to retrieve W1"""
                return self.__W1

        @property
        def b1(self):
            """property to retrieve b1"""
            return self.__b1

        @property
        def A1(self):
            """property to retrieve A1"""
            return self.__A1

        @property
        def W2(self):
            """property to retrieve W2"""
            return self.__W2

        @property
        def b2(self):
            """property to retrieve b2"""
            return self.__b2

        @property
        def A2(self):
            """property to retrieve A2"""
            return self.__A2

        def forward_prop(self, X):
            """Calculates forward propagation of the neural network"""
            Z1 = np.matmul(self.__W1, X) + self.__b1
            self.__A1 = 1 / (1 + np.exp(-Z1))
            Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
            self.__A2 = 1 / (1 + np.exp(-Z2))
            return self.__A1, self.__A2

        def cost(self, Y, A):
            """Calculates the cost of the model using logistic regression"""
            cost = -np.sum((Y * np.log(A)) +
                           ((1 - Y) * np.log(1.0000001 - A))) / Y.shape[1]
            return cost
