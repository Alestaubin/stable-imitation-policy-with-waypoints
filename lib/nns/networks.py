#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
import torch

import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.pardir)
from lipnet.group_sort import GroupSort
from lipnet.bjorck_linear import BjorckLinear
from torch.autograd import Variable


class NN(nn.Module):
    def __init__(self, input_shape: int = 2, output_shape: int = 2):
        """ Build a neural network module using torch.

        Args:
            input_shape (int, optional): Input shape of the network. Defaults to 2.
            output_shape (int, optional): Output shape of the network. Defaults to 1.
        """
        super(NN, self).__init__()

        # nonlinear estimator for the dynamical system
        self.fc1 = nn.Linear(input_shape, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, output_shape)

    def forward(self, x: torch.Tensor):
        """ Forward function connecting the NN architecture and throwing an output.

        Args:
            x (torch.Tensor): The input to neural network.

        Returns:
            torch.Tensor: the velocity.
        """

        x = nn.LeakyReLU(0.1)(self.fc1(x))
        x = nn.LeakyReLU(0.1)(self.fc2(x))
        x = nn.LeakyReLU(0.1)(self.fc3(x))
        x = nn.LeakyReLU(0.1)(self.fc4(x))
        x = self.fc5(x)

        return x


class LSTM(nn.Module):
    def __init__(self, input_shape: int = 2, output_shape: int = 2, hidden_size: int = 2,
        num_layers: int = 1):
        """ Build a neural network module using torch.

        Args:
            input_shape (int, optional): Input shape of the network. Defaults to 2.
            output_shape (int, optional): Output shape of the network. Defaults to 1.
        """
        super(LSTM, self).__init__()
        self.__n_layers = num_layers
        self.__hidden_size = hidden_size

        # nonlinear estimator for the dynamical system
        self.lstm = nn.LSTM(input_size=input_shape, hidden_size=self.__hidden_size, num_layers=self.__n_layers,
            batch_first=True)

        self.fc_1 = nn.Linear(hidden_size, 128)
        self.fc_2 = nn.Linear(128, 64)
        self.fc_3 = nn.Linear(64, 10)
        self.fc_4 = nn.Linear(10, output_shape)

    def forward(self, x):
        """ Forward function connecting the NN architecture and throwing an output.

        Args:
            x (torch.Tensor): The input to neural network.

        Returns:
            torch.Tensor: the velocity.
        """

        h_0 = Variable(torch.zeros(self.__n_layers, x.size(0), self.__hidden_size))
        c_0 = Variable(torch.zeros(self.__n_layers, x.size(0), self.__hidden_size))

        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        hn = hn.view(-1, self.__hidden_size)

        x = F.relu(hn)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = F.relu(self.fc_3(x))
        x = self.fc_4(x)

        return x


class LNET(nn.Module):
    def __init__(self, input_shape, output_shape):
        """Initiate a Lipnet module.

        Args:
            input_dim (int): Size of the input.
            output_dim (int): Size of the output.
        """
        super(LNET, self).__init__()

        # Set GroupSort activation
        self.act1 = GroupSort(1, axis=1)
        self.act2 = GroupSort(1, axis=1)
        self.act3 = GroupSort(1)

        # Fully connected layers
        self.fc1 = BjorckLinear(input_shape, 128)
        self.fc2 = BjorckLinear(128, 64)
        self.fc3 = BjorckLinear(64, 16)
        self.fc4 = BjorckLinear(16, output_shape)

    def forward(self, x):
        """ Forward pass of Lipnet.

        Args:
            x (np.ndarray): Input data, or batch of data.

        Returns:
            np.array: Result of the forward pass of the Lipnet.
        """

        # layer 1
        x = self.fc1(x)
        x = self.act1(x)

        # layer 2
        x = self.fc2(x)
        x = self.act2(x)

        # layer 3
        x = self.fc3(x)
        x = self.act3(x)

        # output layer
        x = self.fc4(x)
        return x
