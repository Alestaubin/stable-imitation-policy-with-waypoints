#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np


class PlanningPolicyInterface:
    """ A standard interface for motion planning policies.

    All the implementations should adhere to this formatting, especially the
    functions signatures to hold a consistent interface during the experiments.

    """

    def fit(self, trajectory: np.ndarray, velocity: np.ndarray, **kwargs):
        """Fit a motion planning policy to the data.

        Args:
            trajectory (np.ndarray): Trajectory data in shape (samples, features).
            velocity (np.ndarray): Velocity data in shape (samples, features).

        Raises:
            NotImplementedError: Raised in case the interface is used without
                implementing this function.
        """

        raise NotImplementedError


    def predict(self, trajectory: np.ndarray):
        """ Predict estimated velocities.

        Args:
            trajectory (np.ndarray): Trajectory in shape (sample size, dimension).

        Raises:
            NotImplementedError: Raised in case the interface is used without
                implementing this function.
        """

        raise NotImplementedError


    def load(self, model_name: str, dir: str = '../res'):
        """ Load the torch model.

        Args:
            model_name (str): Name of the model.
            dir (str, optional): Load directory. Defaults to '../res'.

        Raises:
            NotImplementedError: Raised in case the interface is used without
                implementing this function.
        """

        raise NotImplementedError


    def save(self, model_name: str, dir: str = '../res'):
        """ Save the torch model.

        Args:
            model_name (str): Name of the model.
            dir (str, optional): Save directory. Defaults to '../res'.

        Raises:
            NotImplementedError: Raised in case the interface is used without
                implementing this function.
        """

        raise NotImplementedError

    def lpf(self):
        """ Dummy lpf getter.

        Returns:
            None
        """
        return None
