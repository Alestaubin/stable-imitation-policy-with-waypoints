#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
import math
import numpy as np

from sklearn import mixture
from typing import List

sys.path.append(os.pardir)
from utils.log_config import logger


def diff(arr: np.ndarray or List):
    """ Calculate the difference of consecutive elements.

    TODO: This function is redundant and can be easily replaced.

    Args:
        arr (np.ndarray or List): The input array.

    Returns:
        List: Calculated difference of elements.
    """

    difference = list()
    for i in range(len(arr) - 1):
        difference.append(arr[i + 1] - arr[i])
    return difference


def select_model(bics: List):
    """ Find the best model based on BIC scores.

    Args:
        bics (List): List of the available BIC scores.

    Returns:
        int: Index of the selected model.
    """
    # calculate the first and second order derivative
    diff1 = [0] + diff(bics)
    diff2 = [0] + diff(diff1)

    return diff2.index(max(diff2))


def fit(trajectory: np.ndarray, is_linear: bool = False, num_components_max: int = 10):
    """ Fit gmm to a desired trajectory.

    Args:
        trajectory (np.ndarray): The main trajectory to fit the mixture model.
        is_linear (bool): Set true if the underlying data generation process is a
            linear dynamical system.
        num_components_max (int, optional): Choosing the maximum number
            of Gaussian components.
    Returns:
        mixture.GaussianMixture: the resulting GMM
    """

    # store the bic scores and the corresponding GMMs
    bics = list()
    gmms = list()
    num_components = range(1, num_components_max + 1)

    # fit the gmms
    for num in num_components:

        # fit the model
        gmm = mixture.GaussianMixture(n_components=num)
        gmm.fit(trajectory)
        gmms.append(gmm)

        # get bic score
        current_bic = gmm.bic(trajectory)
        bics.append(current_bic)

    # find the best model
    if is_linear: logger.warn('Adapting a linear model instead of bic scoring')
    gmm = gmms[(select_model(bics))] if not is_linear else gmms[0]

    # return the best model
    return gmm
