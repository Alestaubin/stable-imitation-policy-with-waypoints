#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from datetime import datetime

import numpy as np


def mse(predicted: np.ndarray, label: np.ndarray, dim: int = 2):
    """ Calculate MSE error for predicted and label velocities.

    Note: This has been replaced by a one-liner. Might act quite differently.
    """

    return ((predicted.T - label.T) ** 2).mean(axis=1).sum() / dim


def check_stability(position: np.ndarray, velocity: np.ndarray, attractor: np.ndarray,
        tol: float):
    """ Check if the stability is violated or not.

    Args:
        position (np.ndarray): Position as a 2- or 3-D coordinate.
        velocity (np.ndarray): Velocity respective to the given trajectory.
        attractor (np.ndarray): Attractor point.
        tol (float): Tolerance value.

    Returns:
        np.ndarray: Array of size (2,) indicating the number of data points that violate
                    the first and second Lyapunov stability conditions respectively.
    """

    stable = np.zeros(2)

    v = ((position - attractor).T).dot(position - attractor).item()
    v_dot = ((position - attractor).T).dot(velocity).item()

    if v < -tol:
        stable[0] = 1

    if v_dot > tol:
        stable[1] = 1

    return stable


def get_features_count(max_degree: int) -> int:
    """ Find the number of monomials which maximum power indicated in max_degree.

    Args:
        max_degree (int): Maximum degree of monomial terms.

    Returns:
        int: Number of monomials.
    """

    return int((max_degree + 2) * (max_degree + 1) / 2)


def is_posdef(a: np.ndarray, tol = 0.00) -> bool:
    """ Determine whether a matrix is positive definite or not.

    Args:
        a (np.ndarray): A matrix like array.

    Returns:
        bool: True if matrix a is negative definite.
    """

    return np.all(np.real(np.linalg.eigvals(a)) > tol)


def is_negdef(a: np.ndarray, tol = 0.00) -> bool:
    """ Determine whether a matrix is negative definite or not.

    Args:
        a (np.ndarray): A matrix like array.

    Returns:
        bool: True if matrix a is negative definite.
    """

    return np.all(np.real(np.linalg.eigvals(a)) < tol)


def linear_ds(A: np.ndarray, x: np.ndarray, b: np.ndarray):
    """ Return the value of gradient for a linear DS.

    Args:
        A (np.ndarray): Array like, a matrix.
        x (np.ndarray): State variable, shape must comply with A.
        b (np.ndarray): System's bias term.

    Returns:
        np.array: Value of gradient at state x.
    """

    return np.dot(A, x) + b


def calibrate(pos):
    """ Each dimension is shifted so that the last data point ends in the origin.

    Args:
        pos (np.ndarray): The positions array in the shape of (n_dim * n_samples).

    Returns:
        np.ndarray: The shifted positions array ending in origin.
    """

    return np.array([p - p[-1] for p in pos])


def normalize(arr: np.ndarray):
    """ Normalization of data in the form of array. Each row is first
    summed and elements are then divided by the sum.

    Args:
        arr (np.ndarray): The input array to be normalized in the shape of (n_dim, n_samples).

    Returns:
        np.ndarray: The normalized array.
    """

    assert arr.shape[0] < arr.shape[1]
    max_magnitude = np.max(np.linalg.norm(arr, axis=0))
    return arr / max_magnitude


def time_stamp():
    """Get a time stamp string.
    """

    return datetime.now().strftime("%d-%m-%H-%M")
