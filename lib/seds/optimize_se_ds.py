#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from typing import List, Tuple
from utils.log_config import logger

from scipy.linalg import block_diag
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint, LinearConstraint
from sklearn.mixture import GaussianMixture
from functools import partial

# TODO: The optimization should be vectorized for faster speed.


def objective(parameters: np.ndarray, k: int, trajectory: np.ndarray, velocity: np.ndarray,
        dimension: int, posterior: List):
    """ The objective function that needs to be optimized (minimized) to learn lpv ds.

    Args:
        parameters (np.ndarray): Set of A_s and B_s parameters concatenated in a single array.
        k (int): Number of Gaussian components.
        trajectory (np.ndarray): Trajectory data in the shape of (sample size, dimension, 1).
        velocity (np.ndarray): Velocity data in the shape of (sample size, dimension, 1).
        dimension (int): Dimension of the data. Could also be extracted from vel and traj data.
        posterior (List): List of posterior probabilities regarding to the trajectory data points.

    Returns:
        float: The cost wrt the objective function.
    """
    As, bs = restore_lpvds_params(parameters, dimension, k)

    components = range(k)
    errors = np.empty(velocity.shape)

    for traj_idx, traj_point in enumerate(trajectory):

        # get posterior probability
        weights = posterior[traj_idx]

        # initialize the estimated velocity as zero
        est_vel = np.zeros((dimension,1))

        # get the estimated velocity
        # TODO: Relax this to a matrix form
        for i in components:
            est_vel += weights[i] * (As[i].dot(traj_point) + bs[i])

        # get the reference vector
        ref_vel = velocity[traj_idx]

        # calculate and add the error vector to the array
        errors[traj_idx] = est_vel - ref_vel

    return np.linalg.norm(errors)


def get_constraints(dimension: int, k: int, tol: float):
    """Get constraints for the SEDS optimization problem.

    Args:
        dimension (int): Data dimensions.
        k (int): Number of Gaussian components.
        tol (float): Tolerance.

    Returns:
        scipy.optimize.LinearConstraint: A linear constraint object.
    """

    cons: List[NonlinearConstraint] = []

    # 1. A_k + A_k.T -> Negative Definite (13-b in the paper)
    def negdef_cons(cluster_idx, parameters):
        """ Form the negetive definite constraint of A_k' + A_k -> neg. def.

        Args:
            parameters (np.ndarray): Optimization parameters.
        """
        As, _ = restore_lpvds_params(parameters, dimension, k)
        Ak = As[cluster_idx]

        M = Ak + Ak.T
        return np.max(np.linalg.eigvals(M))

    for cluster_idx in range(k):
        cons.append(NonlinearConstraint(partial(negdef_cons, cluster_idx), lb=-np.inf, ub=-tol))

    # 2. b_k is zero (13-a in the paper and since the target is around origin)
    trans_matrix = transformation_matrix(dimension, k)

    A_length = dimension * dimension * k
    b_length = dimension * k

    A_lb = np.full((A_length), -np.inf)
    b_lb = np.full((b_length), -tol)

    lb = np.concatenate((A_lb, b_lb))
    ub = np.full((A_length + b_length), tol)

    b_cons = LinearConstraint(trans_matrix, lb, ub)
    cons.append(b_cons)

    return cons


def optimize_seds(gmm: GaussianMixture, trajectory: np.ndarray, velocity: np.ndarray,
             tol: float, init: np.ndarray or None) -> Tuple[np.ndarray, np.ndarray]:
    """ Optimization process to find an optimized and feasible non-linear DS by represented by
    a mixture of Gaussian components and linear DS.

    Args:
        gmm (GaussianMixture): A mixture model defined using Scipy library.
        trajectory (np.ndarray): Trajectory data in the shape of (sample size, dimension, 1).
        velocity (np.ndarray): Velocity data in the shape of (sample size, dimension, 1).
        tol (float): Tolerance applicable in constraints.
        init (np.ndarray, optional): Initialization vector. Defaults to None.

    Returns:
        (np.ndarray, np.ndarray): Optimized A_s and b_s parameters.
    """

    # number of components of the gmm
    k = len(gmm.weights_)

    # size and dimension of data
    sample_size, dimension = trajectory.shape

    # posterior of the reference trajectory
    posterior = gmm.predict_proba(trajectory)

    # reshape trajectory and velocity
    trajectory = trajectory.reshape(sample_size, dimension,1)
    velocity = velocity.reshape(sample_size, dimension, 1)

    # initialization of parameters (As , bs)
    if init is None:
        init = np.random.rand(dimension * dimension * k + dimension * k)
    else:
        logger.info(f'Taking prior init {init.shape} into account')

    # get the constraints for this problem
    cons = get_constraints(dimension, k, tol)

    # get the result of the optimization
    # NOTE: different solvers may exhibit various responses to initial guess
    res = minimize(objective, init, (k, trajectory, velocity, dimension, posterior),
        constraints=cons, options={'disp': True, 'maxiter': 5000, 'ftol': tol})

    # collect optimization results
    if not res.success:
        logger.warn(f'Optimization process was not successful because: \n {res.message}')

    As, bs = restore_lpvds_params(res.x, dimension, k)
    return As, bs, res.nit


def restore_lpvds_params(parameters: np.ndarray, dimension: int, k: int):
    """ Restore parameters A_s and b_s from single vector form.

    Args:
        parameters (np.ndarray): Set of A_s and B_s parameters concatenated in a single array.
        dimension (int): The dimension of trajectory and velocity data or the state variables.
        k (int): Number of Gaussian components.

    Returns:
        np.ndarray, np.ndarray:
    """

    # split the array into the A part and b part
    As = parameters[:dimension * dimension * k]
    bs = parameters[dimension * dimension * k:]

    # reshape them into the original form
    As = As.reshape(k, dimension, dimension)
    bs = bs.reshape(k, dimension, 1)

    return (As, bs)


def transformation_matrix(dimension: int, k: int):
    """ Find the transformation matrix for constraints.

    Args:
        dimension (int): Data dimensions.
        k (int): Number of Gaussian components.

    Returns:
        np.ndarray: The constraints transformation matrix.
    """

    # transpose matrix for one A matrix
    transform = np.zeros((dimension ** 2, dimension ** 2))

    # stack the matrices diagonally for the k components
    stack_transform = transform
    for i in range(k - 1):
        stack_transform = block_diag(stack_transform, transform)

    # get the length for b part
    b_length = dimension * k

    # identity matrix for b part
    b_ident = np.identity(b_length)

    # stack the two parts together
    stack_transform = block_diag(stack_transform, b_ident)
    return stack_transform


