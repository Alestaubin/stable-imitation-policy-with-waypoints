#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
sys.path.append(os.pardir)

import numpy as np
import cvxpy as cp

from typing import List, Tuple
from utils.log_config import logger
from sklearn.mixture import GaussianMixture

from functools import partial
from utils.plot_tools import plot_trajectory


def optimize_lpv_ds_from_data(trajectory, velocity, gmm,
                              attractor: np.ndarray = np.zeros((1, 2))):
    """ Optimize lpvds from data reimplemented in Python from the original repository:
    https://github.com/nbfigueroa/ds-opt/.

    Args:
        trajectory (np.ndarray): Position data.
        velocity (np.ndarray): Velocity data.
        ctr_type (int): Constraint type for lpvds.
        gmm (GaussianMixture): GaussianMixture object trained on the data with BIC scoring.
    """

    n_sample, dim = trajectory.shape

    constraints: List = []
    epsilon = 0.001

    K = len(gmm.weights_)
    A_c = np.zeros((dim, dim, K))
    b_c = np.zeros((dim, K))

    P = np.random.rand(dim, dim)
    P = P + P.T

    h_k = gmm.predict_proba(trajectory)

    A_vars = []
    b_vars = []
    Q_vars = []

    for k in range(K):
        A_vars.append(cp.Variable((dim, dim)))
        if k == 0:
            A_vars[k] = cp.Variable((dim, dim), symmetric=True)

        b_vars.append(cp.Variable((dim, 1)))
        Q_vars.append(cp.Variable((dim, dim), symmetric=True))

        A_vars[k].value = -np.eye(dim)
        b_vars[k].value = -np.eye(dim) @ attractor

        constraints.append(cp.transpose(A_vars[k]) @ P + P @ A_vars[k] == Q_vars[k])
        constraints.append(Q_vars[k] <= -epsilon * np.eye(dim))
        constraints.append(b_vars[k] == -A_vars[k] @ attractor)
        Q_vars[k].value = -np.eye(dim)

    raw_velocities = np.zeros((dim, n_sample, K))

    for k in range(K):
        h_K = np.tile(h_k[k, :], (dim, 1))
        f_k = A_vars[k] @ trajectory + np.tile(b_vars[k], (1, n_sample))
        raw_velocities[:, :, k] = h_K * f_k

    predicted_velocity = np.sum(raw_velocities, axis=2)
    error = predicted_velocity - velocity

    auxilary_var = cp.Variable((dim, n_sample))
    Objective = cp.Minimize(cp.sum_squares(auxilary_var))
    constraints.append(auxilary_var == error)
    prob = cp.Problem(Objective, constraints)
    prob.solve(solver=cp.MOSEK)

    if prob.status != cp.OPTIMAL:
        print("Solver failed!")

    for k in range(K):
        A_c[:, :, k] = A_vars[k].value
        b_c[:, k] = b_vars[k].value

    print(prob.solver_stats)
    print("Total error:", Objective.value)
    print("Computation Time:", prob.solver_stats.solve_time)

    return A_c, b_c