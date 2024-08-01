#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time, sys, os
from typing import List

sys.path.append(os.pardir)
from utils.data_loader import load_pylasa_data

import numpy as np
import pandas as pd
import sympy as sp
import cvxpy as cp

from utils.log_config import logger
from utils.utils import is_posdef, is_negdef, mse

from sos.symbolic_plyds import SymbolicPLYDS
from functools import partial


def objective(parameters: List, poly_trajs: np.ndarray, velocity: np.ndarray):
    """ The objective function that needs to be optimized (minimized) to learn lpf ds.

    An objective function can be derived from parameters for 2D data:

                                    dot(x) = f(x)

    The parameters array has the number of rows equal to dimension, meaning that each row
    represents the respective polynomial DS function.

    Args:
        parameters (np.ndarray): Set of polynomial parameters and coefficients
            of shape (n_terms, dimension) passed onto the optimization procecss.

        trajectory (np.ndarray): Positions data in the shape of (sample_size, dimension).
        velocity (np.ndarray): Velocity data in the shape of (sample size, dimension).

    Returns:
        float: The cost wrt the objective function.
    """

    px, py = parameters
    est_velocity = np.apply_along_axis(lambda pos: np.array([cp.matmul(cp.matmul(pos.T, px), pos), cp.matmul(cp.matmul(pos.T, py), pos)]), 1, poly_trajs)

    return mse(est_velocity, velocity)


def optimize(trajectory: np.ndarray, velocity: np.ndarray, max_deg_ds: int,
        dimension: int, max_deg_lpf: int, tol: int, method: str):

    """ Optimization process to find an optimized and feasible non-linear DS represented by
    SOS polynomials.

    Args:
        trajectory (np.ndarray): Trajectory data in the shape of (sample size, dimension).
        velocity (np.ndarray): Velocity data in the shape of (sample size, dimension).

        tol (int): Tolerance applicable in constraints is 10^(-tol).
        init (np.ndarray, optional): Initialization vector. Defaults to None.

        max_deg_ds (int): Maximum degree of the polynomial dynamical system.
        max_deg_lpf (int): Maximum degree of the polynomial lyapunov potential function.
        dimension (int): Dimension of the data. Could also be extracted from vel and traj data.

    Returns:
        Function: The ds with optimized params.
    """
    logger.info(f'Starting SOS/SDP optimization sequence CXVPY')

    # build the sos handler based on degrees
    sos_stability = SymbolicPLYDS(ds_deg=max_deg_ds, lpf_deg=max_deg_lpf,
                                  dim=dimension, simplify_lpf=True, quadratic_lpf=True)
    sos_stability.arrange_constraints()

    # extracting the number of parameters
    ds_pi_size = sos_stability.sos_params["px"].shape[0]
    lpf_qi_size = sos_stability.sos_params["qx"].shape[0]
    dlpf_dt_gi_size = sos_stability.sos_params["gx"].shape[0]

    n_ply_params = dimension * (ds_pi_size ** 2)
    n_lpf_params = 0 # simplified LPF
    n_dlpf_dt_params = dimension * (dlpf_dt_gi_size ** 2)
    logger.info(f'Total parameters count is {n_ply_params}(ds (P)) + {n_lpf_params}(lpf (Q)) + '
        f'{n_dlpf_dt_params}(dlpf_dt (G)) = {n_ply_params + n_lpf_params + n_dlpf_dt_params}')

    px = cp.Variable((ds_pi_size, ds_pi_size))
    qx = cp.Variable((2, 2), value=[[1.0, 0.0], [0.0, 1.0]], symmetric=True)
    gx = cp.Variable((dlpf_dt_gi_size, dlpf_dt_gi_size), NSD=True)

    py = cp.Variable((ds_pi_size, ds_pi_size))
    qy = cp.Variable((2, 2), value=[[1.0, 0.0], [0.0, 1.0]], symmetric=True)
    gy = cp.Variable((dlpf_dt_gi_size, dlpf_dt_gi_size), NSD=True)

    tol = 10 ** (-tol)
    constraints = [gx == gx.T, gy == gy.T]

    for c in sos_stability.sos_params["cons_x"]:
        if len(c) > 1:
            rhs = sp.lambdify([sos_stability.sos_params["gx"]], c[1])
            lhs = sp.lambdify([sos_stability.sos_params["px"],
                               sos_stability.sos_params["py"]], c[0])
            cons = [rhs(gx) - (lhs(px, py)) <= tol, rhs(gx) - (lhs(px, py)) >= -tol]
        else:
            lhs = sp.lambdify([sos_stability.sos_params["gx"]], c[0])
            cons = [lhs(gx) <= tol, lhs(gx) >= -tol]
        constraints += cons

    for c in sos_stability.sos_params["cons_y"]:
        if len(c) > 1:
            rhs = sp.lambdify([sos_stability.sos_params["gy"]], c[1])
            lhs = sp.lambdify([sos_stability.sos_params["px"],
                               sos_stability.sos_params["py"]], c[0])
            cons = [rhs(gy) - (lhs(px, py)) <= tol, rhs(gy) - (lhs(px, py)) >= -tol]
        else:
            lhs = sp.lambdify([sos_stability.sos_params["gy"]], c[0])
            cons = [lhs(gy) <= tol, lhs(gy) >= -tol]
        constraints += cons

    # optimization time
    opt_time = time.time()

    # objective minimization
    n_samples = trajectory.shape[0]
    ones = [np.ones((n_samples, 1))]
    y_pows = [(trajectory[:, 1] ** i).reshape(n_samples, 1) for i in range(1, max_deg_ds // 2 + 1)]
    x_pows = [(trajectory[:, 0] ** i).reshape(n_samples, 1) for i in range(1, max_deg_ds // 2 + 1)]
    poly_trajs = np.hstack((*ones, *x_pows, *y_pows))

    logger.info(f'Running optimization for MSE objective and the constraints with '
                f'level-{tol} tolerance')

    prob = cp.Problem(cp.Minimize(objective([px, py], poly_trajs, velocity)), constraints)
    prob.solve(cp.SCS, verbose=True)

    final_ds = partial(sos_stability.ds, px=px.value, py=py.value)
    final_lpf = lambda x: sos_stability.lpf(x, qx=qx.value, qy=qy.value)[0]
    final_dlpf_dt =  lambda x: sos_stability.dlpf_dt(x, px=px.value, py=py.value,
                            qx=qx.value, qy=qy.value)[0]

    logger.info(f'Optimization was concluded in {opt_time - time.time():.4f} seconds, \n'
    f'Summary: \n\nP: \n {pd.DataFrame(px.value)} \n {pd.DataFrame(py.value)} \n\n'
    f'Q: \n {pd.DataFrame(qx.value)} \n {pd.DataFrame(qy.value)} \n\n'
    f'G: \n {pd.DataFrame(gx.value)} \n {pd.DataFrame(gy.value)} \n\n'
    f'PD Condition Q: {[is_posdef(qx.value), is_posdef(qy.value)]} \n'
    f'ND Condition G: {[is_negdef(gx.value), is_negdef(gy.value)]}\n'
    f'\n{np.linalg.eigvals(gx.value)}, \n{np.linalg.eigvals(gy.value)}\n')


    solution_dict = {"p": [px.value, py.value], "q": [qx.value, qy.value],
                     "g": [gx.value, gy.value]}
    functions_dict = {"ds": final_ds, "lpf": final_lpf, "dlpf_dt": final_dlpf_dt}

    return functions_dict, solution_dict
