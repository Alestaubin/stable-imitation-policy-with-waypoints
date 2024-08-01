#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time, sys, os
from typing import Callable

sys.path.append(os.pardir)
from utils.data_loader import load_pylasa_data

import numpy as np
import pandas as pd
import cvxpy as cp

from utils.log_config import logger
from utils.utils import is_posdef, mse

from sos.symbolic_lgpds import SymbolicLGPDSWOP
from functools import partial


def objective_mse(ds_func: Callable, trajectory: np.ndarray, velocity: np.ndarray):
    """ The objective function that needs to be optimized (minimized) to learn lpf ds.

    An objective function can be derived from parameters for 2D data:

                                    dot(x) = f(x)

    The parameters array has the number of rows equal to dimension, meaning that each row
    represents the respective polynomial DS function.

    Args:
        parameters (np.ndarray): Set of polynomial parameters and coefficients passed
            onto the optimization procecss.

        trajectory (np.ndarray): Positions data in the shape of (sample_size, dimension).
        velocity (np.ndarray): Velocity data in the shape of (sample size, dimension).

    Returns:
        float: The cost wrt the objective function.
    """

    est_velocity = np.apply_along_axis(ds_func, 1, trajectory)
    return mse(est_velocity, velocity)


def optimize(trajectory: np.ndarray, velocity: np.ndarray, max_deg: int,
        dimension: int, method: str):

    """ Optimization process to find an optimized and feasible non-linear DS represented by
    SOS polynomials.

    Args:
        trajectory (np.ndarray): Trajectory data in the shape of (sample size, dimension).
        velocity (np.ndarray): Velocity data in the shape of (sample size, dimension).

        max_deg (int): Maximum degree of the LPF.
        dimension (int): Dimension of the data. Could also be extracted from vel and traj data.

    Returns:
        Dict[str, Callable], Dict[str, nd.array]: The functions with optimized params.
    """
    logger.info(f'Starting SOS/SDP optimization sequence CXVPY')

    # build the sos handler based on degrees
    sos_stability = SymbolicLGPDSWOP(degree=max_deg, dim=dimension)

    # extracting the number of parameters
    lpf_q_size = dimension * (max_deg // 2)

    n_lpf_params = lpf_q_size * lpf_q_size
    logger.info(f'Total parameters count is {n_lpf_params}(lpf (Q))')

    # variables and constraints
    q = cp.Variable((lpf_q_size, lpf_q_size), PSD=True)
    constraints = [q == q.T]

    # optimization time
    opt_time = time.time()

    # objective minimization
    ds_func = partial(sos_stability.ds, q=q)
    prob = cp.Problem(cp.Minimize(objective_mse(ds_func, trajectory, velocity)), constraints)
    prob.solve(method, max_iters=20000000, verbose=True)

    final_ds = partial(sos_stability.ds, q=q.value)
    final_lpf = partial(sos_stability.lpf, q=q.value)
    final_dlpf_dt = partial(sos_stability.dlpf_dt, q=q.value)

    logger.info(f'Optimization was concluded in {opt_time:.4f} seconds, \n'
    f'Q: \n {pd.DataFrame(q.value)} \n\n'
    f'PD Condition Q: {is_posdef(q.value)} \n'
    f'PD Q Eigenvals: {np.linalg.eigvals(q.value)}\n')

    solution_dict = {"q": q.value}
    functions_dict = {"ds": final_ds, "lpf": final_lpf, "dlpf_dt": final_dlpf_dt}

    return functions_dict, solution_dict

if __name__ == '__main__':
    # load a motion type
    positions_py, velocities_py = load_pylasa_data(motion_shape="G")

    # optimize for a sample of params
    funs, sols = optimize(trajectory=positions_py[:6000], velocity=velocities_py[:6000],
                          max_deg=8, dimension=2, method=cp.SCS)

    ds = funs["ds"]
    class ds_c:
        def predict(x):
            preds = np.apply_along_axis(ds, 1, x)
            return preds

    from utils.plot_tools import plot_ds_stream, plot_contours
    plot_ds_stream(ds_c, positions_py[:1000])
    # plot_contours(funs["lpf"], positions_py)
    # plot_contours(funs["dlpf_dt"], positions_py, color='Reds')