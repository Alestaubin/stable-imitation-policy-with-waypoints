#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time, sys, os
from typing import List
sys.path.append(os.pardir)

import numpy as np
import casadi as ca

from utils.log_config import logger
from utils.data_loader import load_pylasa_data


def optimize(trajectory: np.ndarray, velocity: np.ndarray, max_deg: int, dimension: int):
    lpf_q_shape = (dimension * (max_deg // 2), dimension * (max_deg // 2))
    x = ca.MX.sym('x', (dimension, 1))
    p = ca.MX.sym('p', (dimension, 1))
    q = ca.MX.sym('q', (lpf_q_shape[0], lpf_q_shape[1]))

    basis = [x[idx] ** (2*pow) for pow in range(1, max_deg//2 + 1) for idx in range(dimension)]
    b = ca.vertcat(*basis)
    b = ca.reshape(b, (dimension * (max_deg // 2), 1))
    print(f'LPF Basis: {b}')

    # LPF
    lpf = ca.mtimes([b.T, q, q.T, b])
    lpf_function = ca.Function('lpf_function', [x, q], [lpf])
    print(f'LPF: {lpf_function}')

    # LPF gradient
    lpf_gradient = ca.gradient(lpf_function(x, q), x)
    print(f'∇LPF: {lpf_gradient}')

    # DS
    ds = (-1 * (p ** 2)) * lpf_gradient
    ds_function = ca.Function('ds_function', [x, q, p], [ds])
    print(f'DS: {ds_function}')

    # Derivative
    dlpf_dt = ca.mtimes(lpf_gradient.T, ds)
    dlpf_dt_function = ca.Function('dlpf_dt_function', [x, q, p], [dlpf_dt])

    # Dataset
    trajectory_ca = ca.MX(trajectory.T)
    velocity_ca = ca.MX(velocity.T)

    # Objective
    e = ds_function(trajectory_ca, q, p) - velocity_ca
    mse = ca.dot(e, e) / (dimension * trajectory_ca.shape[0])
    mse_function = ca.Function('mse_function', [q, p], [mse])
    print(f'Obj: {mse_function}')

    # Optimize
    opts = {'ipopt.print_level': 5, 'print_time': 5, 'ipopt.tol': 1e-22, 'ipopt.max_iter': 2000}
    nlp = {'x': ca.vertcat(ca.reshape(q, lpf_q_shape[0] * lpf_q_shape[1], 1), p), 'f': mse}
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    # Solve the optimization problem
    initial_guess = np.random.rand(lpf_q_shape[0] * lpf_q_shape[1] + dimension, 1)
    solution = solver(x0=initial_guess)

    # Extract the optimized values
    optimal_q = ca.reshape(solution['x'][:lpf_q_shape[0] * lpf_q_shape[1]],
                           lpf_q_shape[0], lpf_q_shape[1])
    optimal_p = ca.reshape(solution['x'][lpf_q_shape[0] * lpf_q_shape[1]:],
                           dimension, 1)

    print(f'\nMSE: {mse_function(optimal_q, optimal_p)}\n')

    final_ds = lambda x: np.array(ds_function(x, optimal_q, optimal_p)).T
    final_lpf = lambda x: lpf_function(x, optimal_q)
    final_dlpf_dt = lambda x: dlpf_dt_function(x, optimal_q, optimal_p)

    logger.info(f'Optimization is concluded with: \n'
                f'P: {optimal_p} \n '
                f'Q: {optimal_q}')

    solution_dict = {"p": optimal_p, "q": optimal_q}
    functions_dict = {"ds": final_ds, "lpf": final_lpf, "dlpf_dt": final_dlpf_dt}

    return functions_dict, solution_dict


if __name__ == '__main__':
    # load a motion type
    positions_py, velocities_py = load_pylasa_data(motion_shape="N")

    # optimize for a sample of params
    funs, sols = optimize(trajectory=positions_py[:6000], velocity=velocities_py[:6000],
                          max_deg=6, dimension=2)

    ds = funs["ds"] # NOT SURE IF IT'S ALRIGHT
    class ds_c:
        def predict(x):
            preds = ds(x.T)
            return preds

    from utils.plot_tools import plot_ds_stream, plot_contours
    plot_ds_stream(ds_c, positions_py, show_rollouts=True)
    plot_contours(funs["lpf"], positions_py)
    plot_contours(funs["dlpf_dt"], positions_py)
