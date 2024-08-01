#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time, sys, os

sys.path.append(os.pardir)

import numpy as np
import pandas as pd
import sympy as sp

from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
from scipy.linalg import block_diag

from typing import List

from utils.log_config import logger
from utils.data_loader import load_pylasa_data
from utils.utils import is_posdef, is_negdef, mse
from sos.symbolic_plyds import SymbolicPLYDS

from functools import partial

# global vars
iters = 0
start_time = time.time()


def objective(parameters: np.ndarray, ply_trajectory: np.ndarray, velocity: np.ndarray,
        sos_stability: SymbolicPLYDS, dimension: int):
    """ The objective function that needs to be optimized (minimized) to learn lpf ds.

    An objective function can be derived from parameters for 2D data:

                                    dot(x) = f(x)

    The parameters array has the number of rows equal to dimension, meaning that each row
    represents the respective polynomial DS function.

    Args:
        parameters (np.ndarray): Set of polynomial parameters and coefficients
            of shape (n_terms, dimension) passed onto the optimization procecss.

        ply_trajectory (np.ndarray): Positions data in the shape of (sample_size, n_ply_features).
        velocity (np.ndarray): Velocity data in the shape of (sample size, dimension).

        sos_stability (SOSStability): Sympy handled sos stability object.
        dimension (int): Dimension of the data. Could also be extracted from vel and traj data.

    Returns:
        float: The cost wrt the objective function.
    """

    # calculate objective error (TODO: extract coefficients and use multiplication to yield the estimated velocity)
    ds_p, _, _ = restore_opt_params(parameters, sos_stability, dimension)

    px, py = ds_p
    est_velocity = np.apply_along_axis(lambda pos: np.array([pos.T @ px @ pos, pos.T @ py @ pos]), 1, ply_trajectory)

    # mse objective function
    return mse(est_velocity, velocity)


def arrange_constraints(sos_stability: SymbolicPLYDS, dimension: int,
        eq_tol: float, pd_tol: float = 1e-6, simplify_lpf: bool = False):
    """ Get sum of square polynomial conditions for nonlinear optimization.

    Remember that the positive definiteness condition can also be reframed as
    Sylvester's criterion.

    Args:
        sos_stability (SOSStability): Sympy handled sos stability object.
        dimension (int): State space dimension.
        tol (float): Tolerance of nonlinear conditions.

    Returns:
        List[NonlinearConstraint]: A list of nonlinear sos constraints.
    """

    constraints: List[NonlinearConstraint] = list()

    # positive-(semi)definite constratints (TODO: use Sylvester's criterion to ease the calculation)
    def posdef_cons_lpf(x, sos, dim) -> float:
        """ Positive semidefinite constraint for Q matrix.

        Args:
            x (np.ndarray): The entire input parameters.

        Returns:
            float: minimum eigen value.
        """
        # restore only q params
        _, lpf_q, _ = restore_opt_params(x, sos, dim, bl_diag=True)

        # put all matrices in the same matrix in a diagonal way
        return np.real(np.min(np.linalg.eigvals(lpf_q + lpf_q.T)))

    def negdef_cons_dlpf_dt(x, sos, dim) -> float:
        """ Positive semidefinite constraint for G matrix.

        Args:
            x (np.ndarray): The entire input parameters.

        Returns:
            float: maximum eigen value.
        """
        # restore only g params
        _, _, dlpf_dt_g = restore_opt_params(x, sos, dim, bl_diag=True)

        # put all matrices in the same matrix in a diagonal way
        return np.real(np.max(np.linalg.eigvals(dlpf_dt_g + dlpf_dt_g.T)))

    # symmetry constraints
    def sym_cons_lpf(x, sos, dim) -> float:
        """ Positive semidefinite constraint for Q matrix.

        Args:
            x (np.ndarray): The entire input parameters.

        Returns:
            float: minimum eigen value.
        """

        _, lpf_q, _ = restore_opt_params(x, sos, dim, bl_diag=True)
        return np.linalg.norm(lpf_q - lpf_q.T)


    def sym_cons_dlpf_dt(x, sos, dim) -> float:
        """ Positive semidefinite constraint for Q matrix.

        Args:
            x (np.ndarray): The entire input parameters.

        Returns:
            float: minimum eigen value.
        """

        _, _, dlpf_dt_g = restore_opt_params(x, sos, dim, bl_diag=True)
        return np.linalg.norm(dlpf_dt_g - dlpf_dt_g.T)

    logger.info(f'Integration of positive-(semi)definite constraint(s)')

    if not simplify_lpf:
        cons_lpf_positdef = partial(posdef_cons_lpf, sos=sos_stability, dim=dimension)
        cons_lpf_symmetry = partial(sym_cons_lpf, sos=sos_stability, dim=dimension)

        constraints.append(NonlinearConstraint(cons_lpf_positdef, lb=+pd_tol, ub=np.inf))
        constraints.append(NonlinearConstraint(cons_lpf_symmetry, lb=-eq_tol, ub=+eq_tol))

    cons_dlpf_dt_negitdef = partial(negdef_cons_dlpf_dt, sos=sos_stability, dim=dimension)
    cons_dlpf_dt_symmetry = partial(sym_cons_dlpf_dt, sos=sos_stability, dim=dimension)

    constraints.append(NonlinearConstraint(cons_dlpf_dt_negitdef, lb=-np.inf, ub=-pd_tol))
    constraints.append(NonlinearConstraint(cons_dlpf_dt_symmetry, lb=-eq_tol, ub=+eq_tol))

    # build constraints based on problem size (TODO: refactor this to support 3-dim)
    sym_p = [sos_stability.sos_params["px"], sos_stability.sos_params["py"]]
    sym_q = [sos_stability.sos_params["qx"], sos_stability.sos_params["qy"]]
    sym_g = [sos_stability.sos_params["gx"], sos_stability.sos_params["gy"]]
    sym_constraints = sos_stability.sos_params["cons_x"] + sos_stability.sos_params["cons_y"]

    logger.info(f'Adding {len(sym_constraints)} affine constraints')

    for idx, cons in enumerate(sym_constraints):
        if len(cons) == 2:

            def cons_fn_double(x, sos, dim, q, p, g, cons):
                lhs = sp.lambdify(p + q if not simplify_lpf else p, cons[0])
                rhs = sp.lambdify(g, cons[1])
                ds_p, lpf_q, dlpf_dt_g = restore_opt_params(x, sos, dim)
                return lhs(*ds_p, *lpf_q) - rhs(*dlpf_dt_g) if not simplify_lpf else lhs(*ds_p) - rhs(*dlpf_dt_g)

            constraints.append(NonlinearConstraint(partial(cons_fn_double, sos=sos_stability, dim=dimension, p=sym_p, q=sym_q, g=sym_g, cons=cons), lb=-eq_tol, ub=eq_tol))

        if len(cons) == 1:

            def cons_fn_single(x, sos, dim, g, cons):
                rhs = sp.lambdify(g, cons[0])
                _, _, dlpf_dt_g = restore_opt_params(x, sos, dim)
                return rhs(*dlpf_dt_g)

            constraints.append(NonlinearConstraint(partial(cons_fn_single, sos=sos_stability, dim=dimension, g=sym_g, cons=cons), lb=-eq_tol, ub=eq_tol))

    return constraints


def optimize(trajectory: np.ndarray, velocity: np.ndarray, max_deg_ds: int,
        dimension: int, max_deg_lpf: int, tol: float, method: str, quadratic_lpf: bool,
        simplify_lpf: bool):

    """ Optimization process to find an optimized and feasible non-linear DS represented by
    SOS polynomials.

    Args:
        trajectory (np.ndarray): Trajectory data in the shape of (sample size, dimension).
        velocity (np.ndarray): Velocity data in the shape of (sample size, dimension).

        tol (float): Tolerance applicable in constraints.
        init (np.ndarray, optional): Initialization vector. Defaults to None.

        max_deg_ds (int): Maximum degree of the polynomial dynamical system.
        max_deg_lpf (int): Maximum degree of the polynomial lyapunov potential function.
        dimension (int): Dimension of the data. Could also be extracted from vel and traj data.

    Returns:
        Function: The ds with optimized params.
    """
    logger.info(f'Starting SOS/SDP optimization sequence with Scipy')
    tol = 10 ** (-tol)

    # check the sanity of data
    assert velocity.shape == trajectory.shape, "Mismatch in the dataset!"

    # build the sos handler based on degrees
    sos_stability = SymbolicPLYDS(ds_deg=max_deg_ds, lpf_deg=max_deg_lpf, dim=dimension,
                               quadratic_lpf=quadratic_lpf, simplify_lpf=simplify_lpf)
    sos_stability.arrange_constraints()


    # polynomialize the trajectory data
    n_samples = trajectory.shape[0]
    ones = [np.ones((n_samples, 1))]
    y_pows = [(trajectory[:, 1] ** i).reshape(n_samples, 1) for i in range(1, max_deg_ds // 2 + 1)]
    x_pows = [(trajectory[:, 0] ** i).reshape(n_samples, 1) for i in range(1, max_deg_ds // 2 + 1)]
    trajectory = np.hstack((*ones, *x_pows, *y_pows))

    # extracting the number of parameters
    n_ply_params, n_lpf_params, n_dlpf_dt_params = sos_stability.n_params

    logger.info(f'Total parameters count is {n_ply_params}(ds (P)) + {n_lpf_params}(lpf (Q)) + '
        f'{n_dlpf_dt_params}(dlpf_dt (G)) = {n_ply_params + n_lpf_params + n_dlpf_dt_params}')

    # randomly initialize the solution
    init = np.random.rand(n_ply_params + n_lpf_params + n_dlpf_dt_params)

    # optimization time
    opt_time = time.time()

    # arranging lyapunov and sos constraints
    constraints = arrange_constraints(sos_stability, dimension, eq_tol=tol,
                                      simplify_lpf=simplify_lpf)

    # debug logging
    def opt_callback(params, state=None):
        global iters, start_time
        if iters % 1 == 0:
            logger.info(f'Iter: {iters}, Time: {(time.time() - start_time):.4f} '
                f'MSE: {objective(params, trajectory, velocity, sos_stability, dimension):.4f}')
        iters += 1
        start_time = time.time()

    # objective minimization
    logger.info(f'Running {method} optimization for MSE objective and the constraints')
    res = minimize(objective, init, (trajectory, velocity, sos_stability, dimension),
        constraints=constraints, callback=opt_callback, method=method,
        options={'maxiter': 1500, 'ftol': tol} if method == 'SLSQP' else {'maxiter': 5000})

    # find the elapsed time
    optimization_time = time.time() - opt_time

    final_ds_p, final_lpf_q, final_dlpf_dt_g = restore_opt_params(res.x, sos_stability,
                                                                    dimension, bl_diag=False)
    logger.info(f'Optimization was concluded in {optimization_time:.4f} seconds, \n'
        f'Message: {res.status}: {res.message}\n\n'
        f'Summary: \n\nP: \n {pd.DataFrame(final_ds_p[0])} \n {pd.DataFrame(final_ds_p[1])} \n\n'
        f'PD Condition Q: {[is_posdef(final_lpf_q[i]) for i in range(dimension)]} \n'
        f'ND Condition G: {[is_negdef(final_dlpf_dt_g[i]) for i in range(dimension)]}\n'
        f'ND Eigenvals G: {[np.linalg.eigvals(final_dlpf_dt_g[i]) for i in range(dimension)]}\n\n')

    final_ds = partial(sos_stability.ds, px=final_ds_p[0], py=final_ds_p[1])
    final_lpf = partial(sos_stability.lpf, qx=final_lpf_q[0], qy=final_lpf_q[1])
    final_dlpf_dt = partial(sos_stability.dlpf_dt, px=final_ds_p[0],
                            py=final_ds_p[1], qx=final_lpf_q[0], qy=final_lpf_q[1])


    solution_dict = {"p": final_ds_p, "q": final_lpf_q, "g": final_dlpf_dt_g}
    functions_dict = {"ds": final_ds, "lpf": final_lpf, "dlpf_dt": final_dlpf_dt}

    return functions_dict, solution_dict


def restore_opt_params(parameters: np.ndarray, sos_stability: SymbolicPLYDS,
                       dimension: int, bl_diag: bool = False):
    """ Restore parameters single vector form passed onto the optimizer.

    Args:
        parameters (np.ndarray): Set of dlpf_dt_gi, ds_pi, and lpf_qi parameters raveled in a single array.
        dimension (int): The dimension of trajectory and velocity data or the state variables.
        bl_diag (bool): Enables the block diagram output format.

    Returns:
        List[np.ndarray], List[np.ndarray], List[np.ndarray]: ds_p(s), lpf_q(s), and dlpf_dt_g(s) matrices.
    """

    # retrive the size and shaoe of each parameter matrix
    ds_pi_size = sos_stability.sos_params["px"].shape
    lpf_qi_size = sos_stability.sos_params["qx"].shape
    dlpf_dt_gi_size = sos_stability.sos_params["gx"].shape

    ds_pi_param_size = ds_pi_size[0] ** 2
    lpf_qi_param_size = lpf_qi_size[0] ** 2
    dlpf_dt_gi_param_size = dlpf_dt_gi_size[0] ** 2

    simple_lpf = False
    if sos_stability.sos_params["qx"] == sp.Identity(lpf_qi_size[0]):
        simple_lpf = True
        lpf_qi_param_size = 0
        lpf_q = [np.identity(lpf_qi_size[0]), np.identity(lpf_qi_size[0])]

    total_opt_params = dimension * (ds_pi_param_size + lpf_qi_param_size + dlpf_dt_gi_param_size)
    assert len(parameters) == total_opt_params, \
        f'Wrong number of optimization parameters, {len(parameters)} != {total_opt_params}'

    # split the array into matrix parts and reshape them into the original form
    pointer = 0
    ds_p: List[np.ndarray] = \
        [parameters[pointer + dim * ds_pi_param_size : pointer + (dim + 1) * ds_pi_param_size].reshape(ds_pi_size)
         for dim in range(dimension)]

    pointer += dimension * ds_pi_param_size
    if not simple_lpf:
        lpf_q: List[np.ndarray] = \
            [parameters[pointer + dim * lpf_qi_param_size : pointer + (dim + 1) * lpf_qi_param_size].reshape(lpf_qi_size)
                for dim in range(dimension)]

    pointer += dimension * lpf_qi_param_size
    dlpf_dt_g: List[np.ndarray] = \
        [parameters[pointer + dim * dlpf_dt_gi_param_size : pointer + (dim + 1) * dlpf_dt_gi_param_size].reshape(dlpf_dt_gi_size)
            for dim in range(dimension)]

    if bl_diag:
        ds_p = block_diag(*ds_p)
        lpf_q = block_diag(*lpf_q)
        dlpf_dt_g = block_diag(*dlpf_dt_g)

    return ds_p, lpf_q, dlpf_dt_g


if __name__ == '__main__':
    # load a motion type
    positions_py, velocities_py = load_pylasa_data(motion_shape="N", plot_data=False)
    logger.info(f'Handwriting dataset loaded with [{positions_py.shape}, '
                f'{velocities_py.shape}] samples')

    # optimize for a sample of params
    funs, sols = optimize(trajectory=positions_py[:1000], velocity=velocities_py[:1000],
                          max_deg_ds=4, dimension=2, tol=0.01, max_deg_lpf=4,
                          method="trust-constr", quadratic_lpf=False, simplify_lpf=False)

    ds = funs["ds"]
    class ds_c:
        def predict(x):
            preds = np.apply_along_axis(ds, 1, x)
            return preds

    from utils.plot_tools import plot_ds_stream, plot_contours
    plot_ds_stream(ds_c, positions_py[:1000])
    plot_contours(funs["lpf"], positions_py)
    plot_contours(funs["dlpf_dt"], positions_py, color='Reds')