#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import cvxpy as cp

from lib.policy_interface import PlanningPolicyInterface

from lib.sos.optimize_plyds_cvxpy import optimize as optimize_c
from lib.sos.optimize_plyds_scipy import optimize as optimize_s

from lib.utils.log_config import logger


class PLY_DS(PlanningPolicyInterface):
    """ Approximation of a dynamical system using a polynomial approach.

    Global asymptotic stability is ensured using a SOS optimization technique and nonlinear optimization.

    Since a DS dataset can be seen as a time series data, with velocities acting as labels, we seek
    to find a Sum of Squares polynomial to estimate the DS and later ensure stability.
    """

    def __init__(self, max_deg_ply: int = 4, max_deg_lpf: int = 2, data_dim: int = 2,
                 drop_out: float = 0.0):
        """ Initialize a nonlinear DS estimator.

        Args:
            max_deg_ply (int, optional): Maximum degree of the polynomial.
            max_deg_lpf (int, optional): Maximum degree of the lyapunov function.

            data_dim (int, optional): Dimension of the input data. Defaults to 2.
        """

        self.__dim = data_dim
        self.__max_deg_ply = max_deg_ply
        self.__max_deg_lpf = max_deg_lpf
        logger.info(f'{self.__max_deg_ply}-degree polynomial with {self.__max_deg_lpf}-degree '
                    f'lyapunov function initialized')

        # initialize a polynomial transformer
        self.__function_dict = None
        self.__solution_dict = None
        self.__drop_out = drop_out

    def fit(self, trajectory: np.ndarray, velocity: np.ndarray, trajectory_test: np.ndarray = None,
        velocity_test: np.ndarray = None, optimizer: str = "scipy", method: str = "SLSQP",
        tol: float = 1e-4, quadratic_lpf: bool = True, simplify_lpf: bool = False):
        """ Fit a polynomial model to estimate a dynamical systems.

        Args:
            trajectory (np.ndarray): Trajectory data in shape (samples, features).
            velocity (np.ndarray): Velocity data in shape (samples, features).

            trajectory_test (np.ndarray, optional): Test data points. Defaults to None.
            velocity_test (np.ndarray, optional): Test data labels. Defaults to None.
            tol (float, optional): Tolerance for nonlinear conditions, optimization and so on.
                Default to 0.01 but could change baced on sensetivity. Pass an integer between
                [1, 10] for CVXPY MOSEK to tighten the tolerance.
        """

        # apply the drop out
        if self.__drop_out != 0.0:
            kept_indices = np.random.choice(a=len(trajectory), replace=False,
                                            size=int((1 - self.__drop_out) * len(trajectory)))
            trajectory = np.array(trajectory[kept_indices])
            velocity = np.array(velocity[kept_indices])

        # launch the optimizer
        if optimizer == "scipy":
            functions, solutions = optimize_s(trajectory, velocity, self.__max_deg_ply, self.__dim,
                self.__max_deg_lpf, method=method, tol=tol, simplify_lpf=simplify_lpf, quadratic_lpf=quadratic_lpf)

        if optimizer == "cvxpy":
            functions, solutions = optimize_c(trajectory, velocity,
                    self.__max_deg_ply, self.__dim, self.__max_deg_lpf, tol=tol, method=cp.MOSEK)

        self.__function_dict = functions
        self.__solution_dict = solutions

    def predict(self, trajectory: np.ndarray):
        """ Predict estimated velocities from learning PLY_DS.

        Args:
            trajectory (np.ndarray): Trajectory in shape (sample size, dimension).

        Returns:
            np.ndarray: Estimated velocities in shape (sample size, dimension).
        """
        assert self.__function_dict is not None, 'Fit method must be called first!'

        pred_velocity = np.apply_along_axis(self.__function_dict["ds"], 1, trajectory)
        return pred_velocity

    def load(self, model_name: str, dir: str = None):
        """ Load the torch model.

        Args:
            model_name (str): Name of the model.
            dir (str, optional): Load directory. Defaults to '../res'.
        """

        self.__function_dict, self.__solution_dict = pickle.load(open(os.path.join(dir,
            f'{model_name}.pickle'), 'rb'))

    def save(self, model_name: str, dir: str = None):
        """ Save the torch model.

        Args:
            model_name (str): Name of the model.
            dir (str, optional): Save directory. Defaults to '../res'.
        """

        os.makedirs(dir, exist_ok=True)

        pickle.dump(self.get_main_functions(), open(os.path.join(dir, f'{model_name}_ply{self.__max_deg_ply}_lpf{self.__max_deg_lpf}.pickle'), 'wb'))

    def get_solutions(self):
        """Return the acquired solutions.

        Returns:
            Dict: Solutions to the optimization problem.
        """

        return self.__solution_dict

    def get_main_functions(self):
        """Get the main resulting functions.s

        Returns:
            Tuple(DS, LPF, D(LPF)/DT): Dynamical system, lyapunov function, and its derivative.
        """

        return tuple(self.__function_dict.values())