#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time
import numpy as np
import pickle

from sklearn.mixture import GaussianMixture
from policy_interface import PlanningPolicyInterface

from seds.gmm_fitting import fit
from seds.optimize_lpv_ds import optimize_lpv_ds_from_data
from utils.data_loader import generate_synthetic_nonlinear_data, load_pylasa_data
from utils.log_config import logger
from seds.optimize_se_ds import optimize_seds
from utils.utils import check_stability, is_negdef


class SE_DS(PlanningPolicyInterface):
    """ SEDS (Khansari et. al, 2012) original implemention reimplemented in Python.

    The class is responsible for constructing a LPV_DS object, and offering features
        to predict and fit dynamical systems using Gaussian Mixture Models and SQP optimization.
    """

    def __init__(self):
        """ Initialize a SE_DS object.
        """

        # gmm as obtained by gmm_fitting module
        self.__gmm: GaussianMixture = None

        # attributes for a SE-DS: x' = As(x) + bs
        self.__As: np.ndarray = None
        self.__bs: np.ndarray = None
        self.__k: int = 0
        self.__dimension: int = 2
        self.__fitting_time: float = 0.0
        self.__num_iterations = 0

    def fit(self, trajectory: np.ndarray, velocity: np.ndarray, tol: float = 0.00001,
        show_stats: bool = True, initial_guess: np.ndarray or None = None, is_linear: bool = False):
        """ Fit a GMM and extract parameters for the estimated dynamical systems.

        Args:
            trajectory (np.ndarray): Trajectory data in shape (sample size, dimension).
            velocity (np.ndarray): Velocity data in shape (sample size, dimension).
            tol (float, optional): Tolerance. Defaults to 0.00001.
            show_stats (bool, optional): Whether to show optimization stats.
                Defaults to True.

            initial_guess (np.ndarray, optional): This is used to warm-start the
                optimization process.

            is_linear (bool, optional): If set true, the gmm fitting step should only detect
                1 component. Defaults to False.
        """

        logger.info("Fitting GMM to the data")
        fitting_time_s = time.time()

        if self.__gmm is None:
            self.__gmm = fit(trajectory, is_linear)

        logger.info("Starting optimization process")
        self.__As, self.__bs, self.__num_iterations = optimize_seds(self.__gmm, trajectory,
                                                                    velocity, tol, initial_guess)

        fitting_time = time.time() - fitting_time_s

        # harvest the optimization results
        self.__k = self.__As.shape[0]
        self.__fitting_time = fitting_time

        # check negdefinity of matrices (13-b in the paper)
        posdef_count = 0
        for A_k in self.__As:
            if not is_negdef(A_k + A_k.T):
                posdef_count += 1

        # check equality condition of SEDS (13-a in the paper)
        equal_cons = 0
        for b_k in self.__bs:
            if np.linalg.norm(b_k) > 0.0001:
                equal_cons += 1

        if show_stats:
            logger.info(f'Fitting process finished {self.__num_iterations} iterations '
                f'in {fitting_time:.4f} seconds')
            logger.info(f'{self.__k} components detected by GMM')
            logger.info(f'As has the shape of {self.__As.shape} and bs {self.__bs.shape}')
            logger.info(f'Found {posdef_count} violations of SEDS negdef condition')
            logger.info(f'Found {equal_cons} violations of SEDS equality condition')


    def predict(self, trajectory: np.ndarray, stability_check: bool = True, stability_tol = 0.1):
        """ Predict estimated velocities from learning lpv-ds.

        Args:
            trajectory (np.ndarray): Trajectory in shape (sample size, dimension).
            plot (bool, optional): Whether to plot the results or not. Defaults to False.
            scale_factor (int, optional): Scaling the velocities. Defaults to 5.
            proportion (int, optional): Proportion of vectors to be shown as one in
                'proportion'. Defaults to 10.
            stability_check (bool, optional): Check for Lyapunov stability. Defaults to True.
            stability_tol (float, optional): Tolerance for stability violation. Defaults to 0.1.

        Returns:
            np.ndarray: Estimated velocities in shape (sample size, dimension).
        """

        # get the posterior probabilities
        posteriors = self.__gmm.predict_proba(trajectory)

        # define the attractor
        self.__dimension = trajectory.shape[1]
        attr = np.zeros((self.__dimension,1))

        result = []
        for i in range(len(trajectory)):
            x = trajectory[i].reshape(self.__dimension,1)
            weights = posteriors[i]
            result.append(self._estimate(x, weights))

        result = np.array(result)

        # check stability if True
        if stability_check:
            unstable = np.zeros(self.__dimension)

            # count the data points that violate the Lyapunov Stability conditions
            for k in range(len(trajectory)):
                x_ref = trajectory[k].reshape(self.__dimension, 1)
                x_dot = result[k].reshape(self.__dimension, 1)
                unstable += check_stability(x_ref, x_dot, attr, stability_tol)

            logger.debug(f'({unstable[0], unstable[1]}) data points '
                f'violated Lyapunov Stability conditions')

        return result

    def get_performance(self):
        """ Return the fitting time and number of iterations.
        """

        return (self.__fitting_time, self.__num_iterations)

    def get_ds_params(self):
        """ Return the As and bs parameters.
        """

        return self.__As, self.__bs

    def get_gmm_params(self):
        """ Return k and GMM object obtained from the GaussianMixture
        fitting process.
        """

        return self.__k, self.__gmm

    def set_gmm_params(self, k: int, gmm: GaussianMixture):
        """ Set a predefined value of k and mixture model.

        Args:
            k (int): Number of components in mixture models.
            gmm (GaussianMixture): The GMM object.
        """

        self.__k, self.__gmm = k, gmm

    def load(self, model_name: str, dir: str = '../res'):
        """ Load a previously stored model.

        Args:
            model_name (str): Model name.
            dir (str, optional): Path to the load directory. Defaults to '../res'.
        """

        with open(os.path.join(dir, f'{model_name}.pkl'), 'rb') as f:
            data = pickle.load(f)

        self.__gmm, self.__As, self.__bs, self.__k = data
        logger.info(f'Model {model_name} loaded with k = {self.__k} components')

    def save(self, model_name: str, dir: str = '../res'):
        """ Save the model for later use.

        Args:
            model_name (str): Model name.
            dir (str, optional): Path to the save directory. Defaults to '../res'.
        """

        os.makedirs(dir, exist_ok=True)

        data = [self.__gmm, self.__As, self.__bs, self.__k]
        with open(os.path.join(dir, f'{model_name}.pkl'), 'wb') as f:
            pickle.dump(data, f)

        logger.info(f'Model {model_name} saved')

    def _estimate(self, x: np.ndarray, weights: np.ndarray):
        """ Get the estimated velocity.

        Args:
            x (np.ndarray): Position as a vector.
            weights (np.ndarray): Array of weights for each component.

        Returns:
            np.ndarray: Estimated velocity.
        """

        vel = np.zeros(x.shape)
        for i in range(self.__k):
            vel += weights[i] * (self.__As[i].dot(x) + self.__bs[i])
        return vel.reshape(self.__dimension,)


class LPV_DS(PlanningPolicyInterface):
    """ Linear parameter-varying dynamical systems object. This class uses semi-definite
    programming as opposed to original SEDS.

    TODO: Developed in rush! Refactor and sanity checks are required later on.

    The class is responsible for constructing a LPV_DS object, and offering features
        to predict and fit dynamical systems using Gaussian Mixture Models and SQP optimization.
    """

    def __init__(self):
        """ Initialize a LPV_DS object.
        """

        # gmm as obtained by gmm_fitting module
        self.__gmm: GaussianMixture = None

        # attributes for a LPV-DS: x' = As(x) + bs
        self.__As: np.ndarray = None
        self.__bs: np.ndarray = None
        self.__k: int = 0
        self.__dimension: int = 2
        self.__fitting_time: float = 0.0
        self.__num_iterations = 0

    def fit(self, trajectory: np.ndarray, velocity: np.ndarray, tol: float = 0.00001,
        show_stats: bool = True, initial_guess: np.ndarray or None = None):
        """ Fit a GMM and extract parameters for the estimated dynamical systems. Note that unlike
        SEDS, the optimization here could be a semidefinite with nonlinear cost so the core
        functions can use cvxpy.MOSEK or other SDP programming libs.

        Args:
            trajectory (np.ndarray): Trajectory data in shape (sample size, dimension).
            velocity (np.ndarray): Velocity data in shape (sample size, dimension).
            tol (float, optional): Tolerance. Defaults to 0.00001.
            show_stats (bool, optional): Whether to show optimization stats.
                Defaults to True.

            initial_guess (np.ndarray, optional): This is used to warm-start the
                optimization process.

        """

        logger.info("Fitting GMM to the data")
        fitting_time_s = time.time()

        if self.__gmm is None:
            self.__gmm = fit(trajectory)

        logger.info("Starting optimization process")
        self.__As, self.__bs, self.__num_iterations = optimize_lpv_ds_from_data(self.__gmm,
                                                                     trajectory,
                                                                     velocity, tol,
                                                                     init=initial_guess)

        fitting_time = time.time() - fitting_time_s

        # harvest the optimization results
        self.__k = self.__As.shape[0]
        self.__fitting_time = fitting_time


        if show_stats:
            logger.info(f'Fitting process finished {self.__num_iterations} iterations '
                f'in {fitting_time:.4f} seconds')
            logger.info(f'{self.__k} components detected by GMM')
            logger.info(f'As has the shape of {self.__As.shape} and bs {self.__bs.shape}')


    def predict(self, trajectory: np.ndarray, stability_check: bool = True, stability_tol = 0.1):
        """ Predict estimated velocities from learning lpv-ds.

        Args:
            trajectory (np.ndarray): Trajectory in shape (sample size, dimension).
            plot (bool, optional): Whether to plot the results or not. Defaults to False.
            scale_factor (int, optional): Scaling the velocities. Defaults to 5.
            proportion (int, optional): Proportion of vectors to be shown as one in
                'proportion'. Defaults to 10.
            stability_check (bool, optional): Check for Lyapunov stability. Defaults to True.
            stability_tol (float, optional): Tolerance for stability violation. Defaults to 0.1.

        Returns:
            np.ndarray: Estimated velocities in shape (sample size, dimension).
        """

        # get the posterior probabilities
        posteriors = self.__gmm.predict_proba(trajectory)

        # define the attractor
        self.__dimension = trajectory.shape[1]
        attr = np.zeros((self.__dimension,1))

        result = []
        for i in range(len(trajectory)):
            x = trajectory[i].reshape(self.__dimension,1)
            weights = posteriors[i]
            result.append(self._estimate(x, weights))

        result = np.array(result)

        # check stability if True
        if stability_check:
            unstable = np.zeros(self.__dimension)

            # count the data points that violate the Lyapunov Stability conditions
            for k in range(len(trajectory)):
                x_ref = trajectory[k].reshape(self.__dimension, 1)
                x_dot = result[k].reshape(self.__dimension, 1)
                unstable += check_stability(x_ref, x_dot, attr, stability_tol)

            logger.debug(f'({unstable[0], unstable[1]}) data points '
                f'violated Lyapunov Stability conditions')

        return result

    def _estimate(self, x: np.ndarray, weights: np.ndarray):
        """ Get the estimated velocity.

        Args:
            x (np.ndarray): Position as a vector.
            weights (np.ndarray): Array of weights for each component.

        Returns:
            np.ndarray: Estimated velocity.
        """

        vel = np.zeros(x.shape)
        for i in range(self.__k):
            vel += weights[i] * (self.__As[i].dot(x) + self.__bs[i])
        return vel.reshape(self.__dimension,)


def expert_seds_model(motion_shape: str = "G", save_dir: str = "../res"):
    """ Learn and save an SEDS model based on handwriting demonstrations.

    Args:
        motion_shape (str, optional): Shape of the trajectory. Defaults to "G".
    """

    positions_py, velocities_py = load_pylasa_data(motion_shape=motion_shape)

    generator_model = SE_DS()
    generator_model.fit(positions_py, velocities_py)

    logger.info(f'Learning completed, saving the model')
    os.makedirs(save_dir, exist_ok=True)

    generator_model.save(model_name=f'{motion_shape.lower()}_seds', dir=dir)


def prepare_expert_data(motion_shape: str, n_dems: int, n_dems_initial: int = 7, dir: str = "",
                        noise_level: float = 0.02, normalized: bool = True):
    """ Augmenting the dataset to get more samples for methods that use extensive expert data such as GAIL, or BC. The expert here is SEDS method.

    Args:
        motion_shape (str): Shape of the trajectory as in LASA dataset.
        n_dems (int): Number of augmented demonstrations.
        n_dems_initial (int, optional): Number of initial demonstrations in the dataset,
            for LASA it's always 7. Defaults to 7.
        dir (str, optional): Choose the directory to load the previously stored model.
            If no saved seds model found, a new one should be trained and stored first.
    """

    init_trajs, init_vels = load_pylasa_data(motion_shape=motion_shape,
                                             normalized=normalized)
    generator_model = SE_DS()

    if normalized:
        logger.warn(f'Make sure the expert DS is trained on normalized data!')

    try:
        generator_model.load(model_name=f'{motion_shape.lower()}_seds', dir=dir)
    except Exception:
        # learn demonstration if not available already
        expert_seds_model(motion_shape, save_dir=dir)
        generator_model.load(model_name=f'{motion_shape.lower()}_seds', dir=dir)

    return generate_synthetic_nonlinear_data(generator_model, init_trajs,
        init_vels, n_dems_generate=n_dems, n_dems_initial=n_dems_initial,
        noise_level=noise_level)
