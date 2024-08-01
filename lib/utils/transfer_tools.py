#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
import numpy as np

from typing import Dict, Tuple
from sklearn.model_selection import ShuffleSplit

sys.path.append(os.pardir)
from learn_gmm_ds import SE_DS
from utils.log_config import logger
from utils.utils import mse
from utils.plot_tools import plot_ds_stream


def cross_validate(src_trajectory: np.ndarray, src_velocity: np.ndarray,
        trg_trajectory: np.ndarray, trg_velocity: np.ndarray,
        num_of_splits: int = 10, tol: float = 0.00001):
    """ Cross validate the training process on target data using a
    source dataset.

    Args:
        src_trajectory (np.ndarray): Source trajectory in shape (sample size, dimension).
        src_velocity (np.ndarray): Source velocity in shape (sample size, dimension).
        trg_trajectory (np.ndarray): Target trajectory in shape (sample size, dimension).
        trg_velocity (np.ndarray): Target velocity in shape (sample size, dimension).
        num_of_splits (int, optional) CV splits as an integer. Defaults to 10.
        tol (float, optional): Learning process tolerance. Defaults to 0.00001.

    Returns:
        float, float: mean and std of the CV process.
    """
    rs = ShuffleSplit(num_of_splits)

    # sum of mse for all the folds
    error = []

    # initialize a lpvds object
    model = SE_DS()

    for train_index, test_index in rs.split(src_trajectory):

        tra_train = trg_trajectory[train_index]
        vel_train = trg_velocity[train_index]

        tra_test = src_trajectory[test_index]
        vel_test = src_velocity[test_index]

        # fit the model
        model.fit(tra_train, vel_train, tol)

        # get the result of prediction and error
        vel_predicted = model.predict(tra_test)
        error.append(mse(vel_predicted, vel_test))

    error = np.array(error)
    return np.mean(error), np.std(error)


def transfer_retrain(src_model: SE_DS, target_trajectories: np.ndarray,
        target_velocities: np.ndarray, tol = 0.00001, threshold = 1.0,
        is_linear: bool = True, show_plots: bool = False) -> SE_DS:
    """ Having a source model, this function trains an LPVDS on a target
    using warm start.

    Args:
        src_model (LPV_DS): The source model trained on high quality demonstrations.
        target_trajectories (np.ndarray): Target task demonstrated trajectories.
        target_velocities (np.ndarray): Target task demonstrated velocities.
        tol (float, optional): Tolerance for the LPVDS. Defaults to 0.00001.
        show_plots (bool, optional): Whether to show plots. Defaults to False.

    Returns:
        LPV_DS: The trained model for the target task.
    """
    # define a target model
    trg_model = SE_DS()

    # get the ds params of src model
    As_src, bs_src = src_model.get_ds_params()
    k_src, gmm_src = src_model.get_gmm_params()
    logger.info(f'Acquired source params As {As_src.shape}, and bs {bs_src.shape}')

    # train with an initial guess
    initial_param = np.concatenate((As_src.flatten(), bs_src.flatten()))

    # TEMPORARY IDEA: match the exact GMM
    trg_model.set_gmm_params(k_src, gmm_src)

    trg_model.fit(trajectory=target_trajectories, velocity=target_velocities,
        initial_guess=initial_param, is_linear=is_linear, tol=tol)

    if show_plots:
        plot_ds_stream(trg_model, target_trajectories, "Retrained DS Model")

    return trg_model


def evaluate_transfer_retrain(motions_data: Dict[str, np.ndarray],
        transfer_map: Dict[str, str], n_train_dem: int, n_test_dem: int, n_limit_dem: int,
        sample_per_dem: int, is_linear: bool = False, show_plots: bool = False) -> Dict[str, Tuple]:
    """ Apply the transfer retrain method to a dataset of demonstrations. A transfer map is
    provided to facilitate the use of transfer_retrain function between source and target.

    # TODO: perform a random split later on using n_test_dem.

    Args:
        motions_data (Dict[str, np.ndarray]): The demonstrations accessible by a key name.
        transfer_map (Dict[str, str]): A key-value setup to show between which tasks in motion_data
            the transfer attempt should happen. For instance, an entry like 'C-shaped':'G-shaped' means
            that the reference model is trained on 'C-shaped' motion and the weights are used to
            learn a transfer model for reproducing 'G-shaped' motion.

        n_train_dem (int): Number of training demonstrations.
        n_test_dem (int): Number of test demonstrations.
        n_limit_dem (int): Size of the limited dataset in terms of number of demonstrations.
        sample_per_dem (int): Number of samples per demonstration. Should be equal at this point.
    """

    motion_shapes = motions_data.keys()

    split_idx = n_train_dem * sample_per_dem
    limit_idx = n_limit_dem * sample_per_dem
    motions_data_train, motions_data_test = dict(), dict()
    motions_data_limit: Dict[str, np.ndarray] = dict()

    for motion_shape in motion_shapes:
        motions_data_train[motion_shape] = (motions_data[motion_shape][0][:split_idx],
            motions_data[motion_shape][1][:split_idx])

        motions_data_test[motion_shape] = (motions_data[motion_shape][0][split_idx:],
            motions_data[motion_shape][1][split_idx:])

        motions_data_limit[motion_shape] = (motions_data[motion_shape][0][:limit_idx],
            motions_data[motion_shape][1][:limit_idx])

    logger.info(f'Dataset splitted for {motions_data_train.keys()} motions')

    # train reference DS for each motion type
    print('\n============================ Training reference DS =============================')

    reference_ds = {shape: SE_DS() for shape in motion_shapes}
    for motion_shape in motion_shapes:
        reference_ds[motion_shape].fit(*motions_data_train[motion_shape], is_linear=is_linear)

    print('==================================================================================\n')

    # train limited DS with single demonstration
    print('\n============================ Training limited DS ===============================')

    partial_ds = {x: SE_DS() for x in motion_shapes}
    for motion_shape in motion_shapes:
        partial_ds[motion_shape].fit(*motions_data_limit[motion_shape], is_linear=is_linear)

    print('==================================================================================\n')

    # train transfer DS with limited data but a warm-start
    print('\n============================ Training transfer DS ==============================')

    transfer_ds = {x: SE_DS() for x in motion_shapes}
    for motion_shape in motion_shapes:
        target_motion = transfer_map[motion_shape]
        transfer_ds[target_motion] = transfer_retrain(reference_ds[motion_shape],
            *motions_data_limit[target_motion], is_linear=is_linear, show_plots=show_plots)
    print('==================================================================================\n')

    ''' Compare reference, transfer, and partial DS '''
    print('\n============================ Final transfer results ============================')
    results_dict: Dict[str, Tuple] = dict()
    for motion_shape in motion_shapes:
        test_traj, test_vel = motions_data_test[motion_shape]

        partial_pred = partial_ds[motion_shape].predict(test_traj)
        transfer_pred = transfer_ds[motion_shape].predict(test_traj)
        reference_pred = reference_ds[motion_shape].predict(test_traj)

        partial_mse = mse(partial_pred, test_vel)
        transfer_mse = mse(transfer_pred, test_vel)
        reference_mse = mse(reference_pred, test_vel)

        logger.info(f'MSE for {motion_shape}-shaped partial DS is {partial_mse} and '
            f'fitting time is {partial_ds[motion_shape].get_performance()}')
        logger.info(f'MSE for {motion_shape}-shaped transfer DS is {transfer_mse} and '
            f'fitting time is {transfer_ds[motion_shape].get_performance()}')
        logger.info(f'MSE for {motion_shape}-shaped reference DS is {reference_mse} and '
            f'fitting time is {reference_ds[motion_shape].get_performance()}.\n')

        results_dict[motion_shape] = {"reference_mse": reference_mse,
            "partial_mse": partial_mse, "transfer_mse": transfer_mse,
            "reference_time": reference_ds[motion_shape].get_performance(),
            "transfer_time": transfer_ds[motion_shape].get_performance(),
            "partial_time": partial_ds[motion_shape].get_performance()}
    print('=================================================================================\n')

    return results_dict
