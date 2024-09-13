#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

from typing import Tuple, Union
from tqdm.auto import tqdm

import numpy as np
from scipy.io import loadmat
import h5py

sys.path.append(os.pardir)
from lib.policy_interface import PlanningPolicyInterface
from lib.utils.log_config import logger
from lib.utils.utils import calibrate, is_negdef, linear_ds, normalize

import pyLasaDataset as hw_data_module
lasa_selected_motions = ["GShape", "PShape", "Sine", "Worm", "Angle", "CShape", "NShape", "DoubleBendedLine"]
lasa_dataset_motions = hw_data_module.dataset.NAMES_

# dataset configs
N_SAMPLES_LASA_HANDWRITING = 1000
N_DEMONSTRATIONS_LASA_HANDWRITING = 7

def load_hdf5_data(
        dataset: str = "../data/",
        demo_id = 0,
        waypoints_dataset_name = "AWE_waypoints",
        subgoals_dataset_name = "AWE_subgoals",
        subgoal = 0
    ):
    """
    A function to load the observations from a hdf5 dataset file.
    Notes:
        - structure of the hdf5 file:
            data
                demo_0
                    abs_actions (NOTE : abs_actions = np.concatenate([action_pos, action_ori, action_gripper], axis=-1) absolute action is the state of the ee in the demo at time step t)
                    AWE_waypoints (list of waypoint indices)
                    AWE_subgoals (list of subgoal indices)
                    ...
                demo_1
                    ...
                ...
        - the reconstructed trajectories are obtained by linearly interpolating the end-effector positions of the waypoints (125 actions between each waypoint)
    Args:
        dataset (str): The path to the hdf5 dataset file.
        demo_id (int): The index of the demonstration to load.
        waypoints_dataset_name (str): The name of the dataset containing the waypoints.
        subgoals_dataset_name (str): The name of the dataset containing the subgoals.
        subgoal (int): The index of the subgoal to load the data from.
    Returns:
        np.array: The end-effector positions of the waypoints in the segment.
        np.array: The end-effector velocities of the waypoints in the segment.
        np.array: The end-effector orientations of the waypoints in the segment.
        np.array: The gripper actions of the waypoints in the segment.
    """

    f = h5py.File(dataset, 'r')

    demo_waypoints = f[f'data/demo_{demo_id}/{waypoints_dataset_name}']
    #print("demo_waypoints", demo_waypoints)

    subgoals = f[f'data/demo_{demo_id}/{subgoals_dataset_name}']

    segment_waypoints = []

    # define the start and end of the segment
    seg_start = [subgoals[subgoal-1] if subgoal > 0 else 0]
    seg_end = subgoals[subgoal]

    i = 0
    # get the waypoints in the segment
    for i in range(len(demo_waypoints)):
        if demo_waypoints[i] >= seg_start[0] and demo_waypoints[i] <= seg_end:
            segment_waypoints.append(demo_waypoints[i])
        if demo_waypoints[i] > seg_end:
            break

    # get the vel and pos data for each waypoint in the segment
    vel_data = []
    pos_data = []
    euler_data = []
    gripper_data = []
    for waypoint in segment_waypoints:
        vel_data.append(f[f'data/demo_{demo_id}/obs/robot0_eef_vel_lin'][waypoint])
        pos_data.append(f[f'data/demo_{demo_id}/obs/robot0_eef_pos'][waypoint])
        euler_data.append(f[f'data/demo_{demo_id}/sim_euler_angles'][waypoint])
        gripper_data.append(f[f'data/demo_{demo_id}/abs_actions'][waypoint][-1])
        # can get orientation with f[f'data/demo_{demo_id}/obs/robot0_eef_quat'][waypoint]
        # can get gripper action with f[f'data/demo_{demo_id}/abs_action'][waypoint][-1] (the value will be either 1 for open or -1 for close)

    f.close()
    # convert to numpy arrays and return

    return np.array(pos_data), np.array(vel_data), np.array(gripper_data), np.array(euler_data)


def load_snake_data(data_dir: str = "../data/"):
    """ Load LASA dataset when it's in raw mat form from a specific dir.

    Note: Leave the data_dir parameter to None in order to load from default dir.

    Args:
        motion_shape (str): Choose the type of motion to be loaded. A list
            of available motion shapes can be found in this file.
        data_dir (str or None): Path to the data files. Defaults to None.

    Returns:
        Tuple(np.ndarray, np.ndarray): positions, velocities
    """

    # load the raw data
    raw = loadmat(os.path.join(data_dir, f'Messy_snake.mat'))

    # strip out the data portion
    data = raw["Data"]

    # calculate dimension of the data
    dimension = 2

    pos = data[0: dimension]
    vel = data[dimension:]

    pos = normalize(calibrate(pos))
    vel = normalize(vel)

    return (pos.T, vel.T)


def load_mat_data(motion_shape: str, data_dir: Union[str, None]= None):
    """ Load LASA dataset when it's in raw mat form from a specific dir.

    Note: Leave the data_dir parameter to None in order to load from default dir.

    Args:
        motion_shape (str): Choose the type of motion to be loaded. A list
            of available motion shapes can be found in this file.
        data_dir (str or None): Path to the data files. Defaults to None.

    Returns:
        Tuple(np.ndarray, np.ndarray): positions, velocities
    """

    if data_dir is None:
        data_dir = os.path.join(os.pardir, os.pardir, 'data', 'mat')

    # load the raw data
    raw = loadmat(os.path.join(data_dir, f'{motion_shape}.mat'))

    # strip out the data portion
    data = raw["data"].reshape(-1)

    # calculate dimension of the data
    dimension = data[0].shape[0]//2
    logger.info(f'Data dimension is {dimension}')

    # strip out the trajectory and velocity portions
    pos_list = list()
    vel_list = list()

    for demonstration in data:
        pos = demonstration[0: dimension]
        vel = demonstration[dimension:]
        assert len(pos) == len(vel), f'Size mismatch in {motion_shape} data'

        pos_list.append(calibrate(pos))
        vel_list.append(vel)

    # concatenate the results
    concatenated_pos = np.concatenate(pos_list, axis=1)
    concatenated_vel = np.concatenate(vel_list, axis=1)

    return (concatenated_pos.T, concatenated_vel.T)


def load_pylasa_data(motion_shape: str = "Angle", plot_data: bool = False,
    calibrated: bool = True, normalized: bool = True, n_dems: int = 10):
    """ Facilitate the handling of LASA handwriting dataset.

    Refer to https://bitbucket.org/khansari/lasahandwritingdataset/src/master/Readme.txt for
        more info about the dataset and attributes.

    In a quick glance, the dataset objects look like this:
        dt: the average time steps across all demonstrations
        demos: A structure variable containing necessary information
            about all demonstrations. The variable 'demos' has the following
            format:
            - demos{n}: Information related to the n-th demonstration.
            - demos{n}.pos (2 x 1000): matrix representing the motion in 2D
                space. The first and second rows correspond to
                x and y axes in the Cartesian space, respectively.
            - demons{n}.t (1 x 1000): vector indicating the corresponding time
                for each data point.
            - demos{n}.vel (2 x 1000): matrix representing the velocity of the motion.
            - demos{n}.acc (2 x 1000): matrix representing the acceleration of the motion.

    Args:
        motion_shape (str, optional): Choose a motion shape. A list of possible options
            may be found in this file. Defaults to "Angle". Possible options are [Angle, GShape, CShape,
            BendedLine, JShape, Multi_Models_1 to Multi_Models_4, Snake, Sine, Worm, PShape, etc.]. Complete list
            can be found in hw_data_module.dataset.NAMES_.

        plot_data (bool, optional): Whether to plot the designated motion or not. Defaults to False.

    Raises:
        NotImplementedError: Raised if the motion demonstrations are not available in dataset.

    Returns:
        Tuple(np.ndarray, np.ndarray): positions, velocities
    """

    # list of tested motion data
    if motion_shape == "MessySnake":
        pos, vel = load_snake_data()
        return pos, vel
    else:
        data = getattr(hw_data_module.DataSet, motion_shape)

    # extract pos and vel data
    pos_list = list()
    vel_list = list()

    for dem_index, demo in enumerate(data.demos):
        calibrated_pos = calibrate(demo.pos) if calibrated else demo.pos
        normalized_vel = normalize(demo.vel) if normalized else demo.vel
        normalized_pos = normalize(calibrated_pos) if normalized else calibrated_pos

        demo.pos = normalized_pos
        demo.vel = normalized_vel

        pos_list.append(normalized_pos)
        vel_list.append(normalized_vel)

        if dem_index + 1 == n_dems:
            logger.info(f'Stopping at maximum {n_dems} demonstrations')
            break

    if plot_data:
        hw_data_module.utilities.plot_model(data)

    # concatenate the results
    concatenated_pos = np.concatenate(pos_list, axis=1)
    concatenated_vel = np.concatenate(vel_list, axis=1)

    return concatenated_pos.T, concatenated_vel.T


def generate_synthetic_linear_data(A: np.matrix, start_point: Tuple[float, float],
        target_point: Tuple[float, float], n_dems: int = 10, n_samples = 750):

    """ Generate synthetic data in the form of uniformly distributed points from a
    reference dynamical system.

    Assuming a linear dynamical system described by:

                                x_dot = A * x + b + epsilon

    this function samples some points uniformly scattered in the state space and
    uses Gaussian additive noise to model the randomness of the real-world data.

    Args:
        A (np.matrix): Matrix A of the dynamical system ODE model.

        start_point (Tuple[float, float]): start point of the demonstrated trajectories.
        target_point (Tuple[float, float]): target point of the system.
        n_dems (int, optional): Number of samples or demonstrations to generate.
            Defaults to 10.

        n_samples (int, optional): Number of data points or (x_dot, x) pairs in
            each demonstration. Defaults to 750.
    """

    # check if the matrix A complies with stability conditions
    assert is_negdef(A + A.T), "Stability constraint for matrix A is violated!"

    # calculate b based on stability conditions
    b = np.dot(-A, np.array(target_point).T)

    # noise dist and samples
    noise_dist = np.random.uniform
    total_samples = n_dems * n_samples

    synthetic_trajectories = noise_dist(low=[start_point[0], start_point[1]],
        high=[target_point[0], target_point[1]], size=(total_samples, 2))

    synthetic_velocities = np.array([linear_ds(A, x, b) for x in synthetic_trajectories])
    synthetic_velocities = synthetic_velocities.reshape(*synthetic_trajectories.shape)

    return synthetic_trajectories, synthetic_velocities


def generate_synthetic_nonlinear_data(policy: PlanningPolicyInterface,
                                      initial_trajectories: np.ndarray,
                                      initial_velocities: np.ndarray,
                                      n_dems_generate: int = 10, n_samples = 1000,
                                      noise_level: float = 0.01, n_dems_initial: int = 7):

    """ Generate synthetic data in the form of uniformly distributed points from a
    reference nonlinear dynamical system.

    Assuming a linear dynamical system described by:

                                         x_dot = f(x)

    this function samples some points uniformly scattered in the state space and
    uses Gaussian additive noise to model the stochasticness of the real-world data.

    Args:
        lpv_ds (LPV_DS): An linear parameter-varying dynamical system found using the SEDS method.
            This module is used to produce more trajectories in order to directly estimate
            a nonlinear dynamical system.

        initial_trajectories (np.ndarray): Previously demonstrated trajectories of LASA Handwriting Dataset.
        initial_velocities (np.ndarray): Previously demonstrated velocities of LASA Handwriting Dataset.

        n_dems_generate (int, optional): Number of samples or demonstrations to generate.
            Defaults to 10. Includes the original samples.
        n_samples (int, optional): Number of data points or (x_dot, x) pairs in
            each demonstration. Defaults to 1000.

        noise_level (float, optional): Level of integrated noise into the trajectory points. Except for the goal.
            Defaults to 0.1.
        n_dems_initial (int, optional): Number of demonstrations in initial dataset.
    """

    # check that same data augmentation shape is expected
    assert n_samples == initial_trajectories.shape[0] / n_dems_initial
    aug_vels = np.zeros(shape=((n_dems_generate) * n_samples, 2))
    aug_trajs = np.zeros(shape=((n_dems_generate) * n_samples, 2))

    if n_dems_generate <= n_dems_initial:
        return initial_trajectories[:n_dems_generate * n_samples, :], \
               initial_velocities[:n_dems_generate * n_samples, :]

    for dem_idx in (par := tqdm(range(n_dems_generate - n_dems_initial))):
        par.set_description('Augmenting demonstrations')

        initial_traj_id = np.random.randint(n_dems_initial)
        dem_start_idx, dem_end_idx = initial_traj_id * n_samples, (initial_traj_id + 1) * n_samples

        sample_traj = initial_trajectories[dem_start_idx: dem_end_idx]

        goal_point = sample_traj[-1]
        sample_traj = sample_traj[:-1]

        noise_array = np.random.uniform(-noise_level, noise_level, sample_traj.shape)
        sample_traj += noise_array * sample_traj
        sample_traj = np.append(sample_traj, [goal_point], axis=0)

        sample_vel = policy.predict(sample_traj)

        aug_trajs[dem_idx * n_samples: (dem_idx + 1) * n_samples] = sample_traj
        aug_vels[dem_idx * n_samples: (dem_idx + 1) * n_samples] = sample_vel

    aug_trajs[(dem_idx + 1) * n_samples:] = initial_trajectories
    aug_vels[(dem_idx + 1) * n_samples:] = initial_velocities

    logger.info(f'Generated {n_dems_generate} demonstrations with {n_samples} samples each, \n'
                f'total of {aug_trajs.shape[0]} samples')
    return aug_trajs, aug_vels


def generate_synthetic_data_trajs(A: np.matrix, start_point: Tuple[float, float],
        goal_point: Tuple[float, float], n_demonstrations: int = 10, n_samples = 750,
        start_dev: float = 0.1, goal_dev: float = 0.0, traj_dev: float = 0.1):
    """ Generate synthetic data in the form of uniform trajectories.

    Assuming a linear dynamical system described by:

                                    x_dot = A * x + b

    this function produces synthetic trajectories given and uses Gaussian
    additive noise to model the randomness of the real-world data.

    The final data is generated based on the following process:

                                    x_dot = A * x + b + epsilon

    Note that due to the following stability guarantees, b is defined based on
    A and target:
                        A + A^T < 0 (negative definite)
                        b = -A . x* (where x* is the target or goal)

    Args:
        A (np.matrix): Matrix A of the dynamical system ODE model.

        start_point (Tuple[float, float]): start point of the demonstrated trajectories.
        goal_point (Tuple[float, float]): start point of the demonstrated trajectories.

        n_demonstrations (int, optional): Number of samples or demonstrations to generate.
            Defaults to 10.

        n_samples (int, optional): Number of data points or (x_dot, x) pairs in
            each demonstration. Defaults to 750.

        start_dev (float): Deviation of starting point for different trajectories.
            Defaults to 0.1.
        end_dev (float): Deviation of the goal point for different trajectories.
            Defaults to 0.0.
        traj_dev (float): Deviation of the data points between start and goal for
            different trajectories. Defaults to 0.0.
    """
    # check if the matrix A complies with stability conditions
    assert is_negdef(A + A.T), "Stability constraint for matrix A is violated!"

    # calculate b based on stability conditions
    b = np.dot(-A, np.array(goal_point).T)

    # define noise and ds equation
    noise_dist = np.random.uniform

    synthetic_trajectories = np.zeros((n_demonstrations * n_samples, 2))
    synthetic_velocities = np.zeros((n_demonstrations * n_samples, 2))

    for sample_idx in range(n_demonstrations):
        synthetic_traj = np.zeros((n_samples, 2), dtype=np.float64)
        synthetic_vel = np.zeros((n_samples, 2), dtype=np.float64)

        # set start and goal points based on the deviations
        synthetic_traj[0] = np.array(start_point) + noise_dist(-start_dev, +start_dev, 2)
        synthetic_vel[0] = linear_ds(A, synthetic_traj[0], b)

        # if ds is stable at the goal, the velocity should be 0.
        synthetic_traj[n_samples - 1] = np.array(goal_point) + noise_dist(-goal_dev, +goal_dev, 2)
        synthetic_vel[n_samples - 1] = np.array([0.0, 0.0])

        # get the line equation
        start, end = np.array(synthetic_traj[0]), np.array(synthetic_traj[n_samples - 1])
        slope = (end - start)[1] / (end - start)[0]
        intercept = end[1] - slope * end[0]
        y = lambda x: slope * x + intercept

        # construct the rest of n_samples - 2 points evenly
        x_samples = np.linspace(start[0], end[0], num=n_samples - 1, endpoint=False)

        # set the beginning
        for point_idx in range(1, n_samples - 1):
            traj_point = np.array([x_samples[point_idx], y(x_samples[point_idx])])
            synthetic_traj[point_idx] = traj_point + noise_dist(-traj_dev, +traj_dev, 2)
            synthetic_vel[point_idx] = linear_ds(A, synthetic_traj[point_idx], b)

        # fill the main arrays
        synthetic_velocities[sample_idx * n_samples: (sample_idx + 1) * n_samples] = synthetic_traj
        synthetic_trajectories[sample_idx * n_samples: (sample_idx + 1) * n_samples] = synthetic_vel

    return synthetic_trajectories, synthetic_velocities
