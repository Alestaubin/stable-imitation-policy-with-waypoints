#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import torch

import numpy as np

from typing import List, Optional, Union

from lib.learn_nn_ds import NL_DS
from lib.learn_ply_ds import PLY_DS
from lib.utils.utils import time_stamp
from lib.utils.log_config import logger
from lib.utils.plot_tools import plot_ds_2Dstream, plot_trajectory, plot_contours
from lib.utils.data_loader import load_hdf5_data

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def waypoint_policy(learner_type: str,
                    waypoint_positions: np.ndarray,
                    waypoint_velocities: np.ndarray,
                    n_epochs: int,
                    plot: Optional[bool] = False,
                    model_name: Optional[str] = 'waypoint-test',
                    save_dir: str = 'res/',
                    gpu: Optional[bool] = True if torch.cuda.is_available() else False):

    """  Train a stable/unstable policy to learn a nonlinear dynamical system.

    Args:
        learner_type(str): Type of the nonlinear estimator, could be "nn" (unstable), "snds", or
            "sdsef".

        n_epochs (int): Total number of epochs.
        plot (bool, Optional): Whether to plot trajectories and final ds or not.
            Default is True.

        model_name (str, Optional): Name of the model for save and load.
            Default is 'waypoint-test'.

        save_dir (str, Optional): Files will be saved in this directory if not None.
        gpu (bool, Optional): Activate gpu computation. Defaults to True.
    """

    # model and plot names
    name = f'{model_name}-{learner_type}-{time_stamp()}'

    # set the goal point as the last point
    goal = waypoint_position[-1].reshape(1, waypoint_position.shape[1])

    # NOTE: Choose between 3 different models: "sdsef", "nn", "snds" (ours).
    # "snds" uses a neural representation, "sdsef" uses diffeomorphism.
    # "nn" is just unstable behavioral cloning.
    if learner_type in ["snds", "nn", "sdsef", "lnet"]:
        model = NL_DS(network=learner_type, data_dim=waypoint_position.shape[1], goal=goal, gpu=gpu,
                      eps=0.2, alpha=0.1) # NOTE: You might need to play with these values of eps and alpha to tune the model
        model.fit(waypoint_positions, waypoint_velocities, n_epochs=n_epochs, lr_initial=1e-4)

    else:
        raise NotImplementedError(f'Learner type {learner_type} not available!')

    # plot the resulting ds and lpf
    if plot:
        plot_ds_2Dstream(model, waypoint_positions, save_dir=save_dir, file_name=f'{name}-ds', show_rollouts=True)

        if model.lpf(waypoint_position[-1].reshape(1, waypoint_position.shape[1])) is not None:
            plot_contours(model.lpf, waypoint_positions, save_dir=save_dir, file_name=f'{name}-lpf')

    # save the model
    model.save(model_name=name, dir=save_dir)

    # return
    return model

# Main entry NOTE: Your code here to load waypoints and train a stable DS policy
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Nonlinear DS experiments CLI interface.')

    # general params
    parser.add_argument('--learner-type', type=str, default="snds",
                        help='Policy learning method among snds, nn, plyds.')
    parser.add_argument('--num-epochs', type=int, default=10000,
                        help='Number of training epochs.')
    parser.add_argument('--data-dir', type=str,
                        default="data/KITCHEN_SCENE1_put_the_black_bowl_on_the_plate/image_demo_local_with_AWE_waypoints.hdf5",
                        help='Directory for the waypoints data.')
    args = parser.parse_args()

    waypoint_position, waypoint_velocity = load_hdf5_data(dataset=args.data_dir,
                                                          demo_id=1,
                                                          waypoints_dataset_name="waypoints_AWE_waypoints_dp_err005",
                                                          reconstructed_traj_group_name="reconstructed_traj_005",
                                                          subgoal=1)

    # NOTE: Use this part if you need to drop a column to make everything 2D
    # drop_col_idx = 2
    # waypoint_position, waypoint_velocity = np.delete(waypoint_position, drop_col_idx, axis=1), np.delete(waypoint_velocity, drop_col_idx, axis=1)
    print(f'Data loaded with the shape of {waypoint_position.shape}, goal is {waypoint_position[-1]}')

    # learn a policy
    ds_policy = waypoint_policy(args.learner_type, waypoint_positions=waypoint_position, waypoint_velocities=waypoint_velocity, n_epochs=args.num_epochs)

    # NOTE: plot in 3D only if data dimension is 3
    if waypoint_position.shape[1] == 3:
        # generate a trajectory with the trained model (can be extended to multiple)
        dt: float = 0.01

        start_point = waypoint_position[0].reshape(1, waypoint_position.shape[1])
        goal_point = waypoint_position[-1].reshape(1, waypoint_position.shape[1])

        rollout: List[np.ndarray] = []
        rollout.append(start_point)

        distance_to_target = np.linalg.norm(rollout[-1] - goal_point)
        while  distance_to_target > 0.02  and len(rollout) < 5e3: # rollout termination conditions, hardcoded for now
            vel = ds_policy.predict(rollout[-1])
            rollout.append(rollout[-1] + dt * vel)
            distance_to_target = np.linalg.norm(rollout[-1] - goal_point)

        rollout = np.array(rollout).squeeze()
        print(f'Rollout finished with distance to target: {distance_to_target}')

        # plot and investigate the waypoints: customized for 3D
        x = waypoint_position[:, 0]
        y = waypoint_position[:, 1]
        z = waypoint_position[:, 2]

        x_rollout = rollout[:, 0]
        y_rollout = rollout[:, 1]
        z_rollout = rollout[:, 2]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z, c='b', label='waypoints')
        ax.plot(x_rollout, y_rollout, z_rollout, c='r', label='rollout')

        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('X3')
        ax.legend()
        ax.set_title('DS Policy Waypoints')
        plt.savefig('waypoints-policy.png')
