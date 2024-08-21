#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
from typing import List, Optional, Union
import multiprocessing as mp

from lib.learn_nn_ds import NL_DS
from lib.learn_ply_ds import PLY_DS
from lib.utils.utils import time_stamp
from lib.utils.log_config import logger
from lib.utils.plot_tools import plot_ds_2Dstream, plot_trajectory, plot_contours
from lib.utils.data_loader import load_hdf5_data

import h5py

def waypoint_policy(learner_type: str,
                    waypoint_positions: np.ndarray,
                    waypoint_velocities: np.ndarray,
                    n_epochs: int,
                    plot: Optional[bool] = False,
                    model_name: Optional[str] = 'waypoint-test',
                    save_dir: str = 'res/',
                    device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'):

    """ Train a stable/unstable policy to learn a nonlinear dynamical system. """

    name = f'{model_name}-{learner_type}-{time_stamp()}'
    goal = waypoint_positions[-1].reshape(1, waypoint_positions.shape[1])

    if learner_type in ["snds", "nn", "sdsef", "lnet"]:
        model = NL_DS(network=learner_type, data_dim=waypoint_positions.shape[1], goal=goal, device=device,
                      eps=0.2, alpha=0.1)
        model.fit(waypoint_positions, waypoint_velocities, n_epochs=n_epochs, lr_initial=1e-4)
    else:
        raise NotImplementedError(f'Learner type {learner_type} not available!')

    if plot:
        plot_ds_2Dstream(model, waypoint_positions, save_dir=save_dir, file_name=f'{name}-ds', show_rollouts=True)

        if model.lpf(waypoint_positions[-1].reshape(1, waypoint_positions.shape[1])) is not None:
            plot_contours(model.lpf, waypoint_positions, save_dir=save_dir, file_name=f'{name}-lpf')

    model.save(model_name=name, dir=save_dir)
    return model

def plot_rollouts(data, policies):
    for i, ds_policy in enumerate(policies):
        waypoint_position = data["subgoal_" + str(i)]["waypoint_position"]
        if waypoint_position.shape[1] == 3:
            dt = 0.01
            start_point = waypoint_position[0].reshape(1, waypoint_position.shape[1])
            goal_point = waypoint_position[-1].reshape(1, waypoint_position.shape[1])

            rollout = [start_point]
            distance_to_target = np.linalg.norm(rollout[-1] - goal_point)
            while distance_to_target > 0.02 and len(rollout) < 5e3:
                vel = ds_policy.predict(rollout[-1])

                if not isinstance(dt, np.ndarray):
                    dt = np.array(dt, dtype=np.float32)
                if not isinstance(vel, np.ndarray):
                    vel = np.array(vel, dtype=np.float32)

                rollout.append(rollout[-1] + dt * vel)
                distance_to_target = np.linalg.norm(rollout[-1] - goal_point)

            rollout = np.array(rollout).squeeze()
            print(f'Rollout finished with distance to target: {distance_to_target}')

            x, y, z = waypoint_position[:, 0], waypoint_position[:, 1], waypoint_position[:, 2]
            x_rollout, y_rollout, z_rollout = rollout[:, 0], rollout[:, 1], rollout[:, 2]

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(x, y, z, c='b', label='waypoints')
            ax.plot(x_rollout, y_rollout, z_rollout, c='r', label='rollout')
            ax.set_xlabel('X1')
            ax.set_ylabel('X2')
            ax.set_zlabel('X3')
            ax.legend()
            ax.set_title('DS Policy Waypoints')
            fig.savefig(f'waypoints-policy-subgoal-{i}.png')


def train_policy_for_subgoal(subgoal_data, config, subgoal_index):
    waypoint_position = subgoal_data["waypoint_position"]
    waypoint_velocity = subgoal_data["waypoint_velocity"]
    ds_policy = waypoint_policy(
        config['learner_type'],
        waypoint_positions=waypoint_position,
        waypoint_velocities=waypoint_velocity,
        n_epochs=config['num_epochs'],
        device=config['device'],
    )
    print(f"Subgoal {subgoal_index} training complete.")
    return ds_policy

def main(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)

    demos = config['demos']
    data = {}
    demo = demos[0]

    # get number of subgoals in the demo
    with h5py.File(config['data_dir'], 'r') as f: 
        print(f"data/demo_{demo}/{config['reconstructed_traj_group']}/subgoals")
        subgoals = f[f"data/demo_{demo}/{config['reconstructed_traj_group']}/subgoals"]
        num_subgoals = len(subgoals)

    for i in range(num_subgoals):
        waypoint_position, waypoint_velocity = load_hdf5_data(
            dataset=config['data_dir'],
            demo_id=demo,
            waypoints_dataset_name=config['waypoints_dataset'],
            reconstructed_traj_group_name=config['reconstructed_traj_group'],
            subgoal=i
        )
        data["subgoal_" + str(i)] = {"waypoint_position": waypoint_position, "waypoint_velocity": waypoint_velocity}

    print(f'Data loaded from {config["data_dir"]}.')

    # Use multiprocessing to train a policy for each subgoal
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = [pool.apply_async(train_policy_for_subgoal, (data["subgoal_" + str(i)], config, i)) for i in range(num_subgoals)]
        policies = [p.get() for p in results]
    # maybe plot the rollouts NOTE: doesn't work on mac...
    if config['plot']:
        plot_rollouts(data, policies)
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Nonlinear DS experiments CLI interface.')
    parser.add_argument('--config', type=str, required=True, help='Path to the JSON config file.')
    args = parser.parse_args()

    main(args.config)