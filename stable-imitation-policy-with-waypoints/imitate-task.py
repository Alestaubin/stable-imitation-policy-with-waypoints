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

import torch.multiprocessing as mp
from torch.multiprocessing import Queue

from lib.learn_nn_ds import NL_DS
from lib.learn_ply_ds import PLY_DS
from lib.utils.utils import time_stamp
from lib.utils.log_config import logger
from lib.utils.plot_tools import plot_ds_2Dstream, plot_trajectory, plot_contours
from lib.utils.data_loader import load_hdf5_data

from sim.playback_robomimic import playback_dataset

import h5py

def waypoint_policy(learner_type: str,
                    waypoint_positions: np.ndarray,
                    waypoint_velocities: np.ndarray,
                    n_epochs: int,
                    plot: Optional[bool] = False,
                    model_name: Optional[str] = 'waypoint-test',
                    save_dir: str = 'res/',
                    device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu',
                    subgoal: Optional[int] = None,
                    augment_rate: Optional[int] = None,
                    augment_std_dev: Optional[float] = None) :

    """ Train a stable/unstable policy to learn a nonlinear dynamical system. """

    name = f'{model_name}-{learner_type}-subgoal{subgoal}-{time_stamp()}'
    goal = waypoint_positions[-1].reshape(1, waypoint_positions.shape[1])
    # set goal velocity to zero
    # waypoint_velocities[-1] = np.zeros(waypoint_velocities[-1].shape)
    print(f'GOAL: {goal}')

    # Maybe augment the data
    if augment_rate is not None and augment_std_dev is not None:   
        logger.info(f'Augmenting data with rate {augment_rate} and std dev {augment_std_dev}.')
        waypoint_positions, waypoint_velocities = augment_data(waypoint_positions, waypoint_velocities, augment_std_dev, augment_rate)
        
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
    print(f'Model saved as {name} in {save_dir}.')
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
            while distance_to_target > 0.01 and len(rollout) < 5e3:
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
    print(f"Subgoal {subgoal_index} waypoint positions: {waypoint_position}")
    print(f"Subgoal {subgoal_index} waypoint velocities: {waypoint_velocity}")
    ds_policy = waypoint_policy(
        config['learner_type'],
        waypoint_positions=waypoint_position,
        waypoint_velocities=waypoint_velocity,
        n_epochs=config['num_epochs'],
        device=config['device'],
        subgoal=subgoal_index,
        augment_rate=config['augment_rate'],
        augment_std_dev=config['augment_std_dev'],
    )
    print(f"Subgoal {subgoal_index} training complete.")

def augment_data(waypoint_positions, waypoint_velocities, augment_std_dev=0.01, augment_rate=5):
    """Augment the data by adding Gaussian noise to the waypoints."""

    new_positions = []
    new_velocities = []

    for i in range(len(waypoint_positions)):
        # for each original point, generate augment_rate new points
        for _ in range(augment_rate):
            noise = np.random.normal(0, augment_std_dev, waypoint_positions[i].shape)
            new_position = waypoint_positions[i] + noise
            new_positions.append(new_position)
            new_velocities.append(waypoint_velocities[i])

    # Convert to numpy arrays
    new_positions = np.array(new_positions)
    new_velocities = np.array(new_velocities)

    # Combine with the original data 
    augmented_positions = np.vstack((waypoint_positions, new_positions))
    augmented_velocities = np.vstack((waypoint_velocities, new_velocities))
    for i in range(len(augmented_positions)):
        print(f"Augmented position {i}: {augmented_positions[i]}, Augmented velocity {i}: {augmented_velocities[i]}")
    return augmented_positions, augmented_velocities


def main(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)

    demos = config['demos']
    data = {}
    demo = demos[0]

    # get number of subgoals in the demo
    with h5py.File(config['data_dir'], 'r') as f: 
        #print(f"data/demo_{demo}/{config['subgoals_dataset']}")
        subgoals = f[f"data/demo_{demo}/{config['subgoals_dataset']}"]
        num_subgoals = len(subgoals)

    for i in range(num_subgoals):
        waypoint_position, waypoint_velocity, waypoint_orientation, waypoint_gripper_action = load_hdf5_data(
            dataset=config['data_dir'],
            demo_id=demo,
            waypoints_dataset_name=config['waypoints_dataset'],
            subgoals_dataset_name=config['subgoals_dataset'],
            subgoal=i
        )
        data["subgoal_" + str(i)] = {"waypoint_position": waypoint_position, "waypoint_velocity": waypoint_velocity, "waypoint_orientation": waypoint_orientation, "waypoint_gripper_action": waypoint_gripper_action}

    print(f'Data loaded from {config["data_dir"]}.')
    if config['model_names'] is None or config['model_dir'] is None:
        # Use multiprocessing to train a policy for each subgoal
        mp.set_start_method('spawn')  # Must be 'spawn' to avoid issues with CUDA

        ps = []

        # Create and start processes
        for i in range(len(data.keys())):
            p = mp.Process(
                target=train_policy_for_subgoal, 
                args=(data["subgoal_" + str(i)], config, i),
                name=f"{i}")
            p.start()
            ps.append(p)

        # Wait for all processes to finish with a timeout
        for p in ps:
            p.join(timeout=120)  # Add a timeout to avoid indefinite hanging
        return
    else:
        policies = []
        for i, model_name in enumerate(config['model_names']):
            print(f"Loading model {model_name}")
            waypoint_positions = data["subgoal_"+str(i)]["waypoint_position"]
            model = NL_DS(network=config['learner_type'], data_dim=waypoint_position.shape[1], goal=waypoint_position[-1].reshape(1, waypoint_positions.shape[1]), device=config['device'],
                eps=0.2, alpha=0.1)
            # Load the model
            model.load(model_name=model_name, dir=config["model_dir"])
            policies.append(model)
    # maybe plot the rollouts NOTE: doesn't work on mac...
    if config['plot']:
        plot_rollouts(data, policies)
    if config['playback']:
        print("Starting playback...")
        playback_dataset(
            dataset_path=config['data_dir'],
            video_path=config['video_path'],
            camera_names=config['camera_names'],
            video_skip=config['video_skip'],
            policies=policies,
            subgoals=[{"subgoal_pos": subgoal_data["waypoint_position"][-1],
                       "subgoal_ori": subgoal_data["waypoint_orientation"][-1],
                       "subgoal_gripper": subgoal_data["waypoint_gripper_action"][-1]} for subgoal_data in data.values()],
            multiplier=config['multiplier'], 
            force_dim=config['force_dim'],
            force_time_step=config['force_time_step']
        )
    print("Process complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Nonlinear DS experiments CLI interface.')
    parser.add_argument('--config', type=str, required=True, help='Path to the JSON config file.')
    args = parser.parse_args()

    main(args.config)