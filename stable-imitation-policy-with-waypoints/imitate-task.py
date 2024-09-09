#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, proj3d

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
from lib.utils.waypoint_utils import scatter_waypoints, normalize_waypoints, augment_data, plot_rollouts, clean_waypoints

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
                    augment_alpha: Optional[float] = None,
                    augment_distribution: Optional[str] = 'normal',
                    normalize_magnitude: Optional[float] = None,
                    clean: Optional[bool] = False,
                    fhat_layers: Optional[List[int]] = [64, 64],
                    lpf_layers: Optional[List[int]] = [16, 16],
                    eps: Optional[float] = 0.01,
                    alpha: Optional[float] = 0.01,
                    relaxed: Optional[bool] = False, 
                    angular: Optional[bool] = False):

    """ Train a stable/unstable policy to learn a nonlinear dynamical system. """

    if angular:
        name = f'{model_name}-{learner_type}-subgoal{subgoal}-angular-{time_stamp()}'
    else:
        name = f'{model_name}-{learner_type}-subgoal{subgoal}-{time_stamp()}'
    
    goal = waypoint_positions[-1].reshape(1, waypoint_positions.shape[1])

    # set goal velocity to zero
    waypoint_velocities[-1] = np.zeros(waypoint_velocities[-1].shape)
    
    # Maybe normalize the data
    if normalize_magnitude is not None and not angular:
        waypoint_velocities = normalize_waypoints(waypoint_velocities, normalize_magnitude)
    
    # Plot the waypoints
    scatter_waypoints(waypoint_positions, waypoint_velocities, title=f'Subgoal {subgoal} Waypoints')
    
    # Maybe clean the data 
    if clean and not angular:
        waypoint_positions, waypoint_velocities = clean_waypoints(waypoint_positions, waypoint_velocities)
        scatter_waypoints(waypoint_positions, waypoint_velocities, title=f'Subgoal {subgoal} Cleaned Waypoints')
    
    # Maybe augment the data
    if augment_rate is not None and augment_alpha is not None and not angular:   
        logger.info(f'Augmenting data with rate {augment_rate} and alpha {augment_alpha} according to a {augment_distribution} distribution.')
        waypoint_positions, waypoint_velocities = augment_data(waypoint_positions, waypoint_velocities, augment_alpha, augment_rate, augment_distribution)
        scatter_waypoints(waypoint_positions, waypoint_velocities, title=f'Subgoal {subgoal} Augmented Waypoints')

    if learner_type in ["snds", "nn", "sdsef", "lnet"]: 
        model = NL_DS(network=learner_type, 
                      data_dim=waypoint_positions.shape[1], 
                      goal=goal, 
                      device=device,
                      eps=eps,
                      alpha=alpha,
                      relaxed=relaxed,
                      fhat_layers=fhat_layers,
                      lpf_layers=lpf_layers)
        model.fit(waypoint_positions, waypoint_velocities, n_epochs=n_epochs, lr_initial=1e-4)
    else:
        raise NotImplementedError(f'Learner type {learner_type} not available!')

    if plot:
        plot_ds_2Dstream(model, waypoint_positions, save_dir=save_dir, file_name=f'{name}-ds', show_rollouts=True)

        if model.lpf(waypoint_positions[-1].reshape(1, waypoint_positions.shape[1])) is not None:
            plot_contours(model.lpf, waypoint_positions, save_dir=save_dir, file_name=f'{name}-lpf')

    model.save(model_name=name, dir=save_dir)
    #torch.save(model, os.path.join(save_dir, name+".pt"))
    logger.info(f'Model saved as {name} in {save_dir}.')
    return model

def train_policy_for_subgoal(subgoal_data, config, subgoal_index):
    waypoint_position = subgoal_data["waypoint_position"]
    #waypoint_velocity = subgoal_data["waypoint_linear_velocity"]
    waypoint_velocity = np.zeros(waypoint_position.shape)
    waypoint_ang_velocity = None

    for i in range (len(waypoint_position)):
        if i == len(waypoint_position) - 1:
            waypoint_velocity[i] = np.zeros(waypoint_velocity[i].shape)
            #waypoint_ang_velocity[i] = np.zeros(waypoint_ang_velocity[i].shape)
        else:
            waypoint_velocity[i] = waypoint_position[i+1] - waypoint_position[i] 
            #waypoint_ang_velocity[i] = waypoint_ee_orientation[i+1] - waypoint_ee_orientation[i]
    
    # Train a policy for the linear velocity
    _ = waypoint_policy(
        config['training']['learner_type'],
        waypoint_positions=waypoint_position,
        waypoint_velocities=waypoint_velocity,
        n_epochs=config["training"]['num_epochs'],
        device=config["training"]['device'],
        subgoal=subgoal_index,
        augment_rate=config["data_processing"]['augment_rate'],
        augment_alpha=config["data_processing"]['augment_alpha'],
        augment_distribution=config["data_processing"]['augment_distribution'],
        normalize_magnitude=config["data_processing"]['normalize_magnitude'],
        clean=config["data_processing"]['clean'],
        fhat_layers=config["snds"]['fhat_layers'],
        lpf_layers=config["snds"]['lpf_layers'],
        eps=config["snds"]['eps'],
        alpha=config["snds"]['alpha'],
        relaxed=config["snds"]['relaxed'],
        angular=False
    )
    # train a policy for the angular velocity
    """_ = waypoint_policy(
        config['training']['learner_type'],
        waypoint_positions=waypoint_position,
        waypoint_velocities=waypoint_ang_velocity,
        n_epochs=config["training"]['num_epochs'],
        device=config["training"]['device'],
        subgoal=subgoal_index,
        augment_rate=None,
        augment_alpha=None,
        augment_distribution=None,
        normalize_magnitude=config["data_processing"]['normalize_magnitude'],
        clean=False,
        fhat_layers=config["snds"]['fhat_layers'],
        lpf_layers=config["snds"]['lpf_layers'],
        eps=config["snds"]['eps'],
        alpha=config["snds"]['alpha'],
        relaxed=config["snds"]['relaxed'],
        angular=True
    )"""

    logger.info(f"Subgoal {subgoal_index} training complete.")

def main(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)

    demo = config["training"]['demo']
    data = {}

    # get number of subgoals in the demo
    with h5py.File(config["data"]['data_dir'], 'r') as f: 
        subgoals = f[f"data/demo_{demo}/{config['data']['subgoals_dataset']}"]
        num_subgoals = len(subgoals)

    for i in range(num_subgoals):
        waypoint_position, waypoint_velocity, waypoint_gripper_action, waypoint_ee_euler = load_hdf5_data(
            dataset=config["data"]['data_dir'],
            demo_id=demo,
            waypoints_dataset_name=config["data"]['waypoints_dataset'],
            subgoals_dataset_name=config["data"]['subgoals_dataset'],
            subgoal=i
        )
        data["subgoal_" + str(i)] = {"waypoint_position": waypoint_position, "waypoint_linear_velocity": waypoint_velocity, "waypoint_gripper_action": waypoint_gripper_action, "waypoint_ee_euler": waypoint_ee_euler}

    logger.info(f'Data loaded from {config["data"]["data_dir"]}.')
    if config["data"]['linear_policies'] is None or config["data"]['model_dir'] is None:
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

        # Wait for all processes to finish 
        for p in ps:
            p.join(timeout=120)  # Add a timeout to avoid indefinite hanging
        
        return
    else:
        policies = []
        for i, model_name in enumerate(config["data"]['linear_policies']):
            logger.info(f"Loading model {model_name}")
            waypoint_positions = data["subgoal_"+str(i)]["waypoint_position"]
            model = NL_DS(
                network=config["training"]['learner_type'], 
                data_dim=waypoint_position.shape[1], 
                goal=waypoint_position[-1].reshape(1, waypoint_positions.shape[1]), 
                device=config['training']['device'],
                eps=config["snds"]['eps'], 
                alpha=config["snds"]['alpha'],
                relaxed=config["snds"]['relaxed'],
                fhat_layers=config["snds"]['fhat_layers'],
                lpf_layers=config["snds"]['lpf_layers']
                )
            # Load the model
            model.load(model_name=model_name, dir=config["data"]["model_dir"])
            policies.append(model)
        
        # if using dynamics for angular velocity
        if config["data"]["angular_policies"] is not None:
            angular_policies = []
            for i, model_name in enumerate(config["data"]['angular_policies']):
                logger.info(f"Loading model {model_name}")
                waypoint_positions = data["subgoal_"+str(i)]["waypoint_position"]
                model = NL_DS(
                    network=config["training"]['learner_type'], 
                    data_dim=waypoint_position.shape[1], 
                    goal=waypoint_position[-1].reshape(1, waypoint_positions.shape[1]), 
                    device=config['training']['device'],
                    eps=config["snds"]['eps'], 
                    alpha=config["snds"]['alpha'],
                    relaxed=config["snds"]['relaxed'],
                    fhat_layers=config["snds"]['fhat_layers'],
                    lpf_layers=config["snds"]['lpf_layers']
                    )
                # Load the model
                model.load(model_name=model_name, dir=config["data"]["model_dir"])
                angular_policies.append(model)
        else:
            angular_policies = None
    
    # maybe playback the rollout in the simulation
    if config["simulation"]['playback']:
        logger.info("Starting playback...")
        folder_name = time_stamp()
        #create a folder with the current time and date in the videos directory
        video_path = f"videos/{folder_name}"
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        
        # maybe plot the rollouts 
        if config["simulation"]['plot']:
            plot_rollouts(data, policies, video_path)


        video_full_name = video_path + "/" + config["simulation"]['video_name']
        # save a file info.txt in the same directory as the video
        with open(os.path.join(video_path, 'info.txt'), 'w') as f:
            f.write(f"{config}")

        # get the episode number
        ep = "demo_"+ str(config["training"]['demo'])
        # get the initial state of the environment
        # NOTE: this is important because the object positions are not the same across demos  
        with h5py.File(config["data"]['data_dir'], 'r') as f: 
            states = f["data/{}/states".format(ep)][()]
            initial_state = dict(states=states[0])
            initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]

        playback_dataset(
            dataset_path=config["data"]['data_dir'],
            video_name=video_full_name,
            camera_names=config["simulation"]['camera_names'],
            video_skip=config["simulation"]['video_skip'],
            policies=policies,
            angular_policies=angular_policies,
            subgoals=[{
                    "subgoal_pos": subgoal_data["waypoint_position"][-1],
                    "subgoal_euler": subgoal_data["waypoint_ee_euler"][-1],
                    "subgoal_gripper": subgoal_data["waypoint_gripper_action"][-1]} for subgoal_data in data.values()],
            initial_state=initial_state,
            write_video=config["simulation"]['write_video'],
            rollouts=config["testing"]['num_rollouts'],
            max_horizon=config["testing"]['max_horizon'],
            verbose=config["testing"]['verbose'], 
            slerp_steps=config["simulation"]['slerp_steps']
        )
        logger.info(f"Playback complete. Video saved to {video_full_name}")
        

    logger.info("Process complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Nonlinear DS experiments CLI interface.')
    parser.add_argument('--config', type=str, required=True, help='Path to the JSON config file.')
    args = parser.parse_args()

    main(args.config)