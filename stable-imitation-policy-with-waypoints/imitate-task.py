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
                    plot: Optional[bool] = True,
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
                    angular: Optional[bool] = False, 
                    save_model: Optional[bool] = True,
                    show_stats: Optional[bool] = True):

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
        model.fit(waypoint_positions, waypoint_velocities,show_stats=show_stats, n_epochs=n_epochs, lr_initial=1e-4)
    else:
        raise NotImplementedError(f'Learner type {learner_type} not available!')

    if plot:
        plot_ds_2Dstream(model, waypoint_positions, save_dir=save_dir, file_name=f'{name}-ds', show_rollouts=True)

        if model.lpf(waypoint_positions[-1].reshape(1, waypoint_positions.shape[1])) is not None:
            plot_contours(model.lpf, waypoint_positions, save_dir=save_dir, file_name=f'{name}-lpf')
    # maybe save the model
    if save_model:
        model.save(model_name=name, dir=save_dir)
        #torch.save(model, os.path.join(save_dir, name+".pt"))
        logger.info(f'Model saved as {name} in {save_dir}.')
    return model

def train_policy_for_subgoal(subgoal_data, config, subgoal_index):
    waypoint_position = subgoal_data["waypoint_position"]
    waypoint_velocity = subgoal_data["waypoint_linear_velocity"]
    
    # Train a policy for the linear velocity
    model= waypoint_policy(
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
        angular=False,
        save_model=config["training"]['save_model'], 
        show_stats=config["testing"]['verbose']
    )
    logger.info(f"Subgoal {subgoal_index} training complete.")
    return model

def main(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)

    demo = config["training"]['demo']
    data = {}

    # get number of subgoals in the demo
    with h5py.File(config["data"]['data_dir'], 'r') as f: 
        subgoals = f[f"data/demo_{demo}/{config['data']['subgoals_dataset']}"]
        num_subgoals = len(subgoals)
    if config["data"]["waypoints_dataset"] is None:
        logger.info(f'No AWE waypoints provided, using entire trajectory to train the model.')
        # if no waypoints, train the policy on the entire trajectory 
        with h5py.File(config["data"]['data_dir'], 'r') as f: 
            abs_actions = f[f"data/demo_{demo}/abs_actions"]
            ee_pos = abs_actions[:, :3]
            ee_vel = f[f"data/demo_{demo}/obs/robot0_eef_vel_lin"][()]
            ee_euler = abs_actions[:, 3:6]
            waypoint_gripper_actions = abs_actions[:, -1]
            subgoal_indices = f[f"data/demo_{demo}/{config['data']['subgoals_dataset']}"][()]

        for i in range(num_subgoals):
            if i == 0:
                start_idx = 0
            else:
                start_idx = subgoal_indices[i-1]
            end_idx = subgoal_indices[i]+1
            data["subgoal_" + str(i)] = {"waypoint_position": ee_pos[start_idx: end_idx], "waypoint_linear_velocity": ee_vel[start_idx: end_idx], "waypoint_gripper_action": waypoint_gripper_actions[start_idx: end_idx], "waypoint_ee_euler": ee_euler[start_idx: end_idx]}
    else:
        for i in range(num_subgoals):

            waypoint_position, _, waypoint_gripper_action, waypoint_ee_euler = load_hdf5_data(
                dataset=config["data"]['data_dir'],
                demo_id=demo,
                waypoints_dataset_name=config["data"]['waypoints_dataset'],
                subgoals_dataset_name=config["data"]['subgoals_dataset'],
                subgoal=i
            )

            waypoint_velocity = np.zeros(waypoint_position.shape)
            # set the velocity to be the difference between the waypoints
            for j in range (len(waypoint_position)):
                if j == len(waypoint_position) - 1:
                    waypoint_velocity[j] = np.zeros(waypoint_velocity[j].shape)
                else:
                    waypoint_velocity[j] = waypoint_position[j+1] - waypoint_position[j] 
            
            data["subgoal_" + str(i)] = {"waypoint_position": waypoint_position, "waypoint_linear_velocity": waypoint_velocity, "waypoint_gripper_action": waypoint_gripper_action, "waypoint_ee_euler": waypoint_ee_euler}

    logger.info(f'Data loaded from {config["data"]["data_dir"]}.')

    # get the subgoal info
    subgoal_info = [{"subgoal_pos": subgoal_data["waypoint_position"][-1],
                "subgoal_euler": subgoal_data["waypoint_ee_euler"][-1],
                "subgoal_gripper": subgoal_data["waypoint_gripper_action"][-1]} for subgoal_data in data.values()]
    
    with h5py.File(config["data"]['data_dir'], 'r') as f: 
        joint_pos = np.array(f[f"data/demo_{demo}/obs/robot0_joint_pos"])
        subgoals = f[f"data/demo_{demo}/{config['data']['subgoals_dataset']}"]
        for i in range(num_subgoals):
            subgoal_info[i]["index"] = subgoals[i]
            subgoal_info[i]["joint_pos"] = joint_pos[subgoals[i]]

    policies = None

    if config["data"]['linear_policies'] is None or config["data"]['model_dir'] is None:
        if not config["training"]["segmentation"]:
            logger.info(f'Training a single policy for the entire trajectory (no segmentation).')
            # merge all the subgoals into one
            waypoint_positions = np.concatenate([data["subgoal_" + str(i)]["waypoint_position"] for i in range(num_subgoals)], axis=0)
            waypoint_velocities = np.concatenate([data["subgoal_" + str(i)]["waypoint_linear_velocity"] for i in range(num_subgoals)], axis=0)
            waypoint_gripper_actions = np.concatenate([data["subgoal_" + str(i)]["waypoint_gripper_action"] for i in range(num_subgoals)], axis=0)
            waypoint_ee_eulers = np.concatenate([data["subgoal_" + str(i)]["waypoint_ee_euler"] for i in range(num_subgoals)], axis=0)
            data = {"subgoal_0":{"waypoint_position": waypoint_positions, "waypoint_linear_velocity": waypoint_velocities, "waypoint_gripper_action": waypoint_gripper_actions, "waypoint_ee_euler": waypoint_ee_eulers}}
            
        # Use multiprocessing to train a policy for each subgoal
        """
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
        
        return"""
        policies = []
        for i in range(len(data.keys())):
            model = train_policy_for_subgoal(data["subgoal_" + str(i)], config, i)
            policies.append(model)
        # Maybe test the model
        if not config["training"]["test"]: return
    if policies is None:
        policies = []
        for i, model_name in enumerate(config["data"]['linear_policies']):
            logger.info(f"Loading model {model_name}")
            waypoint_positions = data["subgoal_"+str(i)]["waypoint_position"]
            model = NL_DS(
                network=config["training"]['learner_type'], 
                data_dim=waypoint_positions.shape[1], 
                goal=waypoint_positions[-1].reshape(1, waypoint_positions.shape[1]), 
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
    config_file_name = args.config.split("/")[-1].split(".")[0]
    folder_name = time_stamp() + "-" + config_file_name 
    #create a folder with the current time and date in the videos directory
    video_path = f"videos/{folder_name}"
    while os.path.exists(video_path):
        folder_name += f"-1"
        video_path = f"videos/{folder_name}"
    
    if not os.path.exists(video_path):
        os.makedirs(video_path)

    # maybe plot the rollouts 
    if config["simulation"]['plot']:
        logger.info("Plotting rollouts...")
        perturbation = config["simulation"]['perturb_step'] is not None and config["simulation"]['perturb_ee_pos'] is not None
        perturbation_steps = config["simulation"]['perturb_step']
        perturbation_vecs = config["simulation"]['perturb_ee_pos']
        perturb = perturbation_steps is not None and perturbation_vecs is not None

        for j in range(config["testing"]['num_rollouts']):
            if perturb:
                perturbation_step = perturbation_steps[j]
                perturbation_vec = perturbation_vecs[j]
            else:
                perturbation_step = None
                perturbation_vec = None    
            
            logger.info(f"Plotting rollout {j}")
            #print("data", data)
            plot_rollouts(data, subgoal_info, policies, video_path, title=f'{config["training"]["learner_type"]} rollout{j}', perturbation=perturbation, perturbation_step=perturbation_step, perturbation_vec=perturbation_vec, reset_after_subgoal=config["simulation"]['reset_on_fail'])

    
    # maybe playback the rollout in the simulation
    if config["simulation"]['playback']:
        logger.info("Starting playback...")

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
            subgoals=subgoal_info,
            initial_state=initial_state,
            write_video=config["simulation"]['write_video'],
            rollouts=config["testing"]['num_rollouts'],
            max_horizon=config["testing"]['max_horizon'],
            verbose=config["testing"]['verbose'], 
            slerp_steps=config["simulation"]['slerp_steps'],
            perturb_step=config["simulation"]['perturb_step'],
            perturb_ee_pos=config["simulation"]['perturb_ee_pos'],
            noise_alpha=config["simulation"]['noise_alpha'],
            reset_on_fail=config["simulation"]['reset_on_fail'],
            video_path=video_path, 
            grasp_tresh=config["simulation"]['grasp_tresh'],
            release_tresh=config["simulation"]['release_tresh']
        )
        logger.info(f"Playback complete. Video saved to {video_full_name}")
        

    logger.info("Process complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Nonlinear DS experiments CLI interface.')
    parser.add_argument('--config', type=str, required=True, help='Path to the JSON config file.')
    args = parser.parse_args()
    main(args.config)