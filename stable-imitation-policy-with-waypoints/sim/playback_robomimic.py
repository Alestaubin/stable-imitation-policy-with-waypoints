"""
A collection of functions to playback the simulation with the given policies and waypoints.
"""

import h5py
import imageio
import numpy as np

import robomimic
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robosuite.utils.transform_utils as TransUtils
import math
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from lib.utils.log_config import logger
import cv2
from lib.utils.utils import time_stamp
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


def playback_dataset(
    dataset_path,
    video_name=None,
    camera_names=["agentview"],
    video_skip=5,
    policies=None,
    angular_policies=None,
    subgoals=None,
    initial_state=None,
    write_video=True,
    rollouts=1,
    max_horizon=2500,
    verbose=False,
    slerp_steps=None,
    perturb_step=None, 
    perturb_ee_pos=None,
    noise_alpha=None, 
    reset_on_fail=False,
    video_path=None,
    grasp_tresh=0.008,
    release_tresh=0.02,
):
    """
    Playback a dataset with the given policies and waypoints, while also plotting the 
    distance between the current end-effector (EE) position and the subgoal EE position.
    
    args:
        dataset_path (str): path to the hdf5 dataset
        video_name (str): path to save the video to
        render_image_names (list): list of camera names to render
        video_skip (int): Number of steps to skip between video frames
        policies (list): list of policies to use for playback
        subgoals (list): list of subgoals in the trajectory in the form [{"subgoal_pos":subgoal_pos, "subgoal_ori": subgoal_ori, "subgoal_gripper": subgoal_gripper}, ...]
        multiplier (float): scaling factor for the action space
    """
    #if verbose: 
        #logger.info(f"Subgoals: {subgoals}")
    # some arg checking
    if write_video:
        print("writing video to ", video_name)

    # Create environment
    dummy_spec = dict(
        obs=dict(
                low_dim=["robot0_eef_pos"],
                rgb=[],
            ),
    )
    ObsUtils.initialize_obs_utils_with_obs_specs(dummy_spec)

    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    env_meta["env_kwargs"]["controller_configs"]["interpolation"] = "linear"
    env_meta["env_kwargs"]["controller_configs"]["control_delta"] = True # Whether to control the robot using delta or absolute commands (where absolute commands are taken in the world coordinate frame)

    env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=False, render_offscreen=write_video)
    print("=======================================================================================")    
    print("ENV:",env)

    if not EnvUtils.is_robosuite_env(env_meta): 
        raise ValueError("Playback only supported for robosuite environments.")

    # Initialize video writer
    if write_video:
        video_writer = imageio.get_writer(video_name, fps=20)

    slerp = False
    if slerp_steps is not None:
        logger.info(f"Using slerp with {slerp_steps} steps for orientation control")
        slerp = True

    perturb = False
    if perturb_step is not None and perturb_ee_pos is not None:
        logger.info(f"Perturbing the EE position at step {perturb_step} with {perturb_ee_pos}")
        perturb = True

    if len(policies) == 1:
        # handle the case where there is only one policy
        policies = [policies[0] for _ in range(len(subgoals))]
   
    add_noise = noise_alpha is not None
    if add_noise: 
        proprio_corrupter = create_gaussian_noise_corrupter(mean=0.0, std=noise_alpha)
    subgoal_successes = [0,0,0]
    num_successes = 0
    for j in range(rollouts):
        logger.info(f"Rollout {j}")
        trajectory = []
        # load the initial state
        env.reset()
        obs = env.reset_to(initial_state)
        action_num = 0
        success = False
        env_was_reset = False
        for i in range(len(subgoals)):
            if success : 
                print("Task successful")
            subgoal_ee_euler = subgoals[i]["subgoal_euler"]
            # set the threshold for the distance to the subgoal
            # Insight: if grasping, need higher accuracy than if releasing
            threshold = grasp_tresh if subgoals[i]["subgoal_gripper"] == 1 else release_tresh
            first = True
            subgoal_action_num = 0
            distance = math.dist(subgoals[i]["subgoal_pos"], obs["robot0_eef_pos"])
            while distance > threshold and action_num < max_horizon:
                # add info to trajectory
                abs_pos = np.array(obs["robot0_eef_pos"])
                traj_info = (np.array([action_num, i]))
                true_ee_pos = obs["robot0_eef_pos"]
                if add_noise:
                    obs["robot0_eef_pos"] = proprio_corrupter(obs["robot0_eef_pos"])
                # Maybe perturb the EE position
                if perturb and subgoal_action_num == perturb_step[j] and i < 2:
                    logger.info(f"Perturbing the EE position")
                    pseudo_action_num = 0

                    while math.dist(obs["robot0_eef_pos"], perturb_ee_pos[j]) > 0.1 and pseudo_action_num < 50:
                        action_linear = 2* np.array(perturb_ee_pos[j] - obs["robot0_eef_pos"])
                        action_angular = np.zeros(3)
                        action_gripper = np.array([0])
                        action = np.concatenate((action_linear, action_angular, action_gripper))
                        obs, _, _, _ = env.step(action) 
                        if pseudo_action_num % video_skip == 0 and write_video:
                            video_img = []
                            for cam_name in camera_names:
                                video_img.append(put_text(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name), f"Perturbing to {perturb_ee_pos[j]}", font_size=1, thickness=2, position="bottom"))
                            video_img = np.concatenate(video_img, axis=1)
                            video_writer.append_data(video_img)

                        pseudo_action_num += 1
                
                subgoal_action_num += 1

                # Get the current euler angles of the end-effector  
                sim_quat = env.env.sim.data.get_body_xquat('gripper0_eef')
                sim_quat = TransUtils.convert_quat(sim_quat)
                sim_mat = TransUtils.quat2mat(sim_quat)
                sim_euler = TransUtils.mat2euler(sim_mat)
                if first:
                    start_quat = sim_quat
                    start_euler = sim_euler
                    first = False
                    key_rots = R.from_euler('xyz', [start_euler, subgoal_ee_euler], degrees=False)
                    key_times = [0, 1]
                    slerp = Slerp(key_times, key_rots)

                current_ee_pos = obs["robot0_eef_pos"]
                # reshape
                current_ee_pos = np.array(current_ee_pos).reshape(1,3)

                # Get the linear action
                action_linear = np.array(policies[i].predict(current_ee_pos))[0]

                if angular_policies is None:
                    if slerp: # use slerp for orientation control
                        fraction = subgoal_action_num/slerp_steps
                        if fraction >= 1:
                            fraction = 1
                        # We have two rotations, P and Q, and we want to find the rotation R such that applying P and then R is equivalent to applying Q. 
                        # In rotation matrices, this is easy: Q = R*P, so R = Q*P^-1.
                        next_R = slerp(fraction)
                        # get the error rotation between the current rotation and the next rotation  
                        err_R = next_R * R.from_euler('xyz', sim_euler, degrees=False).inv()
                        action_angular = err_R.as_euler('xyz', degrees=False)
                        next_euler = np.array(next_R.as_euler('xyz', degrees=False))
                    else:
                        action_angular = subgoal_ee_euler - sim_euler
                    # normalize if norm is too big
                    if np.linalg.norm(action_angular) > 0.25:
                        action_angular = action_angular / np.linalg.norm(action_angular) * 0.25
                else: 
                    action_angular = angular_policies[i].predict(current_ee_pos)[0] # NOTE: let's see if this works. It does not :(
                
                # add entire action to trajectory
                traj_abs_action = np.concatenate((traj_info, abs_pos, next_euler)).tolist()
                traj_abs_action[0] = int(traj_abs_action[0])
                traj_abs_action[1] = int(traj_abs_action[1])
                trajectory.append(traj_abs_action)
                if i == 0 :
                    action_gripper = np.array ([-1])  # Open the gripper for the first subgoal
                else:
                    action_gripper = np.array([0])

                action = np.concatenate((action_linear, action_angular, action_gripper))
                
                action = np.array(action, copy=True)
                
                # Take the action in the environment
                obs, _, done, _ = env.step(action)

                # Save the frames (agentview)
                if action_num % video_skip == 0 and write_video:
                    video_img = []
                    for k, cam_name in enumerate(camera_names):
                        if k == 0: video_img.append(put_text(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name), f"Subgoal {i}, Rollout {j}", font_size=0.75, thickness=1, position="bottom"))
                        if k == 1: video_img.append(put_text(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name), f"Success: {success}", font_size=0.75, thickness=1, position="bottom"))

                    video_img = np.concatenate(video_img, axis=1)  # Concatenate horizontally
                    video_img = put_text(video_img, f"ee_euer: {sim_euler}, ee_pos: {obs['robot0_eef_pos']}", font_size=0.25, thickness=1, position="top")
                    video_writer.append_data(video_img)
            
                action_num += 1
        
                # check if task is successful
                success = env.is_success()["task"]
                # print feedback
                if action_num % 25 == 0 and verbose:
                    logger.info(f"subgoal_euler: {subgoal_ee_euler}, sim_euler: {sim_euler}")
                    #print("subgoal_quat: ", subgoal_quat, "sim_quat: ", sim_quat)
                    logger.info(f"Distance to subgoal {i}: {distance}, Action number: {action_num}, Current euler: {sim_euler}, Subgoal euler: {subgoal_ee_euler}")
                    logger.info(f"Action: {action}")
                    logger.info(f"Success: {success}")
                
                #update distance
                distance = math.dist(subgoals[i]["subgoal_pos"], true_ee_pos)


            if distance > threshold and action_num >= max_horizon and reset_on_fail:
                # Subgoal failed
                env_was_reset = True
                logger.info(f"subgoal failed")
                action_num = 0
                env.reset_to(initial_state)                
                desired_joint_positions =  subgoals[i]["joint_pos"]
                env.env.sim.data.qpos[env.env.robots[0].joint_indexes] = desired_joint_positions
                video_img = []
                for cam_name in camera_names:
                    video_img.append(put_text(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name), f"Reset env", font_size=1, thickness=2, position="bottom"))
                video_img = np.concatenate(video_img, axis=1)  # Concatenate horizontally
                video_writer.append_data(video_img)
            else:
                # Subgoal succeeded
                subgoal_successes[i] += 1

            if verbose: 
                logger.info(f"Subgoal {i} distance: {distance}")

            # Activate the gripper
            action_gripper = subgoals[i]["subgoal_gripper"]
            action_gripper_string = "Open" if action_gripper == -1 else "Close"
            action = np.zeros_like(action)
            action[-1] = action_gripper
            if verbose:
                logger.info("\n####################################################\n################ Activating gripper ################\n####################################################")
            gripper_action_num = 0
            while True:
                obs, _, _, _ = env.step(action)
                if action_num % video_skip == 0 and write_video:
                    video_img = []
                    for cam_name in camera_names:
                        video_img.append(put_text(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name), f"Gripper Action: {action_gripper_string}", font_size=1, thickness=1, position="top"))
                    video_img = np.concatenate(video_img, axis=1)  # Concatenate horizontally
                    video_writer.append_data(video_img)
                
                action_num += 1
                gripper_action_num += 1
                # if the gripper vel is lower than 0.0001, it is most likely not moving 
                # and the object has been grasped
                if abs(obs["robot0_gripper_qvel"][0]) <= 0.001 and gripper_action_num > 25: 
                    break
            if verbose:
                logger.info("Gripper action done.")
        
        # at this point, all segments have been executed
        # this means: 
        # 1. distance is the distance between the last subgoal and the current end-effector position
        # 2. success is whether the environment considers the task successful
        # 3. env_was_reset is whether the environment was reset at any point during the execution of the subgoals (i.e. one of the subgoals failed)
        # 4. num_successes is the number of successful tasks so far
        success = env.is_success()["task"]
        print(success, not env_was_reset, distance < threshold)
        if success and not env_was_reset and distance < threshold:
            logger.info(f"Task successful")
            num_successes += 1
        if not success or distance > threshold or env_was_reset:
            logger.info(f"Task failed")
        logger.info(f"Subgoal successes: {subgoal_successes}")
        logger.info(f"Number of successful tasks: {num_successes}/{rollouts}")
        print("=======================================================================================")

    # write the trajectory to a csv file 
    with open(f"{video_path}/trajectory_{time_stamp()}.csv", 'w') as f:
        f.write ("action_num, segment_num, x_pos, y_pos, z_pos, x_euler, y_euler, z_euler\n")
        for i, action in enumerate(trajectory):
            f.write(f"{action[0]}, {action[1]}, {action[2]}, {action[3]}, {action[4]}, {action[5]}, {action[6]}, {action[7]}\n")
    # Close video writer
    if write_video:
        video_writer.close()
    

def put_text(img, text, font_size=1, thickness=2, position="top"):
    img = img.copy()
    if position == "top":
        p = (10, 30)
    elif position == "bottom":
        p = (10, img.shape[0] - 60)
    # put the frame number in the top left corner
    img = cv2.putText(
        img,
        str(text),
        p,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_size,
        (0, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return img

def create_gaussian_noise_corrupter(mean, std, low=-np.inf, high=np.inf):
    """
    Creates a corrupter that applies gaussian noise to a given input with mean @mean and std dev @std

    Args:
        mean (float): Mean of the noise to apply
        std (float): Standard deviation of the noise to apply
        low (float): Minimum value for output for clipping
        high (float): Maxmimum value for output for clipping

    Returns:
        function: corrupter
    """

    def corrupter(inp):
        inp = np.array(inp)
        noise = mean + std * np.random.randn(*inp.shape)
        return np.clip(inp + noise, low, high)

    return corrupter

