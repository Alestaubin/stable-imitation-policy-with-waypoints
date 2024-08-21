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




def get_next_ee_pos (policy, current_ee_pos, multiplier = 1):
    #convert to np array
    current_ee_pos = np.array(current_ee_pos)
    # reshape
    current_ee_pos = current_ee_pos.reshape(1,3)
    vel = np.array(policy.predict(current_ee_pos))
    # uncomment the following if using absolute actions controller instead of delta actions controller
    # next_ee_pos = current_ee_pos + vel * multiplier
    # print ("Next ee pos: ", next_ee_pos)
    # return next_ee_pos
    return vel


def playback_dataset(
    dataset_path,
    video_path=None,
    camera_names=["agentview"],
    video_skip=5,
    policies=None,
    subgoals=None,
    multiplier=1
):
    """
    Playback a dataset with the given policies and waypoints, while also plotting the 
    distance between the current end-effector (EE) position and the subgoal EE position.
    
    args:
        dataset_path (str): path to the hdf5 dataset
        video_path (str): path to save the video to
        render_image_names (list): list of camera names to render
        video_skip (int): Number of steps to skip between video frames
        policies (list): list of policies to use for playback
        subgoals (list): list of subgoals in the trajectory in the form [{"subgoal_pos":subgoal_pos, "subgoal_ori": subgoal_ori, "subgoal_gripper": subgoal_gripper}, ...]
        multiplier (float): scaling factor for the action space
    """
    # some arg checking
    write_video = (video_path is not None)
    print("writing video to ", video_path)
    
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
    env_meta["env_kwargs"]["controller_configs"]["multiplier"] = 10 # This value scales the input commands before they are applied by the controller.

    env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=False, render_offscreen=write_video)

    print("ENV:",env)
    print("SUBGOALS:",subgoals)

    if not EnvUtils.is_robosuite_env(env_meta): 
        raise ValueError("Playback only supported for robosuite environments.")

    with h5py.File(dataset_path, "r") as f:
        pass

    # Initialize video writer
    video_writer = imageio.get_writer(video_path, fps=20)

    # Get initial state of environment
    obs = env.reset()
    print("Initial state of environment: ", obs)
    initial_ee_quat = obs["robot0_eef_quat"]
    mat = TransUtils.quat2mat(initial_ee_quat)
    euler = TransUtils.mat2euler(mat)
    print("Initial ee ori: ", euler)
    
    #initial_ee_ori = euler
    action_num = 0
    distances = []

    for i in range(len(subgoals)):
        while math.dist(subgoals[i]["subgoal_pos"], obs["robot0_eef_pos"]) > 0.008 and action_num < 5000:
            current_pos = obs["robot0_eef_pos"]
            subgoal_pos = subgoals[i]["subgoal_pos"]
            distance = math.dist(subgoal_pos, current_pos)

            
            # Get the next ee_pos
            next_ee_pos = np.array(get_next_ee_pos(policies[i], current_pos, multiplier)[0])
            #action_ori = initial_ee_ori
            action_ori = np.array([0, 0, 0])  # NOTE: don't change the orientation for now
            action_gripper = np.array([0])  # NOTE: gripper action is 0 for now
            action = np.concatenate((next_ee_pos, action_ori, action_gripper))
            action = np.array(action, copy=True)
            if action_num % 25 == 0:
                print(f"Subgoal {i} pos: {subgoal_pos}, Current ee pos: {current_pos}, Distance: {distance}, Action number: {action_num}")
                print("Action: ", action)
            
            # Take the action in the environment
            obs, _, _, _ = env.step(action)

            # Save the frames (agentview)
            if action_num % video_skip == 0:
                video_img = []
                for cam_name in camera_names:
                    video_img.append(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
                video_img = np.concatenate(video_img, axis=1)  # Concatenate horizontally
                video_writer.append_data(video_img)
            
            action_num += 1
        # Activate the gripper
        # NOTE : This is a temporary solution. There should be a check that the object has been grasped 
        action_gripper = subgoals[i]["subgoal_gripper"]
        action[-1] = action_gripper
        # do while loop
        while True:
            print("Activating gripper: ", action)
            print("Gripper qpos: ", obs["robot0_gripper_qpos"])
            print("Gripper qvel: ", obs["robot0_gripper_qvel"])
            obs, _, _, _ = env.step(action)
            video_img = []
            for cam_name in camera_names:
                video_img.append(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
            video_img = np.concatenate(video_img, axis=1)  # Concatenate horizontally
            video_writer.append_data(video_img)
            action_num += 1

            # if the gripper vel is lower than 0.0001, it is most likely not moving 
            # and the object has been grasped
            if abs(obs["robot0_gripper_qvel"][0]) <= 0.0001: 
                break

    # Clean and convert distances to a numpy array
    distances = np.array([d for d in distances if isinstance(d, (float, int))], dtype=float)

    # Close video writer
    video_writer.close()
if __name__ == "__main__":
    pass