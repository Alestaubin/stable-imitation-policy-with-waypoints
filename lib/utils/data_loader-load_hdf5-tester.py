#from data_loader import load_hdf5_data
import h5py
import numpy as np


def load_hdf5_data(
        dataset: str = "../data/", 
        demo_id = 0, 
        waypoints_dataset_name = "AWE_waypoints", 
        reconstructed_traj_group_name = "reconstructed_traj",
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
                    reconstructed_traj
                        is_success (bool indicating if the demonstration was successful)
                        subgoals (list of subgoal indices)
                        traj_eef_pos (list of end-effector positions)
                        traj_eef_quat (list of end-effector orientations)
                        traj_eef_vel_ang
                        traj_eef_vel_lin
                        traj_gripper_qpos 
                    ...
                demo_1
                    ...
                ...
        - the reconstructed trajectories are obtained by linearly interpolating the end-effector positions of the waypoints (125 actions between each waypoint)
    Args:
        dataset (str): The path to the hdf5 dataset file.
        demo_id (int): The index of the demonstration to load.
        waypoints_dataset_name (str): The name of the dataset containing the waypoints.
        reconstructed_traj_group_name (str): The name of the group containing the reconstructed trajectory.
        subgoal (int): The index of the subgoal to load the data from.
    Returns:
        np.ndarray: The end-effector positions of the waypoints in the segment.
        np.ndarray: The end-effector velocities of the waypoints in
    """

    f = h5py.File(dataset, 'r')
    
    demo_waypoints = f[f'data/demo_{demo_id}/{waypoints_dataset_name}']

    subgoals = f[f'data/demo_{demo_id}/{reconstructed_traj_group_name}/subgoals']

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
    for waypoint in segment_waypoints:
        vel_data.append(f[f'data/demo_{demo_id}/obs/robot0_eef_vel_lin'][waypoint])
        pos_data.append(f[f'data/demo_{demo_id}/obs/robot0_eef_pos'][waypoint])
        # can get orientation with f[f'data/demo_{demo_id}/obs/robot0_eef_quat'][waypoint] 
        # can get gripper action with f[f'data/demo_{demo_id}/abs_action'][waypoint][-1] (the value will be either 1 for open or -1 for close)
    f.close()
    # convert to numpy arrays
    return np.array(pos_data), np.array(vel_data)



pos, vel = load_hdf5_data(dataset="/Users/alexst-aubin/Library/CloudStorage/GoogleDrive-alex.staubin2@gmail.com/My Drive/MgGill/IL-with-waypoints/data/KITCHEN_SCENE1_put_the_black_bowl_on_the_plate/image_demo_local_with_AWE_waypoints.hdf5",
                          demo_id=1,
                          waypoints_dataset_name="waypoints_AWE_waypoints_dp_err005",
                          reconstructed_traj_group_name="reconstructed_traj_005",
                          subgoal=1)
print(pos)
print(vel)

