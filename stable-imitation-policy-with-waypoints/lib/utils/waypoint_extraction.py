""" 
Automatic waypoint selection

Example usage:
python waypoint_extraction.py --dataset "/Users/alexst-aubin/SummerResearch24/V2/stable-imitation-policy-with-waypoints/stable-imitation-policy-with-waypoints/data/KITCHEN_SCENE1_put_the_black_bowl_on_the_plate/image_demo_local.hdf5" -n "AWE_waypoints_0025" --end_idx 49 -e 0.0025
"""

import h5py
import argparse
import numpy as np
from tqdm import tqdm
import copy

from scipy.spatial.transform import Rotation

import robosuite
import robosuite.utils.transform_utils as T
from log_config import logger


def linear_interpolation(p1, p2, t):
    """Compute the linear interpolation between two 3D points"""
    return p1 + t * (p2 - p1)


def point_line_distance(point, line_start, line_end):
    """Compute the shortest distance between a 3D point and a line segment defined by two 3D points"""
    line_vector = line_end - line_start
    point_vector = point - line_start
    # t represents the position of the orthogonal projection of the given point onto the infinite line defined by the segment
    t = np.dot(point_vector, line_vector) / np.dot(line_vector, line_vector)
    t = np.clip(t, 0, 1)
    projection = linear_interpolation(line_start, line_end, t)
    return np.linalg.norm(point - projection)


def point_quat_distance(point, quat_start, quat_end, t, total):
    pred_point = T.quat_slerp(quat_start, quat_end, fraction=t / total)
    err_quat = (
        Rotation.from_quat(pred_point) * Rotation.from_quat(point).inv()
    ).magnitude()
    return err_quat


def geometric_waypoint_trajectory(actions, gt_states, waypoints, return_list=False):
    """Compute the geometric trajectory from the waypoints"""

    # prepend 0 to the waypoints for geometric computation
    if waypoints[0] != 0:
        waypoints = [0] + waypoints
    gt_pos = [p["robot0_eef_pos"] for p in gt_states]
    gt_quat = [p["robot0_eef_quat"] for p in gt_states]

    keypoints_pos = [actions[k, :3] for k in waypoints]
    keypoints_quat = [gt_quat[k] for k in waypoints]

    state_err = []

    n_segments = len(waypoints) - 1

    for i in range(n_segments):
        # Get the current keypoint and the next keypoint
        start_keypoint_pos = keypoints_pos[i]
        end_keypoint_pos = keypoints_pos[i + 1]
        start_keypoint_quat = keypoints_quat[i]
        end_keypoint_quat = keypoints_quat[i + 1]

        # Select ground truth points within the current segment
        start_idx = waypoints[i]
        end_idx = waypoints[i + 1]
        segment_points_pos = gt_pos[start_idx:end_idx]
        segment_points_quat = gt_quat[start_idx:end_idx]

        # Compute the shortest distances between ground truth points and the current segment
        for i in range(len(segment_points_pos)):
            pos_err = point_line_distance(
                segment_points_pos[i], start_keypoint_pos, end_keypoint_pos
            )
            rot_err = point_quat_distance(
                segment_points_quat[i],
                start_keypoint_quat,
                end_keypoint_quat,
                i,
                len(segment_points_quat),
            )
            state_err.append(pos_err + rot_err)

    # logger.info the average and max error for pos and rot
    # logger.info(f"Average pos error: {np.mean(pos_err_list):.6f} \t Average rot error: {np.mean(rot_err_list):.6f}")
    # logger.info(f"Max pos error: {np.max(pos_err_list):.6f} \t Max rot error: {np.max(rot_err_list):.6f}")

    if return_list:
        return total_traj_err(state_err), state_err
    return total_traj_err(state_err)


def pos_only_geometric_waypoint_trajectory(
    actions, gt_states, waypoints, return_list=False
):
    """Compute the geometric trajectory from the waypoints"""

    # prepend 0 to the waypoints for geometric computation
    if waypoints[0] != 0:
        waypoints = [0] + waypoints

    keypoints_pos = [actions[k] for k in waypoints]
    state_err = []
    n_segments = len(waypoints) - 1

    for i in range(n_segments):
        # Get the current keypoint and the next keypoint
        start_keypoint_pos = keypoints_pos[i]
        end_keypoint_pos = keypoints_pos[i + 1]

        # Select ground truth points within the current segment
        start_idx = waypoints[i]
        end_idx = waypoints[i + 1]
        segment_points_pos = gt_states[start_idx:end_idx]

        # Compute the shortest distances between ground truth points and the current segment
        for i in range(len(segment_points_pos)):
            pos_err = point_line_distance(
                segment_points_pos[i], start_keypoint_pos, end_keypoint_pos
            )
            state_err.append(pos_err)

    # print the average and max error
    logger.info(
        f"Average pos error: {np.mean(state_err):.6f} \t Max pos error: {np.max(state_err):.6f}"
    )

    if return_list:
        return total_traj_err(state_err), state_err
    else:
        return total_traj_err(state_err)


def total_traj_err(err_list):
    # return np.mean(err_list)
    return np.max(err_list)


""" ################# Iterative waypoint selection #################"""

num_waypoints = []
num_frames = []

""" DP waypoint selection """
# use geometric interpretation
def dp_waypoint_selection(
    actions=None,
    gt_states=None,
    err_threshold=None,
    initial_states=None,
    pos_only=False,
):
    if actions is None:
        actions = copy.deepcopy(gt_states)
    elif gt_states is None:
        gt_states = copy.deepcopy(actions)
        
    num_frames = len(actions)

    # make the last frame a waypoint
    initial_waypoints = [num_frames - 1]

    # make the frames of gripper open/close waypoints
    if not pos_only:
        for i in range(num_frames - 1):
            if actions[i, -1] != actions[i + 1, -1]:
                initial_waypoints.append(i)
                # initial_waypoints.append(i + 1)
        initial_waypoints.sort()

    # Memoization table to store the waypoint sets for subproblems
    memo = {}

    # Initialize the memoization table
    for i in range(num_frames):
        memo[i] = (0, [])

    memo[1] = (1, [1])
    func = (
        pos_only_geometric_waypoint_trajectory
        if pos_only
        else geometric_waypoint_trajectory
    )

    # Check if err_threshold is too small, then return all points as waypoints
    min_error = func(actions, gt_states, list(range(1, num_frames)))
    if err_threshold < min_error:
        logger.info("Error threshold is too small, returning all points as waypoints.")
        return list(range(1, num_frames))

    # Populate the memoization table using an iterative bottom-up approach
    for i in range(1, num_frames):
        min_waypoints_required = float("inf")
        best_waypoints = []

        for k in range(1, i):
            # waypoints are relative to the subsequence
            waypoints = [j - k for j in initial_waypoints if j >= k and j < i] + [i - k]

            total_traj_err = func(
                actions=actions[k : i + 1],
                gt_states=gt_states[k : i + 1],
                waypoints=waypoints,
            )

            if total_traj_err < err_threshold:
                subproblem_waypoints_count, subproblem_waypoints = memo[k - 1]
                total_waypoints_count = 1 + subproblem_waypoints_count

                if total_waypoints_count < min_waypoints_required:
                    min_waypoints_required = total_waypoints_count
                    best_waypoints = subproblem_waypoints + [i]

        memo[i] = (min_waypoints_required, best_waypoints)

    min_waypoints_count, waypoints = memo[num_frames - 1]
    waypoints += initial_waypoints
    # remove duplicates
    waypoints = list(set(waypoints))
    waypoints.sort()
    logger.info(
        f"Minimum number of waypoints: {len(waypoints)} \tTrajectory Error: {total_traj_err}"
    )
    logger.info(f"waypoint positions: {waypoints}")

    return waypoints

def subgoal_selection(dataset_path, start_idx, end_idx, waypoints_dataset_name):
    with h5py.File(dataset_path, "a") as f:
        i = start_idx
        while i <= end_idx:
            logger.info(f"Processing episode {i}")
            if f"data/demo_{i}" in f:
                try:
                    waypoint_indices = f[f"data/demo_{i}/{waypoints_dataset_name}"]
                except:
                    logger.info(f"Waypoints for demo_{i} not found in the dataset.")
                    continue
                waypoint_abs_actions = []

                if "abs_actions" not in f[f"data/demo_{i}"]:
                    raise NotImplementedError("Need to convert actions to absolute actions first.")
                
                for idx in waypoint_indices:
                    waypoint_abs_actions.append(f[f"data/demo_{i}/abs_actions"][idx])
                
                subgoals = [] 
                prev_gripper_action = waypoint_abs_actions[0][-1] 
                for j,action in enumerate(waypoint_abs_actions):
                    if action[-1] != prev_gripper_action:
                        subgoals.append(waypoint_indices[j])
                        prev_gripper_action = action[-1]

                subgoals.append(waypoint_indices[-1]) # last waypoint must be a subgoal 
                logger.info(f"Subgoals for demo_{i}: {subgoals}")
                # save the subgoals
                subgoals_dataset_name = str.replace(waypoints_dataset_name, "waypoints", "subgoals")
                if f"data/demo_{i}/{subgoals_dataset_name}" in f:
                    logger.info(f"Deleting existing dataset: data/demo_{i}/{subgoals_dataset_name}")
                    del f[f"data/demo_{i}/{subgoals_dataset_name}"]
                f.create_dataset(f"data/demo_{i}/{subgoals_dataset_name}", data=subgoals)
            else:
                logger.info(f"demo_{i} not found in the dataset.")
            
            i += 1

def main(args):

    # load the dataset
    f = h5py.File(args.dataset, "r+")
    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    assert args.start_idx >= 0 and args.end_idx < len(demos)
    for idx in tqdm(range(args.start_idx, args.end_idx + 1), desc="Saving waypoints"):
        ep = demos[idx]
        logger.info(f"Processing episode {ep}")

        # prepare initial states to reload from
        states = f[f"data/{ep}/states"][()]
        initial_states = []
        for i in range(len(states)):
            initial_states.append(dict(states=states[i]))
            initial_states[i]["model"] = f[f"data/{ep}"].attrs["model_file"]
        traj_len = states.shape[0]

        # load the ground truth eef pos and rot, joint pos, and gripper qpos
        eef_pos = f[f"data/{ep}/obs/robot0_eef_pos"][()]
        eef_quat = f[f"data/{ep}/obs/robot0_eef_quat"][()]
        joint_pos = f[f"data/{ep}/obs/robot0_joint_pos"][()]
        gt_states = []
        for i in range(traj_len):
            gt_states.append(
                dict(
                    robot0_eef_pos=eef_pos[i],
                    robot0_eef_quat=eef_quat[i],
                    robot0_joint_pos=joint_pos[i],
                )
            )

        # load absolute actions
        try:
            actions = f[f"data/{ep}/abs_actions"][()]
        except:
            logger.info("No absolute actions found, need to convert first.")
            raise NotImplementedError

        waypoint_selection = dp_waypoint_selection

        waypoints = waypoint_selection(
            actions=actions,
            gt_states=gt_states,
            err_threshold=args.err_threshold,
            initial_states=initial_states,
        )

        num_waypoints.append(len(waypoints))
        num_frames.append(traj_len)

        # save waypoints
        if f"data/{ep}/{args.group_name}" in f:
            logger.info(f"Deleting existing dataset: data/{ep}/{args.group_name}")
            del f[f"data/{ep}/{args.group_name}"]
        
        f.create_dataset(f"data/{ep}/{args.group_name}", data=waypoints)

    f.close()
    logger.info(
        f"Average number of waypoints: {np.mean(num_waypoints)}, average number of frames: {np.mean(num_frames)}, average waypoint ratio: {np.mean(num_frames) / np.mean(num_waypoints)}"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="robomimic/datasets/lift/ph/low_dim.hdf5",
        help="path to hdf5 dataset",
    )
    parser.add_argument(
        '-n',
        "--group_name",
        type=str,
        default="data",
        help="name of the group in the hdf5 file to save the waypoints to",
    )

    # index of the trajectory to playback. If omitted, playback trajectory 0.
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="(optional) start index of the trajectory to playback",
    )

    parser.add_argument(
        "--end_idx",
        type=int,
        default=199,
        help="(optional) end index of the trajectory to playback",
    )

    # method (possible values: greedy, dp, backtrack)
    parser.add_argument(
        "--method",
        type=str,
        default="dp",
        help="(optional) method for waypoint selection",
    )

    # error threshold for reconstructing the trajectory
    parser.add_argument(
        "-e",
        "--err_threshold",
        type=float,
        default=0.01,
        help="(optional) error threshold for reconstructing the trajectory",
    )
    parser.add_argument(
        "--subgoal_selection",
        action="store_true",
        help="(optional) whether to select subgoals only",
    )

    args = parser.parse_args()
    if args.subgoal_selection:
        subgoal_selection(args.dataset, args.start_idx, args.end_idx, args.group_name)
    else:
        main(args)
        subgoal_selection(args.dataset, args.start_idx, args.end_idx, args.group_name)
