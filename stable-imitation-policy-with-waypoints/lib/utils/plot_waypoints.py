"""
A script to plot the waypoints of the demos in a dataset.

Example usage:
python plot_waypoints.py --dataset "/Users/alexst-aubin/SummerResearch24/V2/stable-imitation-policy-with-waypoints/stable-imitation-policy-with-waypoints/data/KITCHEN_SCENE1_put_the_black_bowl_on_the_plate/image_demo_local.hdf5" --idx_end 49
"""

import h5py 
import numpy as np
from waypoint_utils import scatter_waypoints
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/KITCHEN_SCENE1_put_the_black_bowl_on_the_plate/image_demo_local_with_AWE_waypoints.hdf5",
        help="path to hdf5 dataset",
    )
    parser.add_argument(
        "--waypoints_name",
        type=str,
        default="waypoints_dp",
        help="group name of waypoints",
    )
    parser.add_argument(
        "--idx_start",
        type=int,
        default=0,
        help="start index of demos",
    )
    parser.add_argument(
        "--idx_end",
        type=int,
        default=49,
        help="end index of demos",
    )

    args = parser.parse_args()
    with h5py.File(args.dataset, 'a') as f: 
        for i in range(args.idx_start, args.idx_end):  
            waypoints = f[f"data/demo_{i}/{args.waypoints_name}"]

            velocities = f[f"data/demo_{i}/obs/robot0_eef_vel_lin"]
            positions = f[f"data/demo_{i}/obs/robot0_eef_pos"]

            number_of_waypoints = len(waypoints)
            waypoint_positions = np.zeros((number_of_waypoints, 3))
            waypoint_velocities = np.zeros((number_of_waypoints, 3))
                        
            for j,waypoint in enumerate(waypoints):
                
                waypoint_positions[j] = positions[waypoint]
                waypoint_velocities[j] = velocities[waypoint]
            
            print(f"Processing demo {i} waypoints")
            scatter_waypoints(waypoint_positions, waypoint_velocities, title=f"Demo {i} waypoints")

            