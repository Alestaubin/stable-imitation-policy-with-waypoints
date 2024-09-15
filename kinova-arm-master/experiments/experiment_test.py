#! /usr/bin/env python3

import argparse
import sys
import math

from typing import Dict, List, Tuple

from experiment_env import KinovaDSExperiments
from experiment_env import ControlModes


def reproduce_trajectory():
    with KinovaDSExperiments() as kde:
        traj = kde.get_trajectory()
        traj.load_data()

        dt = 0.1
        kde.set_control_mode(ControlModes.END_EFFECTOR_TWIST)

        for index, row in traj.vel_dataset.iterrows():
            twists_dict = dict(row)
            twists_dict["dt"] = dt
            kde.move(twists_dict, feedback=False)
            print(f'Sending {index} twist command')
        pass


def baseline_sine_motion(perturbed: bool = False):
    start_point = {'linear_x': 0.117,
                    'linear_y': 0.00,
                    'linear_z': 0.255,
                    'angular_x': 1.393,
                    'angular_y': 178.674,
                    'angular_z': 96.029}

    with KinovaDSExperiments(home=False) as kde:
        kde.set_control_mode(ControlModes.JOINT_POSITION)
        joint_position_start = kde.inverse_kinematics(start_point)
        print(f'Moving to ({start_point["linear_x"]:.3f}, {start_point["linear_y"]:.3f}, {start_point["linear_z"]:.3f})')

        kde.move(joint_position_start)
        kde.grip(press=1.0)
        input(f'Place the marker and hit enter to continue')
        kde.pause(5)

        pose_fb = kde.get_endeffector_feedback()["pose"]
        xb, yb, zb = pose_fb['linear_x'], pose_fb['linear_y'], pose_fb['linear_z']
        tx, ty, tz = pose_fb['angular_x'], pose_fb['angular_y'], pose_fb['angular_z']

        # generate a trajectory
        task_space_trajectory: List[Tuple] = []
        x = xb
        while x < 0.430:
            # calculate new end-effector positions
            x_n = x + 0.005
            y_n = yb + 0.15 * math.sin(50 * (x_n - xb))

            # move to calculated position
            print(f'Adding ({x_n:.3f}, {y_n:.3f}, {zb:.3f})')
            target = {'linear_x': x_n, 'linear_y': y_n, 'linear_z': zb, 'angular_x': tx, 'angular_y': ty, 'angular_z': tz}

            if perturbed:
                if x > 0.258:
                    task_space_trajectory.append(target)
            else:
                task_space_trajectory.append(target)

            # feedback simulation
            x = x_n

        if perturbed:
            kde.execute_trajectory(task_space_trajectory[:2])
            kde.pause(20)
            kde.execute_trajectory(task_space_trajectory[2:])
        else:
            kde.execute_trajectory(task_space_trajectory)


def baseline_w_motion(perturbed: bool = False):
    start_point = {'linear_x': 0.100,
                    'linear_y': 0.00,
                    'linear_z': 0.255,
                    'angular_x': 1.393,
                    'angular_y': 178.674,
                    'angular_z': 96.029}

    with KinovaDSExperiments(home=False) as kde:
        kde.set_control_mode(ControlModes.JOINT_POSITION)

        joint_position_start = kde.inverse_kinematics(start_point)
        print(f'Moving to ({start_point["linear_x"]:.3f}, {start_point["linear_y"]:.3f}, {start_point["linear_z"]:.3f})')

        kde.move(joint_position_start)
        kde.grip(press=0.95)
        input(f'Place the marker and hit enter to continue')
        kde.pause(5)

        pose_fb = kde.get_endeffector_feedback()["pose"]
        xb, yb, zb = pose_fb['linear_x'], pose_fb['linear_y'], pose_fb['linear_z']
        tx, ty, tz = pose_fb['angular_x'], pose_fb['angular_y'], pose_fb['angular_z']

        # generate a trajectory
        task_space_trajectory: List[Tuple] = []
        x = xb
        for _ in range(56):
            # calculate new end-effector positions
            x_n = x + 0.005
            y_n = yb + 0.1 * math.sqrt(abs(3 - ((20*(x_n - 0.24)) ** 2)))

            print(f'Adding ({x_n:.3f}, {y_n:.3f}, {zb:.3f})')
            target = {'linear_x': x_n, 'linear_y': y_n, 'linear_z': zb, 'angular_x': tx, 'angular_y': ty, 'angular_z': tz}

            if perturbed:
                if x > 0.280:
                    task_space_trajectory.append(target)
            else:
                task_space_trajectory.append(target)

            # feedback simulation
            x = x_n

        if perturbed:
            kde.execute_trajectory(task_space_trajectory[:2])
            kde.pause(20)
            kde.execute_trajectory(task_space_trajectory[2:])
        else:
            kde.execute_trajectory(task_space_trajectory)


def baseline_pick_and_place(perturbed: bool = False):
    with KinovaDSExperiments() as kde:
        traj = kde.get_trajectory()
        kde.set_control_mode(ControlModes.END_EFFECTOR_POSE)

        # release the gripper and move to the starting point
        kde.grip(press=0.3)
        kde.move(traj.start)
        kde.pause()

        # grip and move to the goal
        kde.grip()
        if perturbed:
            kde.move(traj.middle)
            kde.release_joints()

        # go to goal
        kde.move(traj.goal)
        # release
        kde.grip(press=0.3)


def main():
    """ Main entry point and argument parser for the exp file.
    """

    parser = argparse.ArgumentParser(description='Handle basic experiments for learning DS on Kinove Gen3 Lite 6-DOF arm.')
    parser.add_argument('-r', '--reboot', action='store_true', default=False,
                        help='Reboot the base before experiments.')
    parser.add_argument('-ho', '--home', action='store_true', default=False,
                        help='Send the robot home before experiments.')
    parser.add_argument('-pap', '--pick-and-place', action='store_true', default=False,
                        help='Simple pick and place demo')
    parser.add_argument('-sp', '--simulate-perturbation', action='store_true', default=False,
                        help='Simulate perturbation for Kinova Arm.')
    parser.add_argument('-hw', '--hand-writing', action='store_true', default=False,
                        help='Use handwriting data for imitation.')
    parser.add_argument('-ms', '--motion-shape', type=str, default='Sine',
                        help='Shape of the trajectory (valid when -hw is enabled).')
    parser.add_argument('-sd', '--save-data', action='store_true', default=False,
                        help='Save the data in the dems folder.')

    args = parser.parse_args()


    global save_data
    save_data = args.save_data

    if args.reboot:
        with KinovaDSExperiments(home=False) as kde:
            print(f'Rebooting the arm in progress')
            kde.reboot()
            kde.pause(secs=10)

    if args.home:
        with KinovaDSExperiments() as kde:
            return

    elif args.pick_and_place:
        baseline_pick_and_place(perturbed=args.simulate_perturbation)

    elif args.hand_writing:
        if args.motion_shape == "Prolonged_Sine":
            baseline_sine_motion(perturbed=args.simulate_perturbation)
        elif args.motion_shape == "Root_Parabola":
            baseline_w_motion(perturbed=args.simulate_perturbation)

    else:
        print(f'Pass an argument or -h for help.')

if __name__ == '__main__':
    main()