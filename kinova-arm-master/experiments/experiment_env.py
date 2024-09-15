#! /usr/bin/env python3

import math
import os, sys
from typing import List, Tuple
import numpy as np
import threading as th
import pandas as pd
import pickle

from enum import Enum
from kortex_api_wrap import *
save_data: bool = False


class ControlModes(Enum):
    JOINT_POSITION = 0
    JOINT_VELOCITY = 1
    END_EFFECTOR_POSE = 2
    END_EFFECTOR_TWIST = 3


class RealWorldTrajectory:
    """ A sample real-world trajectory for experiments.
    """

    def __init__(self, trajectory_name: str = "default"):
        # TODO: Could potentially write a parser for coordinates.
        self.start = {'linear_x': 0.206,
                      'linear_y': -0.414,
                      'linear_z': 0.117,
                      'angular_x': -2.338,
                      'angular_y': 177.674,
                      'angular_z': 177.56}

        self.middle = {'linear_x': 0.206,
                       'linear_y': -0.207,
                       'linear_z': 0.215,
                       'angular_x': 177.643,
                       'angular_y': 2.36,
                       'angular_z': 144.643}

        self.goal = {'linear_x': 0.306,
                     'linear_y': 0.247,
                     'linear_z': 0.415,
                     'angular_x': 177.643,
                     'angular_y': 2.36,
                     'angular_z': 144.643}

        self.pos_dataset = pd.DataFrame(columns=['linear_x', 'linear_y', 'linear_z', 'angular_x', 'angular_y', 'angular_z'])
        self.vel_dataset = pd.DataFrame(columns=['linear_x', 'linear_y', 'linear_z', 'angular_x', 'angular_y', 'angular_z'])

        self.capture: bool = False
        self.trajectory_name: str = trajectory_name

    def capture_data(self, basecyclic: BaseCyclicClient, delta_t: float = 0.1):
        """Log feedback data for a base cyclic client.

        Args:
            basecyclic (BaseCyclicClient): base cyclic object is used for getting feedback from the arm.
        """
        # capturing data is not necessary, unless called upon
        while not self.capture:
            time.sleep(0.01)
            continue

        n_samples = 0

        while self.capture:
            vel_dict = endeffector_twist_feedback(basecyclic)
            pos_dict = endeffector_pose_feedback(basecyclic)

            self.pos_dataset = pd.concat([self.pos_dataset, pd.DataFrame(pos_dict, index=[n_samples])])
            self.vel_dataset = pd.concat([self.vel_dataset, pd.DataFrame(vel_dict, index=[n_samples])])

            time.sleep(delta_t)
            n_samples += 1

        print(f'Terminating demonstration data logger with {n_samples} samples')

    def clear_data(self):
        """ Clear datasets to recapture another sequence.
        """

        self.pos_dataset = pd.DataFrame(columns=['linear_x', 'linear_y', 'linear_z', 'angular_x', 'angular_y', 'angular_z'])
        self.vel_dataset = pd.DataFrame(columns=['linear_x', 'linear_y', 'linear_z', 'angular_x', 'angular_y', 'angular_z'])

    def save_data(self, dir: str = os.getcwd()):
        """Save the captured data from demonstrations.

        Args:
            dir (str, optional): Path to dems folder contatining the pos and vels. Defaults to os.getcwd().
        """

        save_dir = os.path.join(dir, 'dems', self.trajectory_name)
        os.makedirs(save_dir, exist_ok=True)
        self.pos_dataset.to_csv(path_or_buf=os.path.join(save_dir, 'pos.csv'))
        self.vel_dataset.to_csv(path_or_buf=os.path.join(save_dir, 'vel.csv'))

    def load_data(self, dir: str = os.getcwd()):
        """Load a previously stored trajectory.

        Args:
            dir (str, optional): Directory of dems folder contatining pos and velocity data. Defaults to os.getcwd().
        """

        load_dir = os.path.join(dir, 'dems', self.trajectory_name)
        self.pos_dataset = pd.read_csv(os.path.join(load_dir, 'pos.csv'), index_col=0)
        self.vel_dataset = pd.read_csv(os.path.join(load_dir, 'vel.csv'), index_col=0)


class KinovaDSExperiments:
    def __init__(self, mode=ControlModes.END_EFFECTOR_POSE, device_ip: str = '192.168.1.10',
                 device_port: int = 10000, username: str = 'admin', password: str = 'admin',
                 session_inactivity_timeout: int = 60000, connection_inactivity_timeout: int = 20000, capture_mode: bool = save_data, home: bool = True):

        # build the transport layer
        self.__transport = TCPTransport() if device_port == DeviceConnection.TCP_PORT else UDPTransport()
        self.__router = RouterClient(self.__transport, RouterClient.basicErrorCallback)
        self.__transport.connect(device_ip, device_port)
        self.__trajectory_handle = RealWorldTrajectory()
        self.__capture_mode = capture_mode


        # ds planning policy
        self.__ds_planner = None

        # create a session
        session_info = Session_pb2.CreateSessionInfo()
        session_info.username = username
        session_info.password = password
        session_info.session_inactivity_timeout = session_inactivity_timeout
        session_info.connection_inactivity_timeout = connection_inactivity_timeout

        self.__session_manager = SessionManager(self.__router)
        print(f'Logging as {session_info.username} on device {device_ip}')
        self.__session_manager.CreateSession(session_info)

        # create base and basecyclic objects
        self.__base = BaseClient(self.__router)
        self.__basecyclic = BaseCyclicClient(self.__router)
        print(f'Initialization complete')

        # start the logger
        if self.__capture_mode:
            self.__data_capture_p = th.Thread(target=self.__trajectory_handle.capture_data, args=(self.__basecyclic,))
            self.__data_capture_p.start()

        # home the robot arm
        if home:
            print(f'Moving to home position')
            self.home()

        self.__control_mode = mode

    def move(self, data: Dict, feedback: bool = False):
        """ Move to a certain cartesian/angles pos or with a specific twist based
        on the control_mode.

        Args:
            data (Dict): Set of joint/cartesian pos or twists.
            feedback (bool, optional): Upon activation, feedback loggers in other threads
                start recording. Defaults to False.
        """
        if feedback and self.__capture_mode:
            self.__trajectory_handle.capture = True

        if self.__control_mode == ControlModes.END_EFFECTOR_POSE:
            endeffector_pose_command(self.__base, endeffector_pose_dict=data)

        elif self.__control_mode == ControlModes.END_EFFECTOR_TWIST:
            endeffector_twist_command(self.__base, duration=data["dt"],
                                      endeffector_twists_dict=data)

        elif self.__control_mode == ControlModes.JOINT_POSITION:
            joints_position_command(self.__base, joint_positions_dict=data)

        elif self.__control_mode == ControlModes.JOINT_VELOCITY:
            joints_velocity_command(self.__base, duration=data["dt"],
                                    joint_velocity_dict=data)

    def grip(self, press: float = 0.7):
        """ Close the gripper.
        """
        change_gripper(self.__base, press)

    def reboot(self):
        """ Reboot the arm.
        """
        self.__base.Reboot()

    def release_joints(self, secs = 10):
        """Releases the joints so the robot can be subject to perturbation
        without going into fault mode.

        Args:
            secs (int, optional): Number of seconds to let go. Defaults to 10.
        """
        self.pause(8)
        self.__base.ClearFaults()
        self.pause(2)

    def pause(self, secs=2):
        """ Pause for a determined time. Only a wrapper for sleep at this point.

        Args:
            secs (int, optional): Seconds to pause. Defaults to 2.
        """
        time.sleep(secs)

    def home(self):
        """ Move to a predefined home position.
        """
        execute_defined_action(self.__base, "Home")

    def retract(self):
        """ Move to a predefined home position.
        """
        execute_defined_action(self.__base, "Retract")

    def zero(self):
        """ Move to a predefined home position.
        """
        execute_defined_action(self.__base, "Zero")

    def get_endeffector_feedback(self):
        """Return a dict with joints pose and twists feedback.

        Returns:
            Dict: full feedback
        """
        return {"pose": endeffector_pose_feedback(self.__basecyclic), "twist": endeffector_twist_feedback(self.__basecyclic)}

    def get_joints_feedback(self):
        """Return a dict with joints pose and twists feedback.

        Returns:
            Dict: full feedback
        """
        return {"position": joints_position_feedback(self.__basecyclic), "velocity": joints_velocity_feedback(self.__basecyclic)}

    def inverse_kinematics(self, pose):
        """Calculate the inverse kinematics.

        Args:
            pose (dict): End-effector Position

        Returns:
            dict: Joint positions optimized by inverse kinematics.
        """

        init_angles_guess = self.get_joints_feedback()["position"]
        joint_angles = inverse_kinematics(self.__base, pose, init_angles_guess)
        return {joint_id: joint_angles.joint_angles[joint_id].value for joint_id in range(6)}

    def execute_trajectory(self, trajectory, is_joint_space: bool = False):
        """Execute a trajectory.

        Args:
            trajectory (List): A list of all the waypoints in the trajectory.
            is_joint_space (bool, optional): False means it's a cartesian trajectory. Defaults to False.
        """

        if is_joint_space:
            execute_jointspace_trajectory(self.__base, trajectory)
        else:
            execute_taskspace_trajectory(self.__base, trajectory)

    def get_trajectory(self):
        """Get the trajectory handler.

        Returns:
            RealWorldTrajectory: the handler
        """

        return self.__trajectory_handle

    def set_control_mode(self, mode: ControlModes):
        """ Set the arm's control mode.

        Args:
            mode (ControlModes): _description_
        """

        assert mode in ControlModes, "Invalid control mode!"
        self.__control_mode = mode

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.__capture_mode:
            self.__trajectory_handle.capture = False
            self.__data_capture_p.join()
            self.__trajectory_handle.save_data()

        self._terminate_connection()

    def _terminate_connection(self):
        # terminate everything
        self.__base.Stop()

        router_options = RouterClientSendOptions()
        router_options.timeout_ms = 1000

        self.__session_manager.CloseSession(router_options)
        self.__transport.disconnect()


