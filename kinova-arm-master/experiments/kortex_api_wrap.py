#! /usr/bin/env python3

import time
import argparse
import threading

from functools import partial
from typing import Dict, List
from google.protobuf import json_format

from kortex_api.TCPTransport import TCPTransport
from kortex_api.UDPTransport import UDPTransport
from kortex_api.RouterClient import RouterClient
from kortex_api.RouterClient import RouterClientSendOptions

from kortex_api.SessionManager import SessionManager
from kortex_api.autogen.messages import DeviceConfig_pb2, Session_pb2, Base_pb2, Common_pb2

from kortex_api.autogen.client_stubs.DeviceConfigClientRpc import DeviceConfigClient
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.client_stubs.ControlConfigClientRpc import ControlConfigClient, ControlConfigPb


from kortex_api.Exceptions.KServerException import KServerException
from kortex_api.Exceptions.KException import KException


#########################################
############ Gripper Movement ###########
#########################################
def change_gripper(base, position: float = 0.1):
    # create the GripperCommand we will send
    gripper_command = Base_pb2.GripperCommand()
    finger = gripper_command.gripper.finger.add()

    # close the gripper with position increments
    gripper_command.mode = Base_pb2.GRIPPER_POSITION
    finger.finger_identifier = 1
    finger.value = position
    base.SendGripperCommand(gripper_command)
    time.sleep(2)


#########################################
############ Basic Movement #############
#########################################
def execute_defined_action(base, action_name: str = "Home"):
    # activate single level servoing mode
    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)

    # move arm to ready position
    print("Moving the arm to a safe position")
    action_type = Base_pb2.RequestedActionType()
    action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
    action_list = base.ReadAllActions(action_type)
    action_handle = None
    for action in action_list.action_list:
        if action.name == action_name:
            action_handle = action.handle

    if action_handle == None:
        print("Can't reach safe position. Exiting")

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        partial(check, e=e),
        Base_pb2.NotificationOptions()
    )

    base.ExecuteActionFromReference(action_handle)

    # leave time to action to complete
    finished = e.wait(15000)
    base.Unsubscribe(notification_handle)

    return finished


def endeffector_pose_feedback(cyclic_base) -> Dict[str, float]:
    feedback = cyclic_base.RefreshFeedback()

    feedback_dict: Dict[str, float] = {}
    feedback_dict["linear_x"] = feedback.base.tool_pose_x
    feedback_dict["linear_y"] = feedback.base.tool_pose_y
    feedback_dict["linear_z"] = feedback.base.tool_pose_z
    feedback_dict["angular_x"] = feedback.base.tool_pose_theta_x
    feedback_dict["angular_y"] = feedback.base.tool_pose_theta_y
    feedback_dict["angular_z"] = feedback.base.tool_pose_theta_z

    return feedback_dict


def endeffector_twist_feedback(cyclic_base) -> Dict[str, float]:
    feedback = cyclic_base.RefreshFeedback()

    feedback_dict: Dict[str, float] = {}
    feedback_dict["linear_x"] = feedback.base.tool_twist_linear_x
    feedback_dict["linear_y"] = feedback.base.tool_twist_linear_y
    feedback_dict["linear_z"] = feedback.base.tool_twist_linear_z
    feedback_dict["angular_x"] = feedback.base.tool_twist_angular_x
    feedback_dict["angular_y"] = feedback.base.tool_twist_angular_y
    feedback_dict["angular_z"] = feedback.base.tool_twist_angular_z

    return feedback_dict


def joints_position_feedback(cyclic_base) -> Dict[str, float]:
    feedback = cyclic_base.RefreshFeedback()

    feedback_dict: Dict[int, float] = {}
    feedback_dict = {joint_id: feedback.actuators[joint_id].position
        for joint_id in range(6)}

    return feedback_dict


def joints_velocity_feedback(cyclic_base) -> Dict[str, float]:
    feedback = cyclic_base.RefreshFeedback()

    feedback_dict: Dict[int, float] = {}
    feedback_dict = {joint_id: feedback.actuators[joint_id].velocity
        for joint_id in range(6)}

    return feedback_dict


def joints_torque_feedback(cyclic_base) -> Dict[str, float]:
    feedback = cyclic_base.RefreshFeedback()

    feedback_dict: Dict[int, float] = {}
    feedback_dict = {joint_id: feedback.actuators[joint_id].torque
        for joint_id in range(6)}

    return feedback_dict


def endeffector_pose_command(base, endeffector_pose_dict: Dict[str, float]):
    action = Base_pb2.Action()
    action.name = "End effector pose command"
    action.application_data = ""

    cartesian_pose = action.reach_pose.target_pose
    cartesian_pose.x = endeffector_pose_dict["linear_x"]
    cartesian_pose.y = endeffector_pose_dict["linear_y"]
    cartesian_pose.z = endeffector_pose_dict["linear_z"]
    cartesian_pose.theta_x = endeffector_pose_dict["angular_x"]
    cartesian_pose.theta_y = endeffector_pose_dict["angular_y"]
    cartesian_pose.theta_z = endeffector_pose_dict["angular_z"]

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        partial(check, e=e),
        Base_pb2.NotificationOptions()
    )

    base.ExecuteAction(action)

    finished = e.wait(15000)
    base.Unsubscribe(notification_handle)
    return finished


def joints_position_command(base, joint_positions_dict: Dict[int, float]):
    action = Base_pb2.Action()
    action.name = "Joints position movement"
    action.application_data = ""

    # Place arm straight up
    for joint_id in joint_positions_dict:
        joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
        joint_angle.joint_identifier = joint_id
        joint_angle.value = joint_positions_dict[joint_id]

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        partial(check, e=e),
        Base_pb2.NotificationOptions()
    )

    base.ExecuteAction(action)
    finished = e.wait(15000)

    base.Unsubscribe(notification_handle)
    return finished


def joints_velocity_command(base, duration: float = 1, joint_velocity_dict: Dict[int, float] = None):
    # create an action
    action = Base_pb2.Action()
    action.name = "Move with joint velocity"
    action.application_data = ""
    joint_speeds = Base_pb2.JointSpeeds()

    for joint_id in joint_velocity_dict:
        joint_speed = joint_speeds.joint_speeds.add()
        joint_speed.joint_identifier = joint_id
        joint_speed.value = joint_velocity_dict[joint_id]
        joint_speed.duration = 0

    base.SendJointSpeedsCommand(joint_speeds)
    time.sleep(duration)

    base.Stop()
    return True


def endeffector_twist_command(base, duration: float = 1, endeffector_twists_dict: Dict[str, float] = None):
    command = Base_pb2.TwistCommand()

    command.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_TOOL
    command.duration = 0

    twist = command.twist
    twist.linear_x = endeffector_twists_dict["linear_x"]
    twist.linear_y = endeffector_twists_dict["linear_y"]
    twist.linear_z = endeffector_twists_dict["linear_z"]
    twist.angular_x = endeffector_twists_dict["angular_x"]
    twist.angular_y = endeffector_twists_dict["angular_y"]
    twist.angular_z = endeffector_twists_dict["angular_z"]

    finished = base.SendTwistCommand(command)

    time.sleep(duration)
    return finished


#########################################
############## Trajectory ###############
#########################################
def populate_endeffector_pose(pose):

    waypoint = Base_pb2.CartesianWaypoint()
    waypoint.pose.x = pose["linear_x"]
    waypoint.pose.y = pose["linear_y"]
    waypoint.pose.z = pose["linear_z"]
    waypoint.blending_radius = 0.005
    waypoint.pose.theta_x = pose["angular_x"]
    waypoint.pose.theta_y = pose["angular_y"]
    waypoint.pose.theta_z = pose["angular_z"]
    waypoint.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE

    return waypoint

def execute_taskspace_trajectory(base, task_space_trajectory):
    waypoints = Base_pb2.WaypointList()
    waypoints.duration = 0.0
    waypoints.use_optimal_blending = True

    for idx, pose in enumerate(task_space_trajectory):
        waypoint = waypoints.waypoints.add()
        waypoint.name = "waypoint_" + str(idx)
        waypoint_cartesian = populate_endeffector_pose(pose)
        if idx == len(task_space_trajectory) - 1:
            waypoint_cartesian.blending_radius = 0.0
        waypoint.cartesian_waypoint.CopyFrom(waypoint_cartesian)

    result = base.ValidateWaypointList(waypoints)
    if (len(result.trajectory_error_report.trajectory_error_elements) == 0):
        print(f'Executing trajectory of {len(task_space_trajectory)} points')
        e = threading.Event()
        notification_handle = base.OnNotificationActionTopic(partial(check, e=e),
                                                            Base_pb2.NotificationOptions())

        base.ExecuteWaypointTrajectory(waypoints)
        finished = e.wait(30000)
        base.Unsubscribe(notification_handle)
        return finished

    else:
        print('Error found in trajectory')
        print(result.trajectory_error_report)
        return False

def populate_angular_position(joint_position, duration):
    waypoint = Base_pb2.AngularWaypoint()
    waypoint.angles.extend(joint_position)
    waypoint.duration = duration * 5.0
    return waypoint


def execute_jointspace_trajectory(base, joint_space_trajectory):
    waypoints = Base_pb2.WaypointList()
    waypoints.duration = 0.0
    waypoints.use_optimal_blending = False

    for idx, joint_position in enumerate(joint_space_trajectory):
        waypoint = waypoints.waypoints.add()
        waypoint.name = "waypoint_" + str(idx)
        durationFactor = 1
        waypoint.angular_waypoint.CopyFrom(populate_angular_position(joint_position, durationFactor))

    result = base.ValidateWaypointList(waypoints)
    if len(result.trajectory_error_report.trajectory_error_elements) == 0:
        print(f'Executing trajectory of {len(joint_space_trajectory)} points')
        e = threading.Event()
        notification_handle = base.OnNotificationActionTopic(
            partial(check, e=e),
            Base_pb2.NotificationOptions()
        )
        base.ExecuteWaypointTrajectory(waypoints)

        finished = e.wait(30000)
        base.Unsubscribe(notification_handle)
        return finished
    else:
        print("Error found in trajectory")
        print(result.trajectory_error_report)
        return finished


#########################################
############ Router Client ##############
#########################################
def rpc_call(base: BaseClient):
    ''' The RouterClientSendOptions exist to modify the default behavior
        of the router. The router default value are
            andForget = False     (not implemented yet)
            delay_ms = 0          (not implemented yet)
            timeout_ms = 10000

        The same function call without the options=router_options is valid and will do the same
        using router's default options.
    '''

    # set router options if needed (many not implemented yet and possibly not necessary)
    router_options = RouterClientSendOptions()
    router_options.timeout_ms = 5000 # 5 seconds

    try:
        requested_action_type = Base_pb2.RequestedActionType()
        print(f'Requested actions type is: {requested_action_type}')
        all_actions = base.ReadAllActions(requested_action_type, options=router_options)
    except Exception as e:
        handle_exception_msg(e)
    else:
        print ("List of all actions in the arm:")
        for action in all_actions.action_list:
            print("============================================")
            print("Action name: {0}".format(action.name))
            print("Action identifier: {0}".format(action.handle.identifier))
            print("Action type: {0}".format(Base_pb2.ActionType.Name(action.handle.action_type)))
            print("Action permissions: ")
            if (action.handle.permission & Common_pb2.NO_PERMISSION): print("\t- {0}".format(Common_pb2.Permission.Name(Common_pb2.NO_PERMISSION)))
            if (action.handle.permission & Common_pb2.READ_PERMISSION): print("\t- {0}".format(Common_pb2.Permission.Name(Common_pb2.READ_PERMISSION)))
            if (action.handle.permission & Common_pb2.UPDATE_PERMISSION): print("\t- {0}".format(Common_pb2.Permission.Name(Common_pb2.UPDATE_PERMISSION)))
            if (action.handle.permission & Common_pb2.DELETE_PERMISSION): print("\t- {0}".format(Common_pb2.Permission.Name(Common_pb2.DELETE_PERMISSION)))
            print("============================================")


#########################################
############## Kinematics ###############
#########################################
def forward_kinematics(base: BaseClient, input_joint_angles):
    """ Compute forward kinematics.

    Find the forward kinematics using the current joint angles. It is of course possible
    to pass on other joint configurations with some modifications.

    Args:
        base (BaseClient): base client object for the reference of calculations.

    Returns:
        bool: Whether the calculations were successful.
    """

    # forward kinematics
    pose = base.ComputeForwardKinematics(input_joint_angles)
    return pose


def inverse_kinematics(base, pose, init_angles_guess={0:0, 1:0, 2:0, 3:0, 4:0, 5:0}):
    """
    Find the inverse kinematics for a given pose. Quite the same as forward kinematics,
    but in the other direction.

    Args:
        base (BaseClient): base client object for the reference of calculations.

    Returns:
        bool: Whether the calculations were successful.
    """

    # object containing cartesian coordinates and angle guess
    input_IkData = Base_pb2.IKData()

    # fill the IKData object with the cartesian coordinates that need to be converted
    input_IkData.cartesian_pose.x = pose["linear_x"]
    input_IkData.cartesian_pose.y = pose["linear_y"]
    input_IkData.cartesian_pose.z = pose["linear_z"]
    input_IkData.cartesian_pose.theta_x = pose["angular_x"]
    input_IkData.cartesian_pose.theta_y = pose["angular_y"]
    input_IkData.cartesian_pose.theta_z = pose["angular_z"]

    # Fill the IKData Object with the guessed joint angles
    for joint_id in init_angles_guess:
        jAngle = input_IkData.guess.joint_angles.add()
        jAngle.value = init_angles_guess[joint_id]

    computed_joint_angles = base.ComputeInverseKinematics(input_IkData)
    return computed_joint_angles


#########################################
####### Notifications and Profiles ######
#########################################
def notification_subscriber(base):
    # you need to determine a callback function
    def notification_callback(data):
        print(f'Notification callback activated.')
        print(json_format.MessageToJson(data))

    # subscribe to ConfigurationChange notifications
    print("Subscribing to ConfigurationChange notifications")
    notif_handle = base.OnNotificationConfigurationChangeTopic(notification_callback, Base_pb2.NotificationOptions())

    # miscellaneous tasks
    time.sleep(3)

    # Create a user profile to trigger a notification
    full_user_profile = Base_pb2.FullUserProfile()
    full_user_profile.user_profile.username = 'cj'
    full_user_profile.user_profile.firstname = 'Carl'
    full_user_profile.user_profile.lastname = 'Johnson'
    full_user_profile.user_profile.application_data = "data"
    full_user_profile.password = "pwd"

    user_profile_handle = Base_pb2.UserProfileHandle()
    try:
        print("Creating user profile to trigger notification")
        user_profile_handle = base.CreateUserProfile(full_user_profile)
    except KException as ex:
        print("Failed to create user profile: ", ex)

    # another task for instance
    time.sleep(3)

    print("Unsubscribing from ConfigurationChange notifications")
    base.Unsubscribe(notif_handle)

    try:
        base.DeleteUserProfile(user_profile_handle)
    except KException:
        print("User profile deletion failed")

    # wait a bit longer then finish
    time.sleep(3)


#########################################
############## Utilities ################
#########################################
def handle_exception_msg(ex):
    print("Unable to get current robot pose")
    print("Error_code:{} , Sub_error_code:{} ".format(ex.get_error_code(), ex.get_error_sub_code()))
    print("Caught expected error: {}".format(ex))

def check(notification, e):
    if notification.action_event == Base_pb2.ACTION_END:
        e.set()
    if notification.action_event == Base_pb2.ACTION_ABORT:
        e.set()

def parse_connection_args(parser = argparse.ArgumentParser()):
    parser.add_argument("-ip", "--device-ip", type=str, help="IP address of destination", default="192.168.1.10")
    parser.add_argument("-u", "--username", type=str, help="username to login", default="admin")
    parser.add_argument("-p", "--password", type=str, help="password to login", default="admin")
    parser.add_argument("-at", "--action-type", type=str, help="Action type for the robot", default="Home")
    return parser.parse_args()


class DeviceConnection:
    """ Creating TCP or UDP connections automatically.

    Returns:
        TCPTransport or UDPTransport: Transport object for communicating with the robot.
    """

    TCP_PORT = 10000
    UDP_PORT = 10001

    def createTcpConnection(args):
        """
        returns RouterClient required to create services and send requests to device or sub-devices,
        """

        return DeviceConnection(args.device_ip, port=DeviceConnection.TCP_PORT, credentials=(args.username, args.password))

    def createUdpConnection(args):
        """
        returns RouterClient that allows to create services and send requests to a device or its sub-devices @ 1khz.
        """

        return DeviceConnection(args.device_ip, port=DeviceConnection.UDP_PORT, credentials=(args.username, args.password))

    def __init__(self, ipAddress, port=TCP_PORT, credentials = ("","")):
        """ Initialize a connection.

        Args:
            ip_address (str): IP address of the robot.
            port (_type_, optional): _description_. Defaults to TCP_PORT.
            credentials (tuple, optional): Credentials. Defaults to ("","").
        """
        self.ip = ipAddress
        self.port = port
        self.credentials = credentials

        self.sessionManager = None
        self.transport = TCPTransport() if port == DeviceConnection.TCP_PORT else UDPTransport()
        self.router = RouterClient(self.transport, RouterClient.basicErrorCallback)

    def __enter__(self):
        self.transport.connect(self.ip, self.port)

        if (self.credentials[0] != ""):
            session_info = Session_pb2.CreateSessionInfo()
            session_info.username = self.credentials[0]
            session_info.password = self.credentials[1]
            session_info.session_inactivity_timeout = 10000
            session_info.connection_inactivity_timeout = 2000

            self.sessionManager = SessionManager(self.router)
            print(f'Logging as {self.credentials[0]} on device {self.ip}')
            self.sessionManager.CreateSession(session_info)

        return self.router

    def __exit__(self, exc_type, exc_value, traceback):

        if self.sessionManager != None:
            router_options = RouterClientSendOptions()
            router_options.timeout_ms = 1000

            self.sessionManager.CloseSession(router_options)
        self.transport.disconnect()


def create_angular_action(actuator_count):

    print("Creating angular action")
    action = Base_pb2.Action()
    action.name = "Example angular action"
    action.application_data = ""

    for joint_id in range(actuator_count):
        joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
        joint_angle.value = 0.0

    return action


def create_cartesian_action(base_cyclic):

    print("Creating Cartesian action")
    action = Base_pb2.Action()
    action.name = "Example Cartesian action"
    action.application_data = ""

    feedback = base_cyclic.RefreshFeedback()

    cartesian_pose = action.reach_pose.target_pose
    cartesian_pose.x = feedback.base.tool_pose_x          # (meters)
    cartesian_pose.y = feedback.base.tool_pose_y - 0.1    # (meters)
    cartesian_pose.z = feedback.base.tool_pose_z - 0.2    # (meters)
    cartesian_pose.theta_x = feedback.base.tool_pose_theta_x # (degrees)
    cartesian_pose.theta_y = feedback.base.tool_pose_theta_y # (degrees)
    cartesian_pose.theta_z = feedback.base.tool_pose_theta_z # (degrees)

    return action


#########################################
############# Main Entery ###############
#########################################
def main():
    # parse input args for TCP connection
    args = parse_connection_args()

    # create connection to the device and get the router
    with DeviceConnection.createTcpConnection(args) as router:

        # create base client
        base = BaseClient(router)
        base_cyclic = BaseCyclicClient(router)

        if args.action_type == "Home":
            execute_defined_action(base)

        if args.action_type == "Endeffector_Twist":
            endeffector_twist_command(base)

        if args.action_type == "Inverse_Kinematics":
            pose = {'linear_x': 0.23,
                    'linear_y': 0.33,
                    'linear_z': 0.34,
                    'angular_x': 90.65,
                    'angular_y': -0.99,
                    'angular_z': 100.96}

            print(inverse_kinematics(base, pose))

        if args.action_type == "Waypoint_Trajectory":
            time.sleep(2)
            traj: List = []

            p2 = {'linear_x': 0.329,
                    'linear_y': 0.250,
                    'linear_z': 0.255,
                    'angular_x': 1.3,
                    'angular_y': 177.99,
                    'angular_z': 96.96}

            goal = {'linear_x': 0.432,
                    'linear_y': -0.007,
                    'linear_z': 0.255,
                    'angular_x': 1.3,
                    'angular_y': 177.99,
                    'angular_z': 96.96}

            middle = {'linear_x': 0.422,
                    'linear_y': 0.139,
                    'linear_z': 0.255,
                    'angular_x': 1.3,
                    'angular_y': 177.99,
                    'angular_z': 96.96}

            traj.append(p2)
            traj.append(middle)
            traj.append(goal)
            execute_taskspace_trajectory(base, traj)

        if args.action_type == "Endeffector_Position":
            time.sleep(2)
            goal = {'linear_x': 0.432,
                    'linear_y': -0.006,
                    'linear_z': 0.255,
                    'angular_x': 1.3,
                    'angular_y': 177.99,
                    'angular_z': 96.96}
            p1 = {'linear_x': 0.236,
                    'linear_y': -0.271,
                    'linear_z': 0.255,
                    'angular_x': 1.3,
                    'angular_y': 177.99,
                    'angular_z': 96.96}
            p2 = {'linear_x': 0.329,
                    'linear_y': 0.250,
                    'linear_z': 0.255,
                    'angular_x': 1.3,
                    'angular_y': 177.99,
                    'angular_z': 96.96}
            p3 = {'linear_x': 0.431,
                'linear_y': -0.083,
                'linear_z': 0.255,
                'angular_x': 1.3,
                'angular_y': 177.99,
                'angular_z': 96.96}
            pose = p2
            endeffector_pose_command(base, pose)

        if args.action_type == "Joint_Position":
            pos = {0:0.0,
                   1:0.0,
                   2:0.0,
                   3:0.0,
                   4:0.0,
                   5:0.0}
            joints_position_command(base, pos)

        if args.action_type == "Joint_Velocity":
            vels = {0:20,
                    1:0.0,
                    2:0.0,
                    3:0.0,
                    4:0.0,
                    5:0.0}
            joints_velocity_command(base, 1, vels)

        if args.action_type == "Feedback":
            print(endeffector_twist_feedback(base_cyclic))
            print(endeffector_pose_feedback(base_cyclic))

        if args.action_type == "Gripper":
            change_gripper(base, 0.5)

        if args.action_type == "Release_Joints":
            # Not available for Gen3Lite :(
            admittance = Base_pb2.Admittance()
            admittance.admittance_mode = Base_pb2.DISABLED
            base.SetAdmittance(admittance)

if __name__ == '__main__':
    main()