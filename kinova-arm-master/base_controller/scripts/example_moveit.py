#!/usr/bin/env python

import sys
import rospy
import moveit_commander
import moveit_msgs.msg


class MoveItHandle(object):
  def __init__(self):

    # Initialize the node
    super(MoveItHandle, self).__init__()
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('example_move_it')

    self.is_gripper_present = rospy.get_param(rospy.get_namespace() + "is_gripper_present", False)
    if self.is_gripper_present:
      gripper_joint_names = rospy.get_param(rospy.get_namespace() + "gripper_joint_names", [])
      self.gripper_joint_name = gripper_joint_names[0]

    # Create the necessary objects
    self.robot = moveit_commander.RobotCommander("robot_description")
    self.scene = moveit_commander.PlanningSceneInterface(ns=rospy.get_namespace())
    self.arm_group = moveit_commander.MoveGroupCommander("arm", ns=rospy.get_namespace())
    self.display_trajectory_publisher = rospy.Publisher(rospy.get_namespace() + 'move_group/display_planned_path',
                                                  moveit_msgs.msg.DisplayTrajectory,
                                                  queue_size=20)


  def reach_joint_angles(self, joint_positions, tolerance):
    self.arm_group.set_goal_joint_tolerance(tolerance)

    # Set the joint target configuration
    self.arm_group.set_joint_value_target(joint_positions)

    # Plan and execute in one command
    success = self.arm_group.go(wait=True)
    return success

  def get_cartesian_pose(self):
    # Get the cartesian position
    pose = self.arm_group.get_current_pose()
    return pose.pose

  def reach_cartesian_pose(self, pose, tolerance, constraints):
    # Set the tolerance
    self.arm_group.set_goal_position_tolerance(tolerance)

    # Set the constraints
    if constraints is not None:
      self.arm_group.set_path_constraints(constraints)

    # Set the Cartesian pose and move
    self.arm_group.set_pose_target(pose)
    return self.arm_group.go(wait=True)

  def reach_gripper_position(self, relative_position):
    # Set a relative position for the gripper
    gripper_joint = self.robot.get_joint(self.gripper_joint_name)
    gripper_max_absolute_pos = gripper_joint.max_bound()
    gripper_min_absolute_pos = gripper_joint.min_bound()

    gripper_joint.move(relative_position * (gripper_max_absolute_pos - gripper_min_absolute_pos) + gripper_min_absolute_pos, True)


def main():
  rospy.loginfo("Initializing node in namespace " + rospy.get_namespace())

  traj_handler = MoveItHandle()

  # Move vertically for 0.1m
  actual_pose = traj_handler.get_cartesian_pose()
  actual_pose.position.z -= 0.1
  traj_handler.reach_cartesian_pose(pose=actual_pose, tolerance=0.01, constraints=None)

  # Example to close the gripper (try different values to open)
  traj_handler.reach_gripper_position(0)

if __name__ == '__main__':
  main()
