#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import rospy
import actionlib

from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
#from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
import control_msgs.msg 

def send_gripper_action_actionlib():
    rospy.init_node('send_gripper_command')

    client = actionlib.SimpleActionClient('/cobotta/gripper_action', control_msgs.msg.GripperCommandAction)
    client.wait_for_server()
    
    goal = control_msgs.msg.GripperCommandGoal()
    #goal.command.position = 0.009
    goal.command.position = 0.015
    goal.command.max_effort = 20.0

    client.send_goal_and_wait(goal, rospy.Duration(10))
    rospy.loginfo("wait for goal ...")

    client.wait_for_result()
    rospy.loginfo("done")


if __name__ == '__main__':
    try:
        send_gripper_action_actionlib()
    except rospy.ROSInterruptException: pass



