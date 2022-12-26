#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import rospy
import actionlib

from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal

def send_joint_position_actionlib():
    rospy.init_node('send_joint_position')

    client = actionlib.SimpleActionClient('/cobotta/arm_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
    client.wait_for_server()
    
    goal = FollowJointTrajectoryGoal()
    goal.trajectory = JointTrajectory()
    goal.trajectory.header.stamp = rospy.Time.now()
    goal.trajectory.joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']

    point = JointTrajectoryPoint()
    #for medium position
    #point.positions = [0, 0.349065, 1.37881, 0.0, 0.349065, 0.0 ]
    #for down drill
    #point.positions = [0, 0.349065, 1.37881, 0.0, 1.8, 0.0 ]
    #point.positions = [-0.06947007730076427, 0.32785061551150196, 1.1529270484057612, -0.027447728032444847, 2.1018933196751166, -0.057238885058253704]


    #for up drill w/ microscope
    #point.positions = [0, 0.349065, 1.37881, 0.0, -1.0, 0.0 ]
    #point.positions =  [-0.09841101588691244, -0.0025998835021951118, 1.678529981048406, -0.07702804160391104, -1.540299192100872, 0.13269802081067522]
    point.positions =  [-0.1581552823655697, -0.045630608405873394, 1.7029259188836383, -0.08495923079331642, -1.5263847115420917, 0.19357662301116457]



    #for side drill
    #point.positions = [-0.08126520957438339, 0.32708126223024014, 1.1045079143370573, -1.57, 1.57, -0.5]

    duration = 5.0
    point.time_from_start = rospy.Duration(duration)
    goal.trajectory.points.append(point)

    client.send_goal(goal)
    rospy.loginfo("wait for goal ...")

    client.wait_for_result()
    rospy.loginfo("done")


if __name__ == '__main__':
    try:
        send_joint_position_actionlib()
    except rospy.ROSInterruptException: pass
