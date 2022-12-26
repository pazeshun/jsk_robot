#! /usr/bin/python

import rospy
import numpy as np
import actionlib

from omni_msgs.msg import OmniState, OmniFeedback
from geometry_msgs.msg import (
    Pose, PoseStamped, Point, Quaternion, WrenchStamped)
from std_msgs.msg import Header, Bool
from trac_ik_python.trac_ik import IK
import time


def get_initial_pose_single():
    current_frame = PoseStamped()
    return PoseStamped(header=Header(frame_id='rarm_link0'),
                       pose=Pose(position=Point(0.3, 0.0, 0.48),
                                 orientation=Quaternion(1.0, 0.0, 0.0, 0.0)))


def sample_Cb(msg):
   
    vel_scale = 1.0
    target_pose.pose.position.x +=  msg.velocity.y * vel_scale
    target_pose.pose.position.y += -msg.velocity.x * vel_scale
    target_pose.pose.position.z +=  msg.velocity.z * vel_scale


    # benchmark IK time
    qinit = [0.] * ik_solver.number_of_joints
    x = y = z = 0.0
    rx = ry = rz = 0.0
    rw = 1.0
    bx = by = bz = 0.001
    brx = bry = brz = 0.1

    
    x = target_pose.pose.position.x
    y = target_pose.pose.position.y
    z = target_pose.pose.position.z

    rx = target_pose.pose.orientation.x
    ry = target_pose.pose.orientation.y
    rz = target_pose.pose.orientation.z
    rw = target_pose.pose.orientation.w


    ini_t = time.time()
    sol = ik_solver.get_ik(qinit,
                           x, y, z,
                           rx, ry, rz, rw,
                           bx, by, bz,
                           brx, bry, brz)
    fin_t = time.time()
    call_time = fin_t - ini_t
    print(msg.pose.position.x)
    #print('{} [ms]'.format(call_time * 1000.0))

    single_target_pose_pub.publish(target_pose)


if __name__ == '__main__':
    print("Sample Subscriber")

    rospy.init_node('SamplePhantomSubscriber')
    current_frame_topic = '/dual_panda/dual_arm_cartesian_pose_controller/rarm_frame'
    dev_topic = "/right_device/phantom/state"
    pose_target_pub_topic = "/dual_panda/dual_arm_cartesian_pose_controller/rarm_target_pose"

    target_pose = get_initial_pose_single()

    sampleCb = rospy.Subscriber(dev_topic, OmniState, sample_Cb)
    single_target_pose_pub = rospy.Publisher(
        pose_target_pub_topic, PoseStamped, queue_size=1)


    ik_solver = IK("base_link",
                       "J6")
    print("IK solver uses link chain:")
    print(ik_solver.link_names)
    print("")

    print("IK solver base frame:")
    print(ik_solver.base_link)
    print("")

    print("IK solver tip link:")
    print(ik_solver.tip_link)
    print("")

    print("IK solver for joints:")
    print(ik_solver.joint_names)
    print("")

    print("IK solver using joint limits:")
    lb, up = ik_solver.get_joint_limits()
    print("Lower bound: " + str(lb))
    print("Upper bound: " + str(up))
    print("")


    rospy.loginfo("Start looping")
    r = rospy.Rate(50)
    while not rospy.is_shutdown():
        r.sleep()


