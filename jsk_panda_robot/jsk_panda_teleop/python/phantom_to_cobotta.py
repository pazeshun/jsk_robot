#! /usr/bin/python
from absl import app
from absl import flags

import rospy
import numpy as np
import actionlib
# import franka_gripper.msg
# from franka_gripper.msg import GraspEpsilon
# import franka_msgs.msg
from scipy.spatial.transform import Rotation as R
from omni_msgs.msg import OmniButtonEvent, OmniState, OmniFeedback
from geometry_msgs.msg import (
    Pose, PoseStamped, Point, Quaternion, WrenchStamped)
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal

from jsk_rviz_plugins.msg import OverlayText
from std_msgs.msg import Header, Bool, Float32

# from franka_example_controllers.msg import ArmsTargetPose
from jsk_panda_teleop.msg import ArmsTargetPose
from jsk_panda_teleop.srv import ControlBilateral,  ControlBilateralResponse

from trac_ik_python.trac_ik import IK
import PyKDL

import kdl_parser_py.urdf as URDF 
#from urdf_parser_py.urdf import URDF

from numpy.random import random
from std_msgs_stamped.msg import Float32Stamped
from std_msgs_stamped.msg import Float32MultiArrayStamped
import time



FLAGS = flags.FLAGS

flags.DEFINE_float(
    'vel_scale',
    1.0,
    'Velocicy scale of master/slave. Larger value means'
    'more movement in robot.')

flags.DEFINE_bool(
    'connect_pose',
    True,
    'Set this to connect TouchX and Panda position'
    'command from start. Otherwise you have to call rosservice.')
flags.DEFINE_bool(
    'connect_force',
    False,
    'Set this to connect TouchX and Panda force'
    'command from start. Otherwise you have to call rosservice.')


def get_status_overlay(moving, arm_name):
    text = OverlayText()
    text.width = 400
    text.height = 600
    text.left = 10 if arm_name == 'larm' else 600
    text.top = 10
    text.text_size = 25
    text.line_width = 2
    text.font = "DejaVu Sans Mono"
    on_txt = """<span style="color: red;">ON</span>"""
    off_txt = """<span style="color: white;">OFF</span>"""
    on_or_off = on_txt if moving else off_txt
    text.text = "TouchX control {}".format(on_or_off)
    return text


# def get_initial_pose():
#     r_i = PoseStamped(header=Header(frame_id='rarm_link0'),
#                       pose=Pose(position=Point(0.3, 0.0, 0.48),
#                                 orientation=Quaternion(0.0, 1.0, 0.0, 0.0)))
#     l_i = PoseStamped(header=Header(frame_id='larm_link0'),
#                       pose=Pose(position=Point(0.3, 0.0, 0.48),
#                                 orientation=Quaternion(1.0, 0.0, 0.0, 0.0)))
#     return r_i, l_i


# def get_initial_pose_single(arm_name):
#     return PoseStamped(header=Header(frame_id='{}_link0'.format(arm_name)),
#                        pose=Pose(position=Point(0.3, 0.0, 0.48),
#                                  orientation=Quaternion(1.0, 0.0, 0.0, 0.0)))


class SingleArmHandler(object):

    def __init__(self, robot_id='dual_panda', arm_name='larm',
                 master_dev_name='left_device', vel_scale=1.0,
                 force_scale=0.1, send_rot=True):
        self.robot_id = robot_id
        self.arm_name = arm_name
        self.side = 'left' if arm_name == 'larm' else 'right'
        self.master_dev_name = master_dev_name
        # [100Hz] * [m/mm] * [1.0 (scaler)]
        self.vel_scale = 0.01 * 0.001 * vel_scale
        self.force_scale = force_scale
        self.send_rot = send_rot
        self.message_ct = 0
        self.message_time = 0
        self.message_time_delta = 0


        # initial rotation of the TouchX
        #master_initial_q = R.from_quat([-0.5, -0.5, -0.5, 0.5])
        #self.master_initial_q = master_initial_q

        # initial rotation of the Cobotta Arm
        self.cobotta_initial_q = None

        #cobotta_initial_q = R.from_quat([1.0, 0, 0, 0])

        # initial rotation of the TouchX
        self.master_initial_q = None

        # master_initial_q = R.from_quat([self.omni_state_initial.pose.orientation.x,
        #                                 self.omni_state_initial.pose.orientation.y,
        #                                 self.omni_state_initial.pose.orientation.z,
        #                                 self.omni_state_initial.pose.orientation.w])

        # conversion from TouchX to Cobotta
        self.q_m_p = None
        #self.q_m_p = cobotta_initial_q * master_initial_q.inv()


        self.target_force = OmniFeedback()
        self.zero_force = None  # for initializatoin
        self.current_frame = PoseStamped()
        self.is_gripper_closed = False
        self.has_error = False
        self.moving = False

        self.joint_state_arm = None
        self.joint_state_gripper = None
        #self.omni_state = OmniState()
        #self.omni_state_initial = OmniState()

        self.setup_ik()
        self.setup_ros()
        self.setup_current_frame()
        self.setup_phantom()
            
        # open gripper
        # self.open_gripper()


    def setup_ros(self):
        current_frame_topic = '/{}/dual_arm_cartesian_pose_controller/{}_frame'.format(
            self.robot_id, self.side)
        dev_topic = "/{}/phantom/state_drop".format(self.master_dev_name)
        force_topic = "{}/{}_state_controller/F_ext".format(
            self.robot_id, self.arm_name)
        force_pub_topic = "/{}/phantom/force_feedback".format(
            self.master_dev_name)
        pose_target_pub_topic = '/{}/dual_arm_cartesian_pose_controller/{}_target_pose'.format(
            self.robot_id, self.arm_name)
        # gripper_topic = '/{}/{}/franka_gripper/grasp'.format(
        #     self.robot_id, self.arm_name)
        error_topic = '/{}/{}/has_error'.format(self.robot_id, self.arm_name)
        status_overlay_topic = '/{}/{}/status_overlay'.format(
            self.robot_id, self.arm_name)

        joint_state_topic = '/cobotta/joint_states'

        ik_time_topic =  '/{}/{}/ik_time'.format(
            self.robot_id, self.arm_name)
        ik_time_stamped_topic =  '/{}/{}/ik_time_stamped'.format(
            self.robot_id, self.arm_name)
        ik_result_topic =  '/{}/{}/ik_result'.format(
            self.robot_id, self.arm_name)
        ik_result_gripper_topic =  '/{}/{}/ik_result'.format(
            self.robot_id, self.arm_name)

        
        self.joint_state_sub = rospy.Subscriber(
            joint_state_topic, JointState, self.joint_state_cb)

        rospy.loginfo("Wait for joint_state_topic ... \n")
        self.joint_state_gripper == None
        self.joint_state_arm == None
        while (self.joint_state_arm == None) or (self.joint_state_gripper == None):
            rospy.wait_for_message(joint_state_topic, JointState)
        rospy.loginfo("Result ... {}\n\n{}\n\n".format(self.joint_state_arm, self.joint_state_gripper))


        self.device_state_sub = rospy.Subscriber(
            dev_topic, OmniState, self.device_state_sub)
        self.device_button_sub = rospy.Subscriber(
            dev_topic, OmniButtonEvent, self.device_button_sub)

        self.panda_force_sub = rospy.Subscriber(
            force_topic, WrenchStamped, self.force_cb)
        self.error_topic = rospy.Subscriber(
            error_topic, Bool, self.has_error_cb)
        self.target_pose = self.set_initial_pose(no_ask=True)
        self.ik_time_msg = Float32(data=time.time())
        self.ik_time_stamped_msg = Float32Stamped(header=Header(frame_id='rarm_link0'),
                                       data=time.time())
        # self.ik_result_msg = Float32MultiArrayStamped(header=Header(frame_id='rarm_link0'),
        #                                data=[0.] * self.ik_solver.number_of_joints)
        self.ik_result_msg = JointState(header=Header(frame_id='rarm_link0'),
                                       position=[0.] * self.ik_solver.number_of_joints)

        self.single_target_pose_pub = rospy.Publisher(
            pose_target_pub_topic, PoseStamped, queue_size=1)
        self.force_feedback_pub = rospy.Publisher(
            force_pub_topic, OmniFeedback, queue_size=1)
        self.status_overlay_pub = rospy.Publisher(
            status_overlay_topic, OverlayText, queue_size=1)
        self.ik_time_pub = rospy.Publisher(
            ik_time_topic, Float32, queue_size=1)
        self.ik_time_stamped_pub = rospy.Publisher(
            ik_time_stamped_topic, Float32Stamped, queue_size=1)
        # self.ik_result_pub = rospy.Publisher(
        #     ik_result_topic, Float32MultiArrayStamped, queue_size=1)
        self.ik_result_pub = rospy.Publisher(
            ik_result_topic, JointState, queue_size=1)
        self.ik_result_gripper_pub = rospy.Publisher(
            ik_result_gripper_topic, JointState, queue_size=1)

        # self.gripper_client = actionlib.SimpleActionClient(
        #     gripper_topic, franka_gripper.msg.GraspAction)
        # self.gripper_client.wait_for_server()


    def setup_ik(self):
        print('trac_IK chain using tcp_joint')
        self.ik_solver = IK("base_link",
                       "tcp") #"J6"

        
        # print("IK solver uses link chain:")
        # print(self.ik_solver.link_names)
        # print("")

        # print("IK solver base frame:")
        # print(self.ik_solver.base_link)
        # print("")

        # print("IK solver tip link:")
        # print(self.ik_solver.tip_link)
        # print("")

        # print("IK solver for joints:")
        # print(self.ik_solver.joint_names)
        # print("")

        # print("IK solver using joint limits:")
        self.lb, self.up = self.ik_solver.get_joint_limits()
        # print("Lower bound: " + str(self.lb))
        # print("Upper bound: " + str(self.up))
        # print("")


    def setup_current_frame(self):
        self.qinit = [0.] * self.ik_solver.number_of_joints

        print('KDL chain using tcp_joint')
        tree_cobotta = URDF.treeFromParam('robot_description')
        chain_cobotta = tree_cobotta[1].getChain('base_link', 'tcp') #'J6'
        jointAngles_cobotta = PyKDL.JntArray(self.ik_solver.number_of_joints)
        for i in range(self.ik_solver.number_of_joints):
            self.qinit[i] = self.joint_state_arm.position[i]
            jointAngles_cobotta[i] = self.joint_state_arm.position[i]
        fk_cobotta = PyKDL.ChainFkSolverPos_recursive(chain_cobotta)
        finalFrame_cobotta = PyKDL.Frame()
        fk_cobotta.JntToCart(jointAngles_cobotta, finalFrame_cobotta)

        (x, y, z) = finalFrame_cobotta.p
        (rx, ry, rz, rw) = finalFrame_cobotta.M.GetQuaternion()

        self.current_frame.header.stamp = rospy.Time.now()
        self.current_frame.pose.position.x = finalFrame_cobotta.p[0]
        self.current_frame.pose.position.y = finalFrame_cobotta.p[1]
        self.current_frame.pose.position.z = finalFrame_cobotta.p[2]
        self.current_frame.pose.orientation.x = rx
        self.current_frame.pose.orientation.y = ry
        self.current_frame.pose.orientation.z = rz
        self.current_frame.pose.orientation.w = rw


    def setup_phantom(self):
        print("setup phantom...")
        rospy.wait_for_message('/right_device/phantom/state_drop', OmniState)
        self.omni_state = OmniState
        self.omni_state_initial = self.omni_state
        print("{}".format(self.omni_state_initial))

        #rospy.wait_for_message("/{}/phantom/state_drop".format(self.master_dev_name), OmniState)


    def joint_state_cb(self, msg):
        if (msg.name==['joint_gripper']):
            self.joint_state_gripper = msg
        else:
            self.joint_state_arm = msg


    def device_button_sub(self, msg):
        self.omni_button = msg
        # control gripper


    def device_state_sub(self, msg):
        self.omni_state = msg

        # control gripper
        if msg.close_gripper and (not self.is_gripper_closed):
            rospy.loginfo("Close {} gripper".format(self.arm_name))
            #self.close_gripper()
            self.is_gripper_closed = True
        if (not msg.close_gripper) and self.is_gripper_closed:
            rospy.loginfo("Open {} gripper".format(self.arm_name))
            #self.open_gripper()
            self.is_gripper_closed = False
        # display status
        # self.display_status()
        if msg.locked:  # not move target pose when locked
            self.moving = False
            return

        # update self.target_pose
        self.moving = True
        self.target_pose.pose.position.x += -msg.velocity.x * self.vel_scale
        self.target_pose.pose.position.y += -msg.velocity.y * self.vel_scale
        self.target_pose.pose.position.z +=  msg.velocity.z * self.vel_scale


        # if self.send_rot:
        #     if self.q_m_p == None:
        #         #cobotta_initial_q = R.from_quat([1.0, 0, 0, 0])
        #         #self.q_m_p = cobotta_initial_q * master_initial_q.inv()
        #         cobotta_initial_q = R.from_quat([self.current_frame.pose.orientation.x,
        #                                          self.current_frame.pose.orientation.y,
        #                                          self.current_frame.pose.orientation.z,
        #                                          self.current_frame.pose.orientation.w])

        #         master_initial_q = R.from_quat([msg.pose.orientation.x,
        #                                         msg.pose.orientation.y,
        #                                         msg.pose.orientation.z,
        #                                         msg.pose.orientation.w])
        #         self.q_m_p = cobotta_initial_q * master_initial_q.inv()

        #     cur_q = R.from_quat([msg.pose.orientation.x,
        #                          msg.pose.orientation.y,
        #                          msg.pose.orientation.z,
        #                          msg.pose.orientation.w])

        #     tar_q = self.q_m_p * cur_q    # apply conversion
        #     self.target_pose.pose.orientation.x = -tar_q.as_quat()[0]
        #     self.target_pose.pose.orientation.y =  tar_q.as_quat()[1]
        #     self.target_pose.pose.orientation.z = -tar_q.as_quat()[2]
        #     self.target_pose.pose.orientation.w =  tar_q.as_quat()[3]

        self.target_pose.header.stamp = rospy.Time.now()


        # publish target pose
        self.single_target_pose_pub.publish(self.target_pose)


    def force_cb(self, msg):
        if self.zero_force is None:
            self.zero_force = np.array([msg.wrench.force.x,
                                        msg.wrench.force.y,
                                        msg.wrench.force.z])
        self.target_force.force.x = (msg.wrench.force.y - self.zero_force[0]) * self.force_scale
        self.target_force.force.y = - (msg.wrench.force.x - self.zero_force[1]) * self.force_scale
        self.target_force.force.z = (msg.wrench.force.z - self.zero_force[2]) * self.force_scale

    def apply_target_force(self):
        self.force_feedback_pub.publish(self.target_force)

    def set_initial_pose(self, no_ask=False):
        if not no_ask:
            res = raw_input(
                'Press Enter to set this {} pose as initial position'.format(
                    self.arm_name))
        self.current_frame.header.frame_id = '{}_link0'.format(self.arm_name)
        #self.current_frame.header.frame_id = '{}_link0'.format(self.arm_name)
        self.current_frame.header.stamp = rospy.Time.now()

        return self.current_frame

    def open_gripper(self):
        # TODO
        epsilon = GraspEpsilon(inner=0.01, outer=0.01)
        # goal = franka_gripper.msg.GraspGoal(
        #     width=0.08, speed=1, force=10, epsilon=epsilon)
        # self.gripper_client.send_goal(goal)

    def close_gripper(self):
        # TODO, currently max force 140[N] is applied
        epsilon = GraspEpsilon(inner=0.01, outer=0.01)
        # goal = franka_gripper.msg.GraspGoal(
        #     width=0, speed=1, force=140, epsilon=epsilon)
        # self.gripper_client.send_goal(goal)

    def has_error_cb(self, msg):
        self.has_error = msg.data

    def display_status(self):
        msg = get_status_overlay(moving=self.moving, arm_name=self.arm_name)
        self.status_overlay_pub.publish(msg)
        
    def __del__(self):
        rospy.loginfo("Exiting {} handelr".format(self.arm_name))



class DualArmHandler(object):

    def __init__(self, vel_scale=1.0,
                 pose_connecting=False, force_connecting=False):
        rospy.init_node('DualPhantomMaster')

        self.pose_connecting = pose_connecting
        self.force_connecting = force_connecting

        self.rarm_handler = SingleArmHandler(
            arm_name='rarm', master_dev_name='right_device',
            vel_scale=vel_scale, send_rot=True)
        self.larm_handler = SingleArmHandler(
            arm_name='larm', master_dev_name='left_device',
            vel_scale=vel_scale, send_rot=True)
        self.arms = [self.rarm_handler, self.larm_handler]

        self.target_pub = rospy.Publisher(
            '/dual_panda/dual_arm_cartesian_pose_controller/arms_target_pose',
            ArmsTargetPose, queue_size=1)
        # self.recover_error_server = actionlib.SimpleActionClient(
        #     '/dual_panda/error_recovery', franka_msgs.msg.ErrorRecoveryAction)
        r_initial_pose, l_initial_pose = self.set_initial_pose()


        self.client = actionlib.SimpleActionClient('/cobotta/arm_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        self.client.wait_for_server()

        self.duration = 0.0



        self.target_pose = ArmsTargetPose(
            right_target=r_initial_pose, left_target=l_initial_pose)

        self.control_bilateral_srv = rospy.Service(
            '/dual_panda/control_bilateral',
            ControlBilateral, self.control_bilateral)


    def control_bilateral(self, req):
        rospy.loginfo(
            "Applying bilateral connection status, changing in {}...".
            format(req.wait))
        rospy.sleep(req.wait)
        self.pose_connecting = req.pose_connecting
        self.force_connecting = req.force_connecting
        if req.reset_phantom:
            r_initial_pose, l_initial_pose = self.set_initial_pose()
            self.target_pose = ArmsTargetPose(
                right_target=r_initial_pose, left_target=l_initial_pose)
        return ControlBilateralResponse(True)

    def set_initial_pose(self):
        return (self.rarm_handler.set_initial_pose(no_ask=True),
                self.larm_handler.set_initial_pose(no_ask=True))

    def loop_call(self):
        # send target pose for both arm if connected
        if self.pose_connecting:
            self.target_pose.right_target = self.rarm_handler.target_pose
            self.target_pose.left_target = self.larm_handler.target_pose

            x = y = z = 0.0
            rx = ry = rz = 0.0
            rw = 1.0
            #bx = by = bz = 0.001
            #brx = bry = brz = 0.1
            bx = by = bz = 0.001
            brx = bry = brz = 0.3

            x = self.rarm_handler.target_pose.pose.position.x
            y = self.rarm_handler.target_pose.pose.position.y
            z = self.rarm_handler.target_pose.pose.position.z

            rx = self.rarm_handler.target_pose.pose.orientation.x
            ry = self.rarm_handler.target_pose.pose.orientation.y
            rz = self.rarm_handler.target_pose.pose.orientation.z
            rw = self.rarm_handler.target_pose.pose.orientation.w

            ini_t = time.time()

            sol = None
            while sol==None:
                sol = self.rarm_handler.ik_solver.get_ik(self.rarm_handler.qinit,
                                                     x, y, z,
                                                     rx, ry, rz, rw,
                                                     bx, by, bz,
                                                     brx, bry, brz)

            fin_t = time.time()
            call_time = fin_t - ini_t

            self.rarm_handler.ik_time_msg.data = float(call_time * 1000.0)
            self.rarm_handler.ik_time_pub.publish(self.rarm_handler.ik_time_msg)

            self.rarm_handler.ik_time_stamped_msg.header = Header(frame_id='rarm_link0')
            self.rarm_handler.ik_time_stamped_msg.header.stamp = rospy.Time.now()
            self.rarm_handler.ik_time_stamped_msg.data = float(call_time * 1000.0)
            self.rarm_handler.ik_time_stamped_pub.publish(self.rarm_handler.ik_time_stamped_msg)


            self.rarm_handler.ik_result_msg.header = Header(frame_id='rarm_link0')
            self.rarm_handler.ik_result_msg.header.stamp = rospy.Time.now()
            if not (sol == None):
                self.rarm_handler.ik_result_msg.name = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
                self.rarm_handler.ik_result_msg.position = sol
                self.rarm_handler.qinit = sol
                #self.rarm_handler.ik_result_msg.position = self.rarm_handler.qinit

            self.rarm_handler.ik_result_pub.publish(self.rarm_handler.ik_result_msg)


            self.goal = FollowJointTrajectoryGoal()
            self.goal.trajectory = JointTrajectory()
            self.goal.trajectory.joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']

            self.goal.trajectory.header.stamp = rospy.Time.now()
            point = JointTrajectoryPoint()
            point.positions = sol

            #self.duration = 0.02
            #self.duration = 0.5
            self.duration = 0.25

            #for sending the command to real robot
            point.time_from_start = rospy.Duration(self.duration)
            self.goal.trajectory.points.append(point)
            self.client.send_goal(self.goal)

            # for gripper
            self.rarm_handler.ik_result_msg.header = Header(frame_id='rarm_link0')
            self.rarm_handler.ik_result_msg.header.stamp = rospy.Time.now()
            if not (sol == None):
                #self.rarm_handler.ik_result_msg.data = sol
                self.rarm_handler.ik_result_msg.name = ['joint_gripper']
                self.rarm_handler.ik_result_msg.position = [0.015026816717667426]
            self.rarm_handler.ik_result_pub.publish(self.rarm_handler.ik_result_msg)

            if not (sol == None):
                self.target_pub.publish(self.target_pose)

        else:
            self.rarm_handler.target_pose = self.rarm_handler.set_initial_pose(
                no_ask=True)
            self.larm_handler.target_pose = self.larm_handler.set_initial_pose(
                no_ask=True)

        # apply force feedback if connected
        if self.force_connecting:
            for arm in self.arms:
                arm.apply_target_force()

        # recover error
        if self.rarm_handler.has_error or self.larm_handler.has_error:
            rospy.loginfo("Detected error in controller, recovering...")
            # recover_goal = franka_msgs.msg.ErrorRecoveryActionGoal()
            # self.recover_error_server.send_goal(recover_goal)


    def run(self):
        rospy.loginfo("Start looping")


        r = rospy.Rate(50)
        while not rospy.is_shutdown():
            self.loop_call()
            r.sleep()
        print('Program finished...')

    def __del__(self):
        rospy.loginfo("Exiting phantom master node")


def main(argv):
    node = DualArmHandler(vel_scale=FLAGS.vel_scale,
                          pose_connecting=FLAGS.connect_pose,
                          force_connecting=FLAGS.connect_force)
    # node = SingleArmHandler(
    #     arm_name='rarm', master_dev_name='right_device',
    #     vel_scale=FLAGS.vel_scale)

    node.run()


if __name__ == '__main__':
    app.run(main)
