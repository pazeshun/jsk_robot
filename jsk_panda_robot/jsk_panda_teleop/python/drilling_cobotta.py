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
from omni_msgs.msg import OmniState, OmniFeedback
from geometry_msgs.msg import (
    Pose, PoseStamped, Point, Quaternion, WrenchStamped)
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal

from jsk_recognition_msgs.msg import ClassificationResult, Accuracy

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


class Microphone(object):
    def __init__(self, topic_name='/sound_classifier/output'):
        self.classifier_output = None
        self.accuracy = None

        self.classifier_output_sub = rospy.Subscriber(
            topic_name, ClassificationResult, self.classifier_output_cb)

        self.accuracy_sub = rospy.Subscriber(
            '{}/criteria'.format(topic_name), Accuracy, self.accuracy_cb)

        rospy.loginfo("Wait for /sound_classifier_output topic ... \n")
        while (self.classifier_output == None):
            rospy.wait_for_message(topic_name, ClassificationResult)

        rospy.loginfo("Wait for /sound_classifier_output/criteria topic ... \n")
        while (self.accuracy == None):
            rospy.wait_for_message('{}/criteria'.format(topic_name), Accuracy)

        self.print_status()

    def print_status(self):
        rospy.loginfo('\n{}, {}'.format(self.classifier_output, self.accuracy))

    def classifier_output_cb(self, msg):
        #msg.label_names = ['strong\n']
        #msg.label_names[0] = 'strong\n'
        self.classifier_output = msg.label_names

    def accuracy_cb(self, msg):
        self.accuracy = msg.accuracy


class AndScale(object):
    def __init__(self, topic_name='/scale_output'):
        self.value_gf = None
        self.value_N = None

        self.scale_device_sub = rospy.Subscriber(
            topic_name, Float32, self.scale_device_cb)

        rospy.loginfo("Wait for /scale_output topic ... \n")
        while (self.value_gf == None):
            rospy.wait_for_message(topic_name, Float32)
        self.print_status()

    def print_status(self):
        rospy.loginfo('{} [gf], {} [N]'.format(self.value_gf, self.value_N))
        
    def scale_device_cb(self, msg):
        self.value_gf = msg.data
        self.value_N = msg.data / 100.0


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

        self.force_sensor = AndScale()
        self.microphone = Microphone()

        self.client = actionlib.SimpleActionClient('/cobotta/arm_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        self.client.wait_for_server()

        self.duration = 0.0

        self.setup_ik()
        self.setup_ros()
        self.set_current_frame()
        self.setup_phantom()
            
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
        error_topic = '/{}/{}/has_error'.format(self.robot_id, self.arm_name)
        status_overlay_topic = '/{}/{}/status_overlay'.format(
            self.robot_id, self.arm_name)

        joint_state_topic = '/cobotta/joint_states'

        self.joint_state_sub = rospy.Subscriber(
            joint_state_topic, JointState, self.joint_state_cb)

        rospy.loginfo("Wait for joint_state_topic ... \n")
        self.joint_state_gripper == None
        self.joint_state_arm == None
        while (self.joint_state_arm == None) or (self.joint_state_gripper == None):
            rospy.wait_for_message(joint_state_topic, JointState)
        rospy.loginfo("Result ... {}\n\n{}\n\n".format(self.joint_state_arm, self.joint_state_gripper))

        self.panda_force_sub = rospy.Subscriber(
            force_topic, WrenchStamped, self.force_cb)
        self.error_topic = rospy.Subscriber(
            error_topic, Bool, self.has_error_cb)
        self.target_pose = self.set_initial_pose(no_ask=True)

        self.single_target_pose_pub = rospy.Publisher(
            pose_target_pub_topic, PoseStamped, queue_size=1)
        self.force_feedback_pub = rospy.Publisher(
            force_pub_topic, OmniFeedback, queue_size=1)
        self.status_overlay_pub = rospy.Publisher(
            status_overlay_topic, OverlayText, queue_size=1)

    def setup_ik(self):
        print('KDL chain using tcp_joint')
        self.TREE_COBOTTA = URDF.treeFromParam('robot_description')

        print('trac_IK chain using tcp_joint')
        self.ik_solver = IK("base_link",
                       "tcp") #"J6"
        self.qinit = [0.] * self.ik_solver.number_of_joints

        # print("IK solver using joint limits:")
        self.lb, self.up = self.ik_solver.get_joint_limits()
        # print("Lower bound: " + str(self.lb))
        # print("Upper bound: " + str(self.up))
        # print("")


        ik_time_topic =  '/{}/{}/ik_time'.format(
            self.robot_id, self.arm_name)
        ik_time_stamped_topic =  '/{}/{}/ik_time_stamped'.format(
            self.robot_id, self.arm_name)
        ik_result_topic =  '/{}/{}/ik_result'.format(
            self.robot_id, self.arm_name)
        ik_result_gripper_topic =  '/{}/{}/ik_result'.format(
            self.robot_id, self.arm_name)
        
        self.ik_time_msg = Float32(data=time.time())
        self.ik_time_stamped_msg = Float32Stamped(header=Header(frame_id='rarm_link0'),
                                       data=time.time())
        self.ik_result_msg = JointState(header=Header(frame_id='rarm_link0'),
                                       position=[0.] * self.ik_solver.number_of_joints)

        self.ik_time_pub = rospy.Publisher(
            ik_time_topic, Float32, queue_size=1)
        self.ik_time_stamped_pub = rospy.Publisher(
            ik_time_stamped_topic, Float32Stamped, queue_size=1)
        self.ik_result_pub = rospy.Publisher(
            ik_result_topic, JointState, queue_size=1)
        self.ik_result_gripper_pub = rospy.Publisher(
            ik_result_gripper_topic, JointState, queue_size=1)

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

    def set_current_frame(self):
        tree_cobotta = self.TREE_COBOTTA
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

    # def device_state_sub(self, msg):
    #     self.omni_state = msg
    #     # control gripper
    #     if msg.close_gripper and (not self.is_gripper_closed):
    #         rospy.loginfo("Close {} gripper".format(self.arm_name))
    #         #self.close_gripper()
    #         self.is_gripper_closed = True
    #     if (not msg.close_gripper) and self.is_gripper_closed:
    #         rospy.loginfo("Open {} gripper".format(self.arm_name))
    #         #self.open_gripper()
    #         self.is_gripper_closed = False
    #     # display status
    #     # self.display_status()
    #     if msg.locked:  # not move target pose when locked
    #         self.moving = False
    #         return

    #     # update self.target_pose
    #     self.moving = True
    #     self.target_pose.pose.position.x += -msg.velocity.x * self.vel_scale
    #     self.target_pose.pose.position.y += -msg.velocity.y * self.vel_scale
    #     self.target_pose.pose.position.z +=  msg.velocity.z * self.vel_scale

    #     self.target_pose.header.stamp = rospy.Time.now()

    #     # publish target pose
    #     self.single_target_pose_pub.publish(self.target_pose)

    #solving IK based on the target_pose value
    def IK_from_pose(self, x, y, z, rx, ry, rz, rw):
        bx = by = bz = 0.001
        brx = bry = brz = 0.1

        sol = None
        while (sol == None):
            sol = self.ik_solver.get_ik(self.qinit,
                                                     x, y, z,
                                                     rx, ry, rz, rw,
                                                     bx, by, bz,
                                                     brx, bry, brz)

        #self.ik_result_msg.header = Header(frame_id='rarm_link0')
        self.ik_result_msg.header = Header(frame_id='{}_link0'.format(self.arm_name))
        self.ik_result_msg.header.stamp = rospy.Time.now()
        if not (sol == None):
            self.ik_result_msg.name = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
            self.ik_result_msg.position = sol
            for i in range(self.ik_solver.number_of_joints):
                self.qinit[i] = sol[i]

        self.ik_result_pub.publish(self.ik_result_msg)

        # for gripper
        self.ik_result_msg.header = Header(frame_id='{}_link0'.format(self.arm_name))
        self.ik_result_msg.header.stamp = rospy.Time.now()
        if not (sol == None):
            self.ik_result_msg.name = ['joint_gripper']
            self.ik_result_msg.position = [0.015026816717667426]
        self.ik_result_pub.publish(self.ik_result_msg)

        return sol

    #move real robot using JTA
    def move(self, sol, duration, sleep=False):
        #setting JTA
        self.goal = FollowJointTrajectoryGoal()
        self.goal.trajectory = JointTrajectory()
        self.goal.trajectory.header.stamp = rospy.Time.now()
        self.goal.trajectory.joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']

        point = JointTrajectoryPoint()
        point.positions = sol

        self.duration = duration

        #for sending the command to real robot
        point.time_from_start = rospy.Duration(self.duration)
        self.goal.trajectory.points.append(point)
        self.client.send_goal(self.goal)

        if sleep == True:
            rospy.sleep(duration)

    def force_cb(self, msg):
        if self.zero_force is None:
            self.zero_force = np.array([msg.wrench.force.x,
                                        msg.wrench.force.y,
                                        msg.wrench.force.z])
        self.target_force.force.x = (msg.wrench.force.y - self.zero_force[0]) * self.force_scale
        self.target_force.force.y = - (msg.wrench.force.x - self.zero_force[1]) * self.force_scale
        self.target_force.force.z = (msg.wrench.force.z - self.zero_force[2]) * self.force_scale


    def set_initial_pose(self, no_ask=False):
        if not no_ask:
            res = raw_input(
                'Press Enter to set this {} pose as initial position'.format(
                    self.arm_name))
        self.current_frame.header.frame_id = '{}_link0'.format(self.arm_name)
        #self.current_frame.header.frame_id = '{}_link0'.format(self.arm_name)
        self.current_frame.header.stamp = rospy.Time.now()

        return self.current_frame

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
        r_initial_pose, l_initial_pose = self.set_initial_pose()

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

    #drilling with circular path
    def drill_circular_path(self,
                            radius = 0.0025,
                            delta_z = 0.005,
                            force_threshold_gf = 50, 
                            loops = 5,
                            sound_target = 0.75,
                            no_sound_ratio_gain = 4.0,
                            sound_feedback_pgain = 0.25,#1.0,
                            circle_div_num = 8,
                            z_top_list = None,
                            duration_r = 0.5,
                            duration_theta = 0.5,
                            duration_z_up = 1.0, 
                            duration_z_down = 10.0):

        if z_top_list == None:
            z_top_list = [0.] * circle_div_num

        prev_sound_label_list = ['strong\n'] * circle_div_num
        prev_sound_intensity_list = [sound_target] * circle_div_num
        modified_ik_z_list = [0.] * circle_div_num

        print('drill_circular_path')
        r = rospy.Rate(100)

        self.rarm_handler.set_current_frame()

        #setting IK parameters
        x_center = self.rarm_handler.target_pose.pose.position.x
        y_center = self.rarm_handler.target_pose.pose.position.y
        z_top = self.rarm_handler.target_pose.pose.position.z

        rx = self.rarm_handler.target_pose.pose.orientation.x
        ry = self.rarm_handler.target_pose.pose.orientation.y
        rz = self.rarm_handler.target_pose.pose.orientation.z
        rw = self.rarm_handler.target_pose.pose.orientation.w

        bx = by = bz = 0.001
        brx = bry = brz = 0.1

        sol = None

        #top center -> top (theta=0) point
        sol = self.rarm_handler.IK_from_pose(x_center + radius * np.sin(0),
                                             y_center + radius * np.cos(0),
                                             z_top,
                                             rx, ry, rz, rw)

        self.target_pub.publish(self.target_pose)
        self.rarm_handler.move(sol, duration_r, sleep = True)

        #circular path for (loops) times
        for j in range(loops):
            for i in range(circle_div_num):
                theta = i / float(circle_div_num) * 2.0 * np.pi

                #set displacement of drill height based on the sound intensity
                if prev_sound_label_list[i][0] == 'no_sound\n':
                    modified_ik_z_list[i] += no_sound_ratio_gain * sound_feedback_pgain * (prev_sound_intensity_list[i] - sound_target)
                else:
                    modified_ik_z_list[i] += sound_feedback_pgain * (prev_sound_intensity_list[i] - sound_target)

                sol = self.rarm_handler.IK_from_pose(x_center + radius * np.sin(theta),
                                                     y_center + radius * np.cos(theta),
                                                     z_top - z_top_list[i] + modified_ik_z_list[i] * 0.001,
                                                     rx, ry, rz, rw)

                self.target_pub.publish(self.target_pose)
                self.rarm_handler.move(sol, duration_theta, sleep = True)

                prev_sound_label_list[i] = self.rarm_handler.microphone.classifier_output
                prev_sound_intensity_list[i] = self.rarm_handler.microphone.accuracy

                rospy.loginfo('{}, {}'.format(prev_sound_label_list[i], prev_sound_intensity_list[i]))
            rospy.loginfo('{}'.format(modified_ik_z_list))

        #bottom (theta=0) point -> top (theta=0) point -> top center
        sol = self.rarm_handler.IK_from_pose(x_center + radius * np.sin(0),
                                             y_center + radius * np.cos(0),
                                             z_top + modified_ik_z_list[0] * 0.001,
                                             rx, ry, rz, rw)

        self.target_pub.publish(self.target_pose)
        self.rarm_handler.move(sol, duration_theta, sleep = True)


        #top (theta=0) point -> top center
        sol = self.rarm_handler.IK_from_pose(x_center,
                                             y_center,
                                             z_top,
                                             rx, ry, rz, rw)

        self.target_pub.publish(self.target_pose)
        self.rarm_handler.move(sol, duration_z_up, sleep = True)

        rospy.loginfo("Finished circular path...")


    #drilling with discrete points
    def drill_discrete_points(self,
                              radius = 0.0025,
                              delta_z = 0.005,
                              force_touch_gf = 2,
                              force_stop_gf = 50, 
                              circle_div_num = 8,
                              duration_r = 0.3,
                              duration_theta = 0.2,
                              duration_z_up = 1.0, 
                              duration_z_down = 10.0):

        r = rospy.Rate(100)

        egg_top_list = [0.] * circle_div_num
        egg_bottom_list = [0.] * circle_div_num
        egg_limit_list = [egg_top_list, egg_bottom_list]

        self.rarm_handler.set_current_frame()

        #setting IK parameters
        x_center = self.rarm_handler.target_pose.pose.position.x
        y_center = self.rarm_handler.target_pose.pose.position.y
        z_top = self.rarm_handler.target_pose.pose.position.z

        rx = self.rarm_handler.target_pose.pose.orientation.x
        ry = self.rarm_handler.target_pose.pose.orientation.y
        rz = self.rarm_handler.target_pose.pose.orientation.z
        rw = self.rarm_handler.target_pose.pose.orientation.w

        bx = by = bz = 0.001
        brx = bry = brz = 0.1

        sol = None


        #drawing a (top) circle around the current position.
        #i = 0 ... r: 0 -> radius
        #i = circle_div_num ... r: radius -> 0
        for i in range(circle_div_num+1):
            touch_flag = False
            theta = i / float(circle_div_num) * 2.0 * np.pi

            x = x_center + radius * np.sin(theta)
            y = y_center + radius * np.cos(theta)
            z = z_top
            duration = duration_theta
            j_MAX = 3

            if (i == 0):
                duration = duration_r

            if (i == circle_div_num):
                x = x_center
                y = y_center
                duration = duration_r
                j_MAX = 1

            #vertical motion
            #j = 0 ... top (on top circle path)
            #j = 1 ... to bottom
            #j = 2 ... to top
            for j in range(j_MAX):
                if j == 0:
                    z = z_top
                elif j == 1:
                    z = z_top - delta_z
                    duration = duration_z_down
                else:
                    z = z_top
                    duration = duration_z_up

                sol = self.rarm_handler.IK_from_pose(x,
                                                     y,
                                                     z,
                                                     rx, ry, rz, rw)

                self.rarm_handler.move(sol, duration, sleep = False)
                self.target_pub.publish(self.target_pose)

                if (j == 1):
                    ini_t = time.time()
                    while not rospy.is_shutdown():
                        r.sleep()
                        cur_t = time.time()
                        elapsed_t = cur_t - ini_t

                        if (self.rarm_handler.force_sensor.value_gf > force_touch_gf):
                            if (touch_flag == False):
                                rospy.loginfo("{}/{} Touch!".format(i, circle_div_num - 1))
                                self.rarm_handler.force_sensor.print_status()
                                self.rarm_handler.set_current_frame()
                                touch_flag = True
                                egg_top_list[i] = z_top - self.rarm_handler.current_frame.pose.position.z

                        if (self.rarm_handler.force_sensor.value_gf > force_stop_gf):
                            rospy.loginfo("{}/{} Interrupt!".format(i, circle_div_num - 1))
                            self.rarm_handler.force_sensor.print_status()
                            self.rarm_handler.client.cancel_goal()
                            self.rarm_handler.set_current_frame()
                            egg_bottom_list[i] = z_top - self.rarm_handler.current_frame.pose.position.z
                            break

                        if (elapsed_t > duration):
                            egg_bottom_list[i] = delta_z
                            break

                else:
                    rospy.sleep(duration)

        rospy.loginfo('{}'.format(egg_limit_list))
        rospy.loginfo("Finished point drills...")
        return egg_limit_list


    #tracking TouchX
    def loop_call(self):
        # send target pose for both arm if connected
        if self.pose_connecting:
            self.target_pose.right_target = self.rarm_handler.target_pose
            self.target_pose.left_target = self.larm_handler.target_pose

            x = y = z = 0.0
            rx = ry = rz = 0.0
            rw = 1.0
            bx = by = bz = 0.001
            brx = bry = brz = 0.1
            #bx = by = bz = 0.001
            #brx = bry = brz = 0.3

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

            # for gripper. temporary
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


    def run(self):
        r = rospy.Rate(100)
        N = 8
        z_top_list = [0.] * N
        egg_limit_list = self.drill_discrete_points(circle_div_num = N)
        #z_top_list = [0.002800617735072175, 0.0027782194598259813, 0.0026013272743231064, 0.00277617417721282, 0.002843016124182085, 0.003072497055753576, 0.003058434517541231, 0.002979028158652419]
        [egg_top_list, egg_bottom_list] = egg_limit_list
        self.drill_circular_path(circle_div_num = N, z_top_list = egg_top_list)
        rospy.loginfo("Start looping")
        while not rospy.is_shutdown():
            break
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
