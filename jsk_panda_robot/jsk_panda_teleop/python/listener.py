#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from omni_msgs.msg import OmniState, OmniFeedback

def callback(msg):
    rospy.loginfo(rospy.get_caller_id() + "I heard %s ", msg.velocity.x)

def listener():
    rospy.init_node("listener", anonymous = True)
    #rospy.Subscriber("chatter", String, callback)
    sampleCb = rospy.Subscriber("/right_device/phantom/state_drop", OmniState, callback)

    print("Waiting for a topic...\n")
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        rate.sleep()

if __name__ == '__main__':
    listener()
