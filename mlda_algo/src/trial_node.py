#!/usr/bin/python
import rospy



class Trial():
    def __init__(self):
        pass


if __name__ == '__main__':
    rospy.init_node('trial_node')
    rospy.loginfo('Trial node started')
    rospy.spin()