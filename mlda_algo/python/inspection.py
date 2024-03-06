#!/usr/bin/python
import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped, PolygonStamped
from nav_msgs.msg import Path, Odometry

class Inspection():
    def __init__(self):
        # Topic Defintions
        self.TOPIC_CMD_VEL = "/cmd_vel"
        self.TOPIC_FRONT_SCAN = "/front/scan"
        self.TOPIC_LOCAL_FOOTPRINT = "/move_base/local_costmap/footprint"
        self.TOPIC_GLOBAL_PLAN = "/move_base/TrajectoryPlannerROS/global_plan"
        self.TOPIC_LOCAL_PLAN = "/move_base/TrajectoryPlannerROS/local_plan"
        self.TOPIC_ODOM = "/odometry/filtered"
        
        # Object to store
        self.scan = LaserScan()
        self.cmd_vel = Twist()
        self.global_plan = Path()
        self.local_plan = Path()
        self.odom = Odometry()
        self.footprint = PolygonStamped()


        # Subscribe
        
        # self.subscriber = rospy.SubscribeListener(self.TOPIC, self.inspect, ca)
        pass
    
    
    def inspect(self):
        pass

if __name__ == "__main__":
    rospy.init_node('inspection_node')
    rospy.loginfo("Inspection Node Started")
    inspect = Inspection()
    while not rospy.is_shutdown():
        pass
    