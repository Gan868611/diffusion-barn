#!/usr/bin/python3
import casadi
import rospy
import numpy as np
import mpc_algo 
import math


from nav_msgs.msg import Path, Odometry, OccupancyGrid
from geometry_msgs.msg import Twist, PoseStamped,PolygonStamped, Quaternion
# import tf

class ROSNode():
    def __init__(self):
        self.TOPIC_VEL = "/cmd_vel"
        self.TOPIC_GLOBAL_PLAN = "/move_base/TrajectoryPlannerROS/global_plan"
        self.TOPIC_LOCAL_PLAN = "/move_base/TrajectoryPlannerROS/local_plan"
        self.TOPIC_ODOM = "/odometry/filtered"
        self.TOPIC_MPC_PLAN = "/mpc_plan"
        
        self.pub_vel = rospy.Publisher(self.TOPIC_VEL, Twist, queue_size=1)
        self.pub_mpc  = rospy.Publisher(self.TOPIC_MPC_PLAN, Path, queue_size=1)
        
        self.sub_odometry = rospy.Subscriber(self.TOPIC_ODOM, Odometry, self.callback_odom)
        self.sub_global_plan = rospy.Subscriber(self.TOPIC_GLOBAL_PLAN, Path, self.callback_global_plan)
        self.sub_local_plan = rospy.Subscriber(self.TOPIC_LOCAL_PLAN, Path, self.callback_local_plan)

        self.cmd_vel = Twist()
        self.odometry = Odometry()
        self.global_plan = Path()
        self.local_plan = Path()
        
        
        self.mpc = mpc_algo.NMPC()
        
    
    def callback_odom(self,data):
        self.odometry = data
        yaw = self.quaternion_to_yaw(data.pose.pose.orientation)
        print("Theta: ", yaw)

    def callback_global_plan(self,data):
        self.global_plan = data
        print("Global")
        
    def callback_local_plan(self, data):
        self.local_plan = data
        print("Local")

    def quaternion_to_yaw(self, orientation):
    # Convert quaternion orientation data to yaw angle of robot
        q0 = orientation.x
        q1 = orientation.y
        q2 = orientation.z
        q3 = orientation.w
        theta = math.atan2(2.0*(q2*q3 + q0*q1), 1.0 - 2.0*(q1*q1 + q2*q2))

        return theta
    
    def publish_trajectory(self, mpc_x_traj, mpc_y_traj, theta_traj):
        mpc_traj_msg = Path()
        mpc_traj_msg.header.stamp = rospy.Time.now()
        mpc_traj_msg.header.frame_id = "base_link"
        for i in range(len(mpc_x_traj)):
            pose = PoseStamped()
            pose.pose.position.x = mpc_x_traj[i]
            pose.pose.position.y = mpc_y_traj[i]
            pose.pose.orientation = Quaternion(0,0,0,1)
            mpc_traj_msg.poses.append(pose)
            
        self.pub_mpc.publish(mpc_traj_msg)
        pass

    def publish_velocity(self, result):
        #TODO: Velocity
        vel = Twist()
        vel.linear.x = 0.5
        vel.angular.z = 0.1
        self.pub_vel.publish(vel)
        pass

    def run(self):
        # The subscriber is getting the latest data
        # Setup the MPC
        #TODO: Do this
        self.mpc.setup()
        # solve
        result = self.mpc.solve() # Return the optimization variables
        # Control and take only the first step 
        
        self.publish_velocity(result)
        
        # Get from the MPC results
        mpc_x_traj = np.arange(0,2,0.1)
        mpc_y_traj = np.zeros(mpc_x_traj.shape[0])
        theta_traj = np.zeros(mpc_x_traj.shape[0])
        self.publish_trajectory(mpc_x_traj, mpc_y_traj, theta_traj)
        pass
    
    
if __name__ =="__main__":
    rospy.init_node("nmpc")
    rospy.loginfo("Non-Linear MPC Node running")
    node = ROSNode()
    pause = rospy.Rate(10) # 10 Hz
    while not rospy.is_shutdown():
        node.run()
        pause.sleep()