#!/usr/bin/python
import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped, PolygonStamped
from nav_msgs.msg import Path, Odometry
from jackal_helper.msg import ResultData

import numpy as np
from tf.transformations import euler_from_quaternion
from tf_conversions import Quaternion
import csv
import os
import copy

INF_CAP = 10000
class Inspection():
    def __init__(self):
        # Topic Defintions
        self.TOPIC_FRONT_SCAN = "/front/scan" # Front Laser Scan (LIDAR)
        self.TOPIC_ODOM = "/odometry/filtered" # Odometry (pose and twist)
        self.TOPIC_CMD_VEL = "/cmd_vel" # Command Velocity (action)

        self.TOPIC_LOCAL_PLAN = "/move_base/TrajectoryPlannerROS/local_plan" # Local plan
        self.TOPIC_LOCAL_FOOTPRINT = "/move_base/local_costmap/footprint" # the bounding box of the robot
        self.TOPIC_GLOBAL_PLAN = "/move_base/TrajectoryPlannerROS/global_plan" # Global plan
        self.TOPIC_MPC = "/mpc_plan" # MPC plan
        self.RESULT_DATA = "/result_data" # Result Data
        
        self.data = []
        self.data_dict = {}
        self.look_ahead = 0.325

        # Object to store
        self.scan = LaserScan()
        self.cmd_vel = Twist()
        self.global_plan = Path()
        self.local_plan = Path()
        self.odometry = Odometry()
        self.footprint = PolygonStamped() 
        
        # init CSV File
        print("Write to CSV file")
        file_path = "/jackal_ws/src/mlda-barn-2024/imit_data_2.csv"
        self.metadata_rows = ["success", "actual_time", "optimal_time", "world_idx", "timestep"]
        self.lidar_rows = ["lidar_" + str(i) for i in range(720)]
        self.odometry_rows = ['pos_x', 'pos_y', 'pose_heading', 'twist_linear', 'twist_angular']
        self.action_rows = ['cmd_vel_linear', 'cmd_vel_angular']
        self.goal_rows = ['local_goal_x', 'local_goal_y']
        self.data_rows = self.lidar_rows + self.odometry_rows + self.action_rows + self.goal_rows
        self.fieldnames = self.metadata_rows + self.data_rows

        # Subscribe        
        self.sub_front_scan = rospy.Subscriber(self.TOPIC_FRONT_SCAN, LaserScan, self.callback_front_scan)
        self.sub_odometry = rospy.Subscriber(self.TOPIC_ODOM, Odometry, self.callback_odometry)
        self.sub_global_plan = rospy.Subscriber(self.TOPIC_GLOBAL_PLAN, Path, self.callback_global_plan)
        self.sub_local_plan = rospy.Subscriber(self.TOPIC_MPC, Path, self.callback_local_plan)
        self.sub_footprint = rospy.Subscriber(self.TOPIC_LOCAL_FOOTPRINT, PolygonStamped, self.callback_footprint)
        self.sub_cmd_vel = rospy.Subscriber(self.TOPIC_CMD_VEL, Twist, self.callback_cmd_vel)
        self.sub_result = rospy.Subscriber(self.RESULT_DATA, ResultData, self.callback_result_data)
        
        file_exist = False
        if os.path.exists(file_path):
            file_exist = True
        
        self.csv_file = open(file_path, 'a')
        self.writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
        if not file_exist:
            self.writer.writeheader()

    def compute_local_goal(self, pos_x, pos_y):
        # check which is shortest distance to pos
        min_dist = INF_CAP
        local_goal_x = 0
        local_goal_y = 0
        global_plan = self.global_plan
        for i in range(len(global_plan.poses)):
            global_x = global_plan.poses[i].pose.position.x
            global_y = global_plan.poses[i].pose.position.y
            dist = np.sqrt((pos_x - global_x)**2 + (pos_y - global_y)**2)
            if dist < min_dist and dist > self.look_ahead:
                min_dist = dist
                local_goal_x = global_x
                local_goal_y = global_y
        return local_goal_x, local_goal_y

    def update_row(self):
        # check if global_plan has attribute poses
        if len(self.data_dict) >= len(self.data_rows) - len(self.goal_rows) and len(self.global_plan.poses) > 0:
            # check if data_dict is NaN
            data_dict = copy.deepcopy(self.data_dict)
            self.data_dict = {}
            for key in data_dict.keys():
                if np.isnan(data_dict[key]):
                    return
                
            local_goal_x, local_goal_y = self.compute_local_goal(data_dict["pos_x"], data_dict["pos_y"])
            data_dict["local_goal_x"] = local_goal_x
            data_dict["local_goal_y"] = local_goal_y
            self.data.append(data_dict)
            

    def callback_result_data(self, metadata):
        print("---- Processing Result Data ----")
        print("Length of data: ", len(self.data))
        print("Metadata: ", metadata)

        data = copy.deepcopy(self.data)
        self.data = []
        for i in range(len(data)):
            data[i]["world_idx"] = metadata.world_idx
            data[i]["success"] = metadata.success
            data[i]["actual_time"] = metadata.actual_time
            data[i]["optimal_time"] = metadata.optimal_time
            data[i]["timestep"] = i

        # if metadata.success:
        #     self.writer.writerows(data)
        self.writer.writerows(data)

    def callback_front_scan(self, data):
        self.scan = data
        if 1:
            # print("Scan points: ", len(data.ranges), "From Max: ", data.range_max, "| Min: ", round(data.range_min,2))
            # print("Angle from: ", np.degrees(data.angle_min).round(2), " to: ", np.degrees(data.angle_max).round(2), " increment: ", np.degrees(data.angle_increment).round(3))
           
            # update the data_dict
            assert(len(data.ranges) == 720)
            for i in range(720):
                if data.ranges[i] > data.range_max:
                    self.data_dict["lidar_" + str(i)] = 10
                else:
                    self.data_dict["lidar_" + str(i)] = data.ranges[i]
            self.update_row()
    
    def calc_local(self, x, y, gx, gy, heading):
        local_x = (gx - x) * np.cos(heading) + (gy - y) * np.sin(heading)
        local_y = -(gx - x) * np.sin(heading) + (gy - y) * np.cos(heading)
        return local_x, local_y

    def callback_odometry(self, data):
        if 1:
            self.odometry = data
            # print("==========================")
            # print("----------------------- pose.position")
            # print(data.pose.pose.position)
            # print("----------------------- pose.orientation")
            # print(data.pose.pose.orientation)
            q = Quaternion()
            q.x = data.pose.pose.orientation.x
            q.y = data.pose.pose.orientation.y
            q.z = data.pose.pose.orientation.z
            q.w = data.pose.pose.orientation.w
            # print("----------------------- pose.heading")
            heading_rad = np.array(euler_from_quaternion([q.x, q.y, q.z, q.w])[2])
            heading_deg = np.degrees(heading_rad)
            # print("Rad: " + str(heading_rad.round(3)))
            # print("Degree: " + str( heading_deg.round(3)))

            # print("----------------------- twist.linear")
            # print(data.twist.twist.linear)
            # print("----------------------- twist.angular")
            # print(data.twist.twist.angular)

            # update the data_dict
            self.data_dict["pos_x"] = data.pose.pose.position.x
            self.data_dict["pos_y"] = data.pose.pose.position.y
            self.data_dict["pose_heading"] = heading_rad
            self.data_dict["twist_linear"] = data.twist.twist.linear.x
            self.data_dict["twist_angular"] = data.twist.twist.angular.z

            # local_x, local_y = self.calc_local(data.pose.pose.position.x, data.pose.pose.position.y, 0, 10, heading_rad)
            # print("Local x: ", round(local_x,3), "; Local y: ", round(local_y,3))

            self.update_row()
    
    def callback_cmd_vel(self, data):
        if 1:
            if -0.001 < data.linear.x < 0.001 and -0.001 < data.angular.z < 0.001:
                return
            print("Linear: ", round(data.linear.x,3), "; Angular: ", round(data.angular.z,3))            
            # update the data_dict
            self.data_dict["cmd_vel_linear"] = data.linear.x
            self.data_dict["cmd_vel_angular"] = data.angular.z
            self.update_row()

    def callback_footprint(self, data):
        self.footprint = data
        if 0: 
            points_array = []
            for point in data.polygon.points:
                points_array.append([point.x, point.y,point.z])
            np_array = np.array(points_array)
            print("Number of points on the Polygon: ", len(data.polygon.points))
            print("Points: ", np.round(np_array,3))

    def callback_global_plan(self, data):
        self.global_plan = data
        if 0: 
            print("Global Path points ", len(data.poses))
            print(data.poses[3])
            print("Local Path points ", len(self.local_plan.poses))
    
    def callback_local_plan(self, data):
        self.local_plan = data
        if 0:
            print("Local Path points ", len(data.poses))
            print(data.poses[3])
            print("Global Path points ", len(self.global_plan.poses))



if __name__ == "__main__":
    rospy.init_node('inspection_node')
    rospy.loginfo("Inspection Node Started")
    inspect = Inspection()
    rospy.spin()
    