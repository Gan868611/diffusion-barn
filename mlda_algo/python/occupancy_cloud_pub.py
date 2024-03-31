#!/usr/bin/python
from operator import index
import rospy
import sensor_msgs.point_cloud2 as pc2
import tf.transformations as tft
from sensor_msgs.msg import PointCloud2, LaserScan, PointField
from std_msgs.msg import Header
from nav_msgs.msg import Path, Odometry, OccupancyGrid, MapMetaData
import laser_geometry.laser_geometry as lg
import math
import numpy as np
import tf

class OccupancyToCloud():
    def __init__(self):

        self.TOPIC_LOCAL_MAP = "/move_base/local_costmap/costmap"
        self.TOPIC_MAP_CLOUD = "/map/cloud"
        self.map = OccupancyGrid()
        self.map_grid = None
        self.point_cloud = PointCloud2()

        self.sub_map = rospy.Subscriber(self.TOPIC_LOCAL_MAP, OccupancyGrid, self.callback_map)
        self.pub_point = rospy.Publisher(self.TOPIC_MAP_CLOUD, PointCloud2, queue_size=1)
    

    def callback_map(self, data):
        self.map = data
        map_data = np.array(self.map.data)
        map_Ox = self.map.info.origin.position.x
        map_Oy = self.map.info.origin.position.y
        print(self.map.info.origin)

        print("Occupied: ", np.count_nonzero(map_data >= 90))
        print("Free: ", np.count_nonzero(map_data < 90))
        # print("Total: ", data.data.count(100) + data.data.count(0) + data.data.count(-1))
        # print(map_data)


        print("Map Origin: ", map_Ox, map_Oy, " wrt ", self.map.header.frame_id)
        self.map_res = self.map.info.resolution
        map_width = self.map.info.width
        map_height = self.map.info.height
        print("Map: ", self.map.info.width, self.map.info.height)
        print("Map res: ", self.map.info.resolution)
        print(map_data.shape)
        self.map_grid = np.reshape(map_data, (map_width, map_height))
        self.map_origin = np.array([map_Ox, map_Oy])






    def run(self, trans, rot):
        pointcloud = PointCloud2()

        rot_matrix = tft.quaternion_matrix(rot)
        trans_matrix = tft.translation_matrix(trans)
        # print(rot_matrix.shape, trans_matrix.shape)
        final_matrix = np.matmul(trans_matrix, rot_matrix)
        # print(final_matrix)

        x_map_to_chassis = trans[0]
        y_map_to_chassis = trans[1]
        # print("Map to Chassis: ", x_map_to_chassis, y_map_to_chassis)

        box_distance = 2 #m

        top_left = np.array([x_map_to_chassis - box_distance/2, y_map_to_chassis - box_distance/2])
        top_right = np.array([x_map_to_chassis - box_distance/2, y_map_to_chassis + box_distance/2])
        bot_left = np.array([x_map_to_chassis + box_distance/2, y_map_to_chassis - box_distance/2])
        bot_right = np.array([x_map_to_chassis + box_distance/2, y_map_to_chassis + box_distance/2])

        index_top_left = np.array([int(top_left[0]/self.map_res), int((top_left[1])/self.map_res)])
        # index_top_right = np.array([int(top_right[0]/self.map_res), int((top_right[1])/self.map_res)])
        # index_bot_left = np.array([int(bot_left[0]/self.map_res), int((bot_left[1])/self.map_res)])
        index_bot_right = np.array([int(bot_right[0]/self.map_res), int((bot_right[1])/self.map_res)])
        # print("Index Top Left: ", index_top_left)
        # print("Index Top Right: ", index_top_right)
        # print("Index Bot Left: ", index_bot_left)
        # print("Index Bot Right: ", index_bot_right)

        box_grid = self.map_grid[index_top_left[0]:index_bot_right[0], index_top_left[1]:index_bot_right[1]]
        # print("Shape: ", box_grid.shape)




        header = Header()
        header.frame_id = "/odom"
        # header.frame_id = "/front_laser"

        header.stamp = rospy.Time.now()

        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1), 
                  PointField('intensity', 12, PointField.FLOAT32, 1)]
        points = []

        # Origin
        points.append([self.map_origin[0], self.map_origin[1], 0, 1])
        points.append([top_left[0], top_left[1], 0, 1])
        # points.append([index_bot_right[0]*self.map_res, index_bot_right[1]*self.map_res, 0, 1])

        # for i in range(box_grid.shape[0]):
        #     for j in range(box_grid.shape[1]):
        #         if box_grid[j,i] == 100:  # Convert to Odom
        #             x = i*self.map_res
        #             y = j*self.map_res
        #             points.append([x, y, 0, 1])

        pointcloud = pc2.create_cloud(header, fields, points)
        self.pub_point.publish(pointcloud)
        # print(len(points))

if __name__ == "__main__":
    rospy.init_node("map_to_cloud")
    rospy.loginfo("Map to Cloud")
    l = OccupancyToCloud()
    pause = rospy.Rate(20)

    listener = tf.TransformListener()

    while not rospy.is_shutdown():
        try:
            (trans,rot) = listener.lookupTransform('/odom', '/base_link', rospy.Time(0))
            l.run(trans,rot)
        except Exception as e: 
            # print(e)
            continue
        pause.sleep()