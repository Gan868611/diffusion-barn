import casadi
import rospy
import numpy as np


class NMPC:
    def __init__(self, freq = 5):
        self.f = freq # Controller frequency [Hz]
        self.h = 1/self.f
        # self.rate = rospy.Rate(self.f)
        
        self.L = 0.37558 # Distance between 2 wheels
        # self.L = rospy.get_param("/mlda/L")
        # Using rosparam
        self.v_max = 0.8 # Max velocity [m/s]
        self.a_max = 0.5 # Max acceleration [m/s^2]
        self.w_max = 0.5 # Max angular vel [rad/s]
        
        
        # Obstacle avoidance
        pass
    
    def setup(self):
        pass
    
    def solve(self):
        
        
        
        pass