import casadi as ca
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
        self.v_min = 0.02
        self.v_max = 0.8 # Max velocity [m/s]
        self.a_max = 0.5 # Max acceleration [m/s^2]

        self.w_max = 0.01 # Max angular vel [rad/s]
        self.w_min = 0.8 # Max angular vel [rad/s]
        
        # Obstacle avoidance
        pass
    
    def setup(self):
                # --- State and control variables ---
        # Variables
        # X = [x0, y0, theta0, vr0, vl0, ar0, al0, (...), xN, yN, thetaN, vrN, vlN, arN, alN]
        self.n = 7 # n is "Degree of freedom" in 1 instance. # N is the number of time step
        self.X = ca.MX.sym('X',self.N*self.n)
        # Bounds on variables
        lbx = [-np.inf,-np.inf,-np.inf,-self.v_max,-self.v_max,-self.a_max,-self.a_max]*self.N
        self.lbx = np.array(lbx)
        ubx = [np.inf,np.inf,np.inf,self.v_max,self.v_max,self.a_max,self.a_max]*self.N
        self.ubx = np.array(ubx)


        # --- Constraints ---
        # Vehicle dynamics constraints
        # Position (x, y, theta)
        gx = self.X[0::self.n][1:] - self.X[0::self.n][:-1] - 0.5*self.h*(((self.X[3::self.n][1:] + self.X[4::self.n][1:])/2)*np.cos(self.X[2::self.n][1:]) + ((self.X[3::self.n][:-1] + self.X[4::self.n][:-1])/2)*np.cos(self.X[2::self.n][:-1]))
        gy = self.X[1::self.n][1:] - self.X[1::self.n][:-1] - 0.5*self.h*(((self.X[3::self.n][1:] + self.X[4::self.n][1:])/2)*np.sin(self.X[2::self.n][1:]) + ((self.X[3::self.n][:-1] + self.X[4::self.n][:-1])/2)*np.sin(self.X[2::self.n][:-1]))
        gtheta = self.X[2::self.n][1:] - self.X[2::self.n][:-1] - 0.5*self.h*(((self.X[3::self.n][1:] - self.X[4::self.n][1:])/self.L) + ((self.X[3::self.n][:-1] - self.X[4::self.n][:-1])/self.L))
        # Velocity (v_r, v_l)
        gv_r = self.X[3::self.n][1:] - self.X[3::self.n][:-1] - 0.5*self.h*(self.X[5::self.n][1:] + self.X[5::self.n][:-1])
        gv_l = self.X[4::self.n][1:] - self.X[4::self.n][:-1] - 0.5*self.h*(self.X[6::self.n][1:] + self.X[6::self.n][:-1])
        # Positive linear velocity
        gv = self.X[3::self.n][:-1] + self.X[4::self.n][:-1] - self.v_min
        # Minimum angular velocity
        gw_min = (((self.X[3::self.n][:-1] - self.X[4::self.n][:-1])/self.L)*((self.X[3::self.n][:-1] - self.X[4::self.n][:-1])/self.L)) - self.w_min
        # Maximum angular velocity
        gw_max = self.w_max - (((self.X[3::self.n][:-1] - self.X[4::self.n][:-1])/self.L)*((self.X[3::self.n][:-1] - self.X[4::self.n][:-1])/self.L))


        ### Each of the term above has N-1 columns
        self.g = ca.vertcat(gx, gy, gtheta, gv_r, gv_l, gv, gw_min, gw_max)

        ### This is not dynamic contraints, because they only relate the existing optimization variables
        pass
    
    def solve(self, init_values, traj):
        
        
        
        pass