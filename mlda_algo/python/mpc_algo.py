import casadi as ca
import rospy
import numpy as np
import time

class NMPC:
    def __init__(self, freq = 5):
        self.f = freq # Controller frequency [Hz]
        self.h = 1/self.f
        # self.rate = rospy.Rate(self.f)
        
        self.L = 0.37558 # Distance between 2 wheels
        # self.L = rospy.get_param("/mlda/L")
        # Using rosparam
        self.v_min = 0.1
        self.v_max = 0.8 # Max velocity [m/s]
        self.a_max = 0.5 # Max acceleration [m/s^2]

        self.w_max = 0.01 # Max angular vel [rad/s]
        self.w_min = 0.8 # Max angular vel [rad/s]
        
        
        self.a_weights = 0.2

        
        self.N = 20
        
        
        self.opt_states = None
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
    
    def solve(self, x_ref, y_ref, X0):
        start_time = time.time()

        
        # === Set constraints bound
        per_step_constraints = 8
        init_value_constraints = 5
        self.lbg = np.zeros((self.N - 1) * per_step_constraints + init_value_constraints)
        self.ubg = np.zeros((self.N - 1) * per_step_constraints + init_value_constraints)
        
        # Set inequality bound
        self.ubg[(-2*(self.N - 1)):] = np.inf # Velocity positive
        
        ## All constraints g
        self.g = self.g[:(self.N - 1) * per_step_constraints]  # Since we are recycling the same instance
        # We have to truncate the previous optimization run
        
        # Init-value constraints expression
        for i in range(5):
            g0 = self.X[i::self.n][i] - X0[i]
            self.g = ca.vertcat(self.g, g0)
    
        # --- Cost function --- 

        J = 0
        for i in range(self.N):
            # Reference following error cost
            ref_follow_error = ((self.X[0::self.n][i] - x_ref[i])*(self.X[0::self.n][i] - x_ref[i])) + ((self.X[1::self.n][i] - y_ref[i])*(self.X[1::self.n][i] - y_ref[i]))
            
            # Successive control cost
            if i != (self.N-1):
                successive_error = ((self.X[5::self.n][i+1]-self.X[5::self.n][i])*(self.X[5::self.n][i+1]-self.X[5::self.n][i]))+((self.X[6::self.n][i+1]-self.X[6::self.n][i])*(self.X[6::self.n][i+1]-self.X[6::self.n][i]))

            # Cost function calculation
            J += (ref_follow_error + self.a_weights*successive_error)
        
        
        # === Initial guess
        if self.opt_states == None:
            init_guess = 0
        else:
            init_guess = ca.vertcat(self.opt_states[7:], self.opt_states[-7:])
        
        # === Solution
        options = {'ipopt.print_level':0, 'print_time': 0, 'expand': 1} # Dictionary of the options
        problem = {'f': J , 'x': self.X, 'g': self.g} # Dictionary of the problem
        solver = ca.nlpsol('solver', 'ipopt', problem, options) #ipopt: interior point method
        
        solution = solver(x0 = init_guess, lbx = self.lbx, ubx = self.ubx, lbg = self.lbg, ubg = self.ubg)
        
        # === Optimal control 
        vr_opt = solution['x'][3::self.n][2]
        vl_opt = solution['x'][4::self.n][2]
        
        v_opt = (vr_opt + vl_opt)/2
        w_opt = (vr_opt - vl_opt)/self.L
        
        print("V: ", v_opt, " W: ", w_opt)
        # Intial guess for next steps
        self.opt_states = solution['x']
        solve_time = round(time.time() - start_time,5)
        
        return v_opt, w_opt, solve_time
        