import casadi as ca
import rospy
import numpy as np
import time

class NMPC:
    def __init__(self, freq = 20, N = 20):
        self.f = freq # Controller frequency [Hz]
        self.h = 1/self.f
        # self.rate = rospy.Rate(self.f)
        
        self.L = 0.37558 # Distance between 2 wheels
        # self.L = rospy.get_param("/mlda/L")
        # Using rosparam
        # For each wheels
        self.v_max = 1 # Max velocity [m/s]
        self.v_min = 0.1 # Min velocity [m/s]
        
        self.a_max = 1 # Max acceleration [m/s^2]

        self.w_max = 5 # Max angular vel [rad/s]
        self.w_min = -5 # Max angular vel [rad/s]
        
        
        self.a_weights = 0.2

        self.N = N
        
        
        self.opt_states = None
        # Obstacle avoidance
        pass
    
    def setup(self, rate):
        self.h = 1/rate
        # --- State and control variables ---
        # Variables
        # X = [x0   y0  theta0  v   w   a, (...), xN, yN, thetaN, vN, wN, aN]
        # X = [0    1   2       3   4   5, ...]
        self.n = 6 # n is "Degree of freedom" in 1 instance. # N is the number of time step
        
        self.X = ca.MX.sym('X',self.N*self.n)
        
        # Bounds on variables
        lbx = [-np.inf,-np.inf,-np.inf,-self.v_max,-self.w_max,-self.a_max]*self.N
        self.lbx = np.array(lbx)
        ubx = [np.inf,np.inf,np.inf,self.v_max, self.w_max,self.a_max]*self.N
        self.ubx = np.array(ubx)

        # --- Constraints ---
        # Vehicle dynamics constraints
        # Position (x, y, theta)
        gx = self.X[0::self.n][1:] - self.X[0::self.n][:-1] - self.h*(self.X[3::self.n][:-1]*np.cos(self.X[2::self.n][:-1]))
        gy = self.X[1::self.n][1:] - self.X[1::self.n][:-1] - self.h*(self.X[3::self.n][:-1]*np.sin(self.X[2::self.n][:-1]))
        gtheta = self.X[2::self.n][1:] - self.X[2::self.n][:-1] * self.h * self.X[4::self.n][:-1] # Theta and angular velocity
        gv = self.X[3::self.n][1:] - self.X[3::self.n][:-1] - self.h*(self.X[5::self.n][:-1])
                
        # Positive linear velocity
        gv_min = (self.X[3::self.n]) - self.v_min
        gv_max = self.v_max  - (self.X[3::self.n])
        # Minimum angular velocity
        gw_min = (self.X[4::self.n]) - self.w_min
        # Maximum angular velocity
        gw_max = self.w_max - (self.X[4::self.n])


        ### Each of the term above has N-1 columns   
        self.g = ca.vertcat(gx, gy, gtheta, gv,  gv_min, gv_max, gw_min, gw_max)
    

        ### This is not dynamic contraints, because they only relate the existing optimization variables
    
    def solve(self, x_ref, y_ref, theta_ref, X0):
        start_time = time.time()

        
        # === Set constraints bound
        per_step_constraints = 8
        init_value_constraints = 5
        final_value_contraints = 1
        self.lbg = np.zeros((self.N - 1) * per_step_constraints + init_value_constraints + final_value_contraints)
        self.ubg = np.zeros((self.N - 1) * per_step_constraints + init_value_constraints + final_value_contraints)
        # Set inequality bound
        self.ubg[(-4*(self.N - 1)-init_value_constraints-final_value_contraints):(-init_value_constraints -final_value_contraints)] = np.inf # Velocity positive
        
        ## All constraints g
        self.g = self.g[:(self.N - 1) * per_step_constraints]  # Since we are recycling the same instance
        # We have to truncate the previous optimization run
        
        # Init-value constraints expression
        for i in range(init_value_constraints):
            g0 = self.X[i::self.n][0] - X0[i]
            self.g = ca.vertcat(self.g, g0)
        
        
        # Final constraints
        gfx = self.X[0::self.n][self.N-1] - x_ref[self.N-1]
        # gfy = self.X[1::self.n][self.N-1] - y_ref[self.N-1]
        # gftheta = self.X[2::self.n][self.N-1] - theta_ref[self.N-1]
        self.g = ca.vertcat(self.g, gfx)
        
        # print("Constraints: ", self.g.shape)
        # print("lbg: ", self.lbg.shape)
        # print("ubg: ", self.ubg.shape)
        # --- Cost function --- 

        J = 0
        self.v_weights = 1
        self.weight_position_error = 10
        self.weight_theta_error = 1
        self.weight_acceleration = 1
        self.v_ref = 0.2
        for i in range(self.N):
            # Position Error cost
            position_error_cost = (self.X[0::self.n][i] - x_ref[i])**2 + (self.X[1::self.n][i] - y_ref[i])**2
            
            # Theta Error cost 
            theta_error_cost = (self.X[2::self.n][i] - theta_ref[i])**2
            # theta_error_cost = 0

            # Reference velocity cost
            # reference_velocity_cost = (self.X[3::self.n][i] - self.v_ref)**2 + (self.X[4::self.n][i] - self.v_ref)**2
            reference_velocity_cost = 0
            # Cost function calculation
            J += (self.weight_position_error*position_error_cost + self.v_weights*reference_velocity_cost + self.weight_theta_error*theta_error_cost)
        
        
        # === Initial guess
        if self.opt_states == None:
            init_guess = 0
        else:
            init_guess = ca.vertcat(self.opt_states[self.n:], self.opt_states[-self.n:])
            # init_guess = 0
        
        # === Solution
        options = {'ipopt.print_level':1, 'print_time': 0, 'expand': 1} # Dictionary of the options
        # options={}
        problem = {'f': J , 'x': self.X, 'g': self.g} # Dictionary of the problem
        solver = ca.nlpsol('solver', 'ipopt', problem,options) #ipopt: interior point method
        
        solution = solver(x0 = init_guess, lbx = self.lbx, ubx = self.ubx, lbg = self.lbg, ubg = self.ubg)
        
        # === Optimal control 
        v_opt = solution['x'][3:self.n][1]
        w_opt = solution['x'][4:self.n][1]
        
        # Intial guess for next steps
        self.opt_states = solution['x']
        solve_time = round(time.time() - start_time,5)
        
        
        
        print("============Debugging=======")
        # print("Initial state: ", X0[0], " ", X0[1])
        # print("solution at t = 0: ", self.opt_states[0], " ", self.opt_states[1])
        print("V: ", v_opt, " W: ", w_opt)
        print("Cost: ", solution['f'])
        
        # print(solution)
            
            
        return v_opt, w_opt, solve_time
        