from robosuite.controllers.base_controller import Controller
import robosuite.utils.transform_utils as T

import numpy as np

class HybridMotionForceController(Controller):


    def __init__(self,
                 sim,
                 eef_name,
                 joint_indexes,
                 actuator_range,
                 input_max=1,
                 input_min=-1,
                 output_max=(0.05, 0.05, 0.05, 0.5, 0.5, 0.5),
                 output_min=(-0.05, -0.05, -0.05, -0.5, -0.5, -0.5),
                 policy_freq=20,
                 **kwargs # does nothing; used so no error raised when dict is passed with extra terms used previously
                 ):

        super().__init__(
            sim,
            eef_name,
            joint_indexes,
            actuator_range,
        )

        # Control dimension
        self.control_dim = 0    # action space dimension.

        # input and output max and min (allow for either explicit lists or single numbers)
        self.input_max = self.nums2array(input_max, self.control_dim)
        self.input_min = self.nums2array(input_min, self.control_dim)
        self.output_max = self.nums2array(output_max, self.control_dim)
        self.output_min = self.nums2array(output_min, self.control_dim)

        # control frequency
        self.control_freq = policy_freq

        # subspace
        self.S_f = np.array([[0, 0, 1, 0, 0, 0]]).reshape([6,1])     # force-control-subspace (only doing force control in z)

        self.S_v = np.array([[1, 0, 0, 0, 0],                        # motion-control-subspace (x, y, ori_x, ori_y, ori_z)
                        [0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 1]])                            

        # stiffness of the interaction [should be estimated (this one is chosen at random)]
        self.K = np.array([[1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 100, 0, 0, 0],
                    [0, 0, 0, 5, 0, 0],
                    [0, 0, 0, 0, 5, 0],
                    [0, 0, 0, 0, 0, 1]])

        self.C = np.linalg.inv(self.K)

        # inverse subspaces
        self.S_v_inv = self.get_S_inv(self.S_v)
        self.S_f_inv = self.get_S_inv(self.S_f)

        # derivative of stiffnes interaction
        self.K_dot = self.get_K_dot()

        # force control dynamics
        self.K_Plambda = 90                       # force gain
        self.K_Dlambda = 2*np.sqrt(self.K_Plambda)#self.K_Plambda*0.001     # force damping

        # position control dynamics
        self.Pp = 150                         # x and y pos gain 
        self.Dp = 2*np.sqrt(self.Pp)          # x and y pos damping

        # orientation control dynamics
        self.Po = 100                                   # orientation gain
        self.Do = 2*np.sqrt(self.Po)                   # orientation damping

        self.K_Pr = np.array([[self.Pp, 0, 0, 0, 0],    # Stiffness matrix
                              [0, self.Pp, 0, 0, 0],
                              [0, 0, self.Po, 0, 0],
                              [0, 0, 0, self.Po, 0],
                              [0, 0, 0, 0, self.Po]])

        self.K_Dr = np.array([[self.Dp, 0, 0, 0, 0],    # Damping matrix
                              [0, self.Dp, 0, 0, 0],
                              [0, 0, self.Do, 0, 0],
                              [0, 0, 0, self.Do, 0],
                              [0, 0, 0, 0, self.Do]])

        # initialize robot
        self.robot = None
        self.probe_id = None

        # initialize desired trajectories  
        self.p_d = np.zeros(2)              # position trajectory
        self.r_d_dot = np.zeros(5) 
        self.r_d_ddot = np.zeros(5)         

        self.f_d = 5                        # force trajectory (N)
        self.f_d_dot = 0
        self.f_d_ddot = 0 

        self.goal_ori = np.array([-0.69192486,  0.72186726, -0.00514253, -0.01100909])  # (x, y, z, w) quaternion

        # initialize trajectory point from environment
        self.traj_pos = None                # will be sat from the enviromnent

        # initialize measurements
        self.z_force = 0                    # force in z-direction
        self.v = np.zeros(5)                # angular and linear (excluding z) velocity


    def _initialize_measurements(self):
        self.probe_id = self.sim.model.body_name2id(self.robot.gripper.root_body)
        self.z_force = self.robot.ee_force[-1]
        #self.z_force = self.sim.data.cfrc_ext[self.probe_id][-1]
        self.v = self.get_eef_velocity()


    # Must be called in environment's reset function
    def set_robot(self, robot):
        self.robot = robot
        self._initialize_measurements()


    def quatdiff_in_euler_radians(self, quat_curr, quat_des):
        quat_dist = T.quat_distance(quat_curr, quat_des)
        return -T.mat2euler(T.quat2mat(quat_dist))


    # Fetch linear (excluding z) and angular velocity of eef
    def get_eef_velocity(self):
        lin_v = self.robot._hand_vel[:-1]
        ang_v = self.robot._hand_ang_vel

        return np.append(lin_v, ang_v)


    # Fetch the derivative of the force as in equation (9.66) in chapter 9.4 of The Handbook of Robotics
    def get_lambda_dot(self):
        return np.linalg.multi_dot([self.S_f_inv, self.K_dot, self.J_full, self.joint_vel])


    # Fetch the psudoinverse of S_f or S_v as in equation (9.34) in chapter 9.3 of The Handbook of Robotics
    def get_S_inv(self, S):
        a = np.linalg.inv(np.linalg.multi_dot([S.T, self.C, S]))
        return np.array(np.linalg.multi_dot([a, S.T, self.C]))


    # Fetch K' as in equation (9.49) in chapter 9.3 of The Handbook of Robotics
    def get_K_dot(self):
        return np.linalg.multi_dot([self.S_f, self.S_f_inv, np.linalg.inv(self.C)])


    # Calculate the error in position and orientation (in the subspace subject to motion control)
    def get_delta_r(self, ori, goal_ori, p, p_d):
        delta_pos = p_d - p[:2]
        delta_ori = self.quatdiff_in_euler_radians(ori, goal_ori)   

        return np.append(delta_pos, delta_ori)


    # Calculate f_lambda (part of equation 9.62) as in equation (9.65) in chapter 9.3 of The Handbook of Robotics
    def calculate_f_lambda(self, f_d_ddot, f_d_dot, f_d):
        lambda_dot = self.get_lambda_dot()    
        
        lambda_a = f_d_ddot 
        lambda_b = np.dot(self.K_Dlambda,(f_d_dot - lambda_dot))
        lambda_c = np.dot(self.K_Plambda,(f_d - self.z_force))
 
        return max(lambda_a + lambda_b + lambda_c, 0)


    # Calculate alpha_v (part of equation 9.62) as on page 213 in chapter 9.3 of The Handbook of Robotics
    def calculate_alpha_v(self, ori, r_d_ddot, r_d_dot, p, p_d):
        delta_r = self.get_delta_r(ori, self.goal_ori, p, p_d)
        return r_d_ddot + np.dot(self.K_Dr, r_d_dot - self.v) + np.dot(self.K_Pr, delta_r)


    # Calculate alpha (part of equation 9.16) as in equation (9.62) in chapter 9.3 of The Handbook of Robotics
    def calculate_alpha(self, alpha_v,f_lambda):
        P_v = np.dot(self.S_v, self.S_v_inv)
        C_dot = np.dot(np.identity(6) - P_v, self.C)

        return np.dot(self.S_v, alpha_v) + f_lambda * np.dot(C_dot, self.S_f).flatten()


    def set_goal(self, action):

        timestep = 1 / self.control_freq        # should be self.model_timestep instead?

        # update position trajectory
        prev_p_d = self.p_d
        self.p_d = self.traj_pos[:-1]

        prev_r_d_dot = self.r_d_dot
        p_dot = np.subtract(self.p_d, prev_p_d) / timestep
        ori_dot = np.array([0, 0, 0])
        self.r_d_dot = np.append(p_dot, ori_dot)

        self.r_d_ddot = np.subtract(self.r_d_dot, prev_r_d_dot) / timestep

        # update force trajectory
        self.f_d = self.f_d     # constant
        self.f_d_dot = 0
        self.f_d_ddot = 0                   
                 

    def run_controller(self):
        
        # Update state
        self.update()

        # eef measurements
        self.z_force = self.robot.ee_force[-1] # self.sim.data.cfrc_ext[self.probe_id][-1]
        self.v = self.get_eef_velocity()
        
        pos = self.ee_pos
        ori = T.mat2quat(self.ee_ori_mat)   # (x, y, z, w) quaternion

        h_e = np.array([0, 0, self.z_force, 0, 0, 0])

        # control law
        alpha_v = self.calculate_alpha_v(ori, self.r_d_ddot, self.r_d_dot, pos, self.p_d) 
        f_lambda = self.calculate_f_lambda(self.f_d_ddot, self.f_d_dot, self.f_d)
        alpha = self.calculate_alpha(alpha_v, -f_lambda)

        cartesian_inertia = np.linalg.inv(np.linalg.multi_dot([self.J_full, np.linalg.inv(self.mass_matrix), self.J_full.T]))

        # torque computations
        external_torque = np.dot(self.J_full.T, h_e)
        torque = np.linalg.multi_dot([self.J_full.T ,cartesian_inertia, alpha]) + external_torque  # NOTE MODIFIED FROM DEFAULT. Removed coriolis force

        # Always run superclass call for any cleanups at the end
        super().run_controller()

        return torque


    def update_initial_joints(self, initial_joints):
        # First, update from the superclass method
        super().update_initial_joints(initial_joints)

        # We also need to reset the goal in case the old goals were set to the initial configuration
        self.reset_goal()


    def reset_goal(self):
        """
        Resets the goal to the current state of the robot
        """
        self.p_d = np.array(self.ee_pos)[:-1]


    @property
    def control_limits(self):
        """
        Returns the limits over this controller's action space, overrides the superclass property

            2-tuple:
                - (np.array) minimum action values
                - (np.array) maximum action values
        """
        low, high = self.input_min, self.input_max
        return low, high


    @property
    def name(self):
        return "HMFC"
