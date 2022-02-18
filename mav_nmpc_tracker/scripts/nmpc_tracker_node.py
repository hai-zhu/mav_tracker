#!/usr/bin/env python

import numpy as np
import rospy
import tf
from nav_msgs.msg import Odometry
from mav_msgs.msg import RollPitchYawrateThrust
from trajectory_msgs.msg import MultiDOFJointTrajectory
from nmpc_tracker_solver import MPC_Formulation_Param
from nmpc_tracker_solver import acados_mpc_solver_generation

g = 9.8066


class Mav_Nmpc_Tracker:
    def __init__(self, mpc_form_param, tracking_mode):
        # MPC formulation settings
        self.mpc_form_param_ = mpc_form_param

        # mav mass, and settins
        self.mass_ = self.mpc_form_param_.mass
        self.tracking_mode_ = tracking_mode  # track, hover, home
        self.odom_time_out_ = 0.2
        self.traj_time_out_ = 0.2

        # state
        self.mav_state_current_ = np.zeros(9)
        self.mav_control_current_ = np.array(4)

        # MPC settings
        self.mpc_dt_ = self.mpc_form_param_.dt
        self.mpc_N_ = self.mpc_form_param_.N
        self.mpc_Tf_ = self.mpc_form_param_.Tf
        self.mpc_nx_ = 9
        self.mpc_nu_ = 3
        self.mpc_ny_ = 9
        self.mpc_ny_e_ = 6

        # MPC variables
        self.mpc_pos_ref_ = np.zeros((3, self.mpc_N_))
        self.mpc_vel_ref_ = np.zeros((3, self.mpc_N_))
        self.mpc_u_ref_ = np.zeros((self.mpc_nu_, self.mpc_N_))
        self.mpc_x_plan_ = np.zeros((self.mpc_nx_, self.mpc_N_))
        self.mpc_u_plan_ = np.zeros((self.mpc_nu_, self.mpc_N_))
        self.mpc_x_next_ = np.zeros(self.mpc_nx_)
        self.mpc_u_now_ = np.zeros(self.mpc_nu_)
        self.mpc_feasible_ = False

        # MPC solver
        self.mpc_solver_ = acados_mpc_solver_generation(self.mpc_form_param_)

        # ROS subscriber
        self.odom_sub_ = rospy.Subscriber("/mav_sim_odom", Odometry, self.set_odom)
        self.received_first_odom_ = False
        self.odom_received_time_ = rospy.Time.now()
        self.traj_sub_ = rospy.Subscriber("/command/trajectory", MultiDOFJointTrajectory, self.set_traj_ref)
        self.traj_received_time_ = rospy.Time.now()
        self.traj_pos_ref_ = np.zeros((3, self.mpc_N_))
        self.traj_vel_ref_ = np.zeros((3, self.mpc_N_))

        # ROS publisher
        self.roll_pitch_yawrate_thrust_pub_ = rospy.Publisher("/mav_roll_pitch_yawrate_thrust_cmd",
                                                              RollPitchYawrateThrust, queue_size=1)

    def set_odom(self, odom_msg):
        if self.received_first_odom_ is False:
            self.received_first_odom_ = True
            rospy.loginfo('First odometry received!')
        # read data
        self.odom_received_time_ = rospy.Time.now()
        px = odom_msg.pose.pose.position.x
        py = odom_msg.pose.pose.position.y
        pz = odom_msg.pose.pose.position.z
        vx = odom_msg.twist.twist.linear.x
        vy = odom_msg.twist.twist.linear.y
        vz = odom_msg.twist.twist.linear.z
        rpy = tf.transformations.euler_from_quaternion([odom_msg.pose.pose.orientation.x,
                                                        odom_msg.pose.pose.orientation.y,
                                                        odom_msg.pose.pose.orientation.z,
                                                        odom_msg.pose.pose.orientation.w])
        self.mav_state_current_ = np.array([px, py, pz, vx, vy, vz, rpy[0], rpy[1], rpy[2]])

    def set_traj_ref(self, traj_msg):
        self.traj_received_time_ = rospy.Time.now()
        for iStage in range(0, self.mpc_N_):
            self.traj_pos_ref_[0, iStage] = traj_msg.points[iStage].transforms[0].translation.x
            self.traj_pos_ref_[1, iStage] = traj_msg.points[iStage].transforms[0].translation.y
            self.traj_pos_ref_[2, iStage] = traj_msg.points[iStage].transforms[0].translation.z
            self.traj_vel_ref_[0, iStage] = traj_msg.points[iStage].velocities[0].linear.x
            self.traj_vel_ref_[1, iStage] = traj_msg.points[iStage].velocities[0].linear.y
            self.traj_vel_ref_[2, iStage] = traj_msg.points[iStage].velocities[0].linear.z

    def set_mpc_ref(self, mode):
        if mode == 'track':  # trajectory tracking
            self.mpc_pos_ref_ = self.traj_pos_ref_
            self.mpc_vel_ref_ = self.traj_vel_ref_
        elif mode == 'hover':  # hovering
            self.mpc_pos_ref_ = np.tile(self.mav_state_current_[0:3].reshape((-1, 1)), (1, self.mpc_N_))
            self.mpc_vel_ref_ = np.tile(self.mav_state_current_[3:6].reshape((-1, 1)), (1, self.mpc_N_))
        elif mode == 'home':  # flying to origin
            self.mpc_pos_ref_ = np.tile(np.array([0.0, 0.0, 1.0]).reshape((-1, 1)), (1, self.mpc_N_))
            self.mpc_vel_ref_ = np.tile(np.array([0.0, 0.0, 0.0]).reshape((-1, 1)), (1, self.mpc_N_))
        else:
            rospy.logwarn('Tracking mode is not correctly set!')
        self.mpc_u_ref_ = np.tile(np.array([0.0, 0.0, g * self.mass_]).reshape((-1, 1)), (1, self.mpc_N_))

    def reset_acados_solver(self):
        # initial condition
        self.mpc_solver_.constraints_set(0, 'lbx', self.mav_state_current_)
        self.mpc_solver_.constraints_set(0, 'ubx', self.mav_state_current_)
        # initialize plan
        for iStage in range(0, self.mpc_N_):
            self.mpc_solver_.set(iStage, 'x', self.mav_state_current_)
            self.mpc_solver_.set(iStage, 'u', np.array([0.0, 0.0, g * self.mass_]))

    def initialize_acados_solver(self):
        # initial condition
        self.mpc_solver_.constraints_set(0, 'lbx', self.mav_state_current_)
        self.mpc_solver_.constraints_set(0, 'ubx', self.mav_state_current_)
        # initialize plan
        x_traj_init = np.concatenate((self.mpc_x_plan_[:, 1:], self.mpc_x_plan_[:, -1:]), axis=1)
        u_traj_init = np.concatenate((self.mpc_u_plan_[:, 1:], self.mpc_u_plan_[:, -1:]), axis=1)
        for iStage in range(0, self.mpc_N_):
            self.mpc_solver_.set(iStage, 'x', x_traj_init[:, iStage])
            self.mpc_solver_.set(iStage, 'u', u_traj_init[:, iStage])

    def set_acados_solver_ref(self):
        for iStage in range(0, self.mpc_N_):
            yref = np.concatenate((self.mpc_pos_ref_[:, iStage],
                                   self.mpc_vel_ref_[:, iStage],
                                   self.mpc_u_ref_[:, iStage]))
            self.mpc_solver_.set(iStage, 'yref', yref)
        yref_e = np.concatenate((self.mpc_pos_ref_[:, self.mpc_N_ - 1],
                                 self.mpc_vel_ref_[:, self.mpc_N_ - 1]))
        self.mpc_solver_.set(self.mpc_N_, 'yref', yref_e)

    def run_acados_solver(self):
        # initialize solver
        if self.mpc_feasible_ is True:
            self.initialize_acados_solver()
        else:
            self.reset_acados_solver()

        # set solver ref
        self.set_acados_solver_ref()

        # call the solver
        time_before_solver = rospy.get_rostime()
        solver_status = self.mpc_solver_.solve()

        # deal with infeasibility
        if solver_status != 0:  # if infeasible
            self.mpc_feasible_ = False
            rospy.logwarn("MPC infeasible, will try again.")
            # solve again
            self.reset_acados_solver()
            solver_status_alt = self.mpc_solver_.solve()
            if solver_status_alt != 0:  # if infeasible again
                self.mpc_feasible_ = False
                rospy.logwarn("MPC infeasible again.")
                return
            else:
                self.mpc_feasible_ = True
        else:
            self.mpc_feasible_ = True

        solver_time = (rospy.get_rostime() - time_before_solver).to_sec() * 1000.0
        rospy.loginfo('MPC computation time is: %s ms.', solver_time)

        # obtain solution
        for iStage in range(0, self.mpc_N_):
            self.mpc_x_plan_[:, iStage] = self.mpc_solver_.get(iStage, 'x')
            self.mpc_u_plan_[:, iStage] = self.mpc_solver_.get(iStage, 'u')
        self.mpc_x_next_ = self.mpc_x_plan_[:, 1]
        self.mpc_u_now_ = self.mpc_u_plan_[:, 1]

    def calculate_roll_pitch_yawrate_thrust_cmd(self):
        # if odom and traj command received
        time_now = rospy.Time.now()
        if (time_now-self.odom_received_time_).to_sec() > self.odom_time_out_:
            rospy.logwarn('Odometry time out! Will try to make the MAV hover.')
            self.mpc_feasible_ = False  # will not run mpc if odometry not received
        elif (time_now-self.traj_received_time_).to_sec() > self.traj_time_out_ \
                and self.tracking_mode_ == 'track':
            rospy.logwarn('Trajectory command time out! Will try to make the MAV hover.')
            self.set_mpc_ref('hover')
            self.run_acados_solver()
        else:
            self.set_mpc_ref(self.tracking_mode_)
            self.run_acados_solver()

        # control commands
        if self.mpc_feasible_ is True:
            roll_cmd = self.mpc_u_now_[0]
            pitch_cmd = self.mpc_u_now_[1]
            thrust_cmd = self.mpc_u_now_[2]
        else:
            roll_cmd = 0.0
            pitch_cmd = 0.0
            thrust_cmd = g*self.mass_

        # TODO: yaw controller
        yawrate_cmd = 0.0

        # obtained command
        self.mav_control_current_ = np.array([roll_cmd, pitch_cmd, yawrate_cmd, thrust_cmd])

    def pub_roll_pitch_yawrate_thrust_cmd(self):
        cmd_msg = RollPitchYawrateThrust()
        cmd_msg.header.stamp = rospy.get_rostime()
        cmd_msg.roll = self.mav_control_current_[0]
        cmd_msg.pitch = self.mav_control_current_[1]
        cmd_msg.yaw_rate = self.mav_control_current_[2]
        cmd_msg.thrust.x = 0.0
        cmd_msg.thrust.y = 0.0
        cmd_msg.thrust.z = self.mav_control_current_[3]
        self.roll_pitch_yawrate_thrust_pub_.publish(cmd_msg)


def nmpc_tracker_control():
    # create a node
    rospy.loginfo("Starting NMPC tracking...")
    rospy.init_node("mav_nmpc_tracker_node", anonymous=False)
    hz = 50
    dt = 1.0 / hz
    rate = rospy.Rate(hz)
    rospy.sleep(1.0)

    # fetch param
    # tracking mode
    tracking_mode = rospy.get_param("~tracking_mode")
    # horizon
    mpc_form_param = MPC_Formulation_Param()
    mpc_form_param.dt = rospy.get_param("~dt")
    mpc_form_param.N = rospy.get_param("~N")
    mpc_form_param.Tf = mpc_form_param.N * mpc_form_param.dt
    # mav dynamics
    mpc_form_param.mass = rospy.get_param("~mass")
    mpc_form_param.roll_time_constant = rospy.get_param("~roll_time_constant")
    mpc_form_param.roll_gain = rospy.get_param("~roll_gain")
    mpc_form_param.pitch_time_constant = rospy.get_param("~pitch_time_constant")
    mpc_form_param.pitch_gain = rospy.get_param("~pitch_gain")
    mpc_form_param.drag_coefficient_x = rospy.get_param("~drag_coefficient_x")
    mpc_form_param.drag_coefficient_y = rospy.get_param("~drag_coefficient_y")
    # control bound
    mpc_form_param.roll_max = rospy.get_param("~roll_max")
    mpc_form_param.pitch_max = rospy.get_param("~pitch_max")
    mpc_form_param.thrust_min = rospy.get_param("~thrust_min")
    mpc_form_param.thrust_max = rospy.get_param("~thrust_max")
    # cost weights
    mpc_form_param.q_x = rospy.get_param("~q_x")
    mpc_form_param.q_y = rospy.get_param("~q_y")
    mpc_form_param.q_z = rospy.get_param("~q_z")
    mpc_form_param.q_vx = rospy.get_param("~q_vx")
    mpc_form_param.q_vy = rospy.get_param("~q_vy")
    mpc_form_param.q_vz = rospy.get_param("~q_vz")
    mpc_form_param.r_roll = rospy.get_param("~r_roll")
    mpc_form_param.r_pitch = rospy.get_param("~r_pitch")
    mpc_form_param.r_thrust = rospy.get_param("~r_thrust")

    # create a nmpc tracker
    nmpc_tracker = Mav_Nmpc_Tracker(mpc_form_param, tracking_mode)

    while not rospy.is_shutdown():
        if nmpc_tracker.received_first_odom_ is False:
            rospy.logwarn('Waiting for first Odometry!')
        else:
            nmpc_tracker.calculate_roll_pitch_yawrate_thrust_cmd()
            nmpc_tracker.pub_roll_pitch_yawrate_thrust_cmd()
        rate.sleep()


if __name__ == "__main__":
    nmpc_tracker_control()
