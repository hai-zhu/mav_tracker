#!/usr/bin/env python

import numpy as np
import rospy
import tf
from nav_msgs.msg import Odometry
from mav_msgs.msg import RollPitchYawrateThrust
from mavros_msgs.msg import AttitudeTarget
from trajectory_msgs.msg import MultiDOFJointTrajectory
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from nmpc_tracker_solver import MPC_Formulation_Param
from nmpc_tracker_solver import acados_mpc_solver_generation

g = 9.8066

# The frame by default is NWU

class Mav_Nmpc_Tracker:
    def __init__(self, mpc_form_param, tracking_mode, yaw_command_mode):
        # MPC formulation settings
        self.mpc_form_param_ = mpc_form_param

        # mode
        self.tracking_mode_ = tracking_mode  # track, hover, home
        self.yaw_command_mode_ = yaw_command_mode  # yaw, yawrate

        # mav mass, and settins
        self.mass_ = self.mpc_form_param_.mass
        self.thrust_scale_ = self.mpc_form_param_.thrust_scale
        self.odom_time_out_ = 0.2
        self.traj_time_out_ = 1.0

        # state
        self.mav_state_current_ = np.zeros(9)

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
        self.mpc_success_ = False

        # MPC solver
        self.mpc_solver_ = acados_mpc_solver_generation(self.mpc_form_param_)

        # ROS subscriber
        # self.odom_sub_ = rospy.Subscriber("/mav_odometry", Odometry, self.set_odom)
        self.odom_sub_ = rospy.Subscriber("/mavros/local_position/odom_local", Odometry, self.set_odom)
        self.received_first_odom_ = False
        self.odom_received_time_ = rospy.Time.now()
        # self.traj_sub_ = rospy.Subscriber("/mav_trajectory", MultiDOFJointTrajectory, self.set_traj_ref)
        self.traj_sub_ = rospy.Subscriber("/command/trajectory", MultiDOFJointTrajectory, self.set_traj_ref)
        self.traj_received_time_ = rospy.Time.now()
        self.traj_pos_ref_ = np.zeros((3, self.mpc_N_))
        self.traj_vel_ref_ = np.zeros((3, self.mpc_N_))

        # ROS publisher
        self.roll_pitch_yawrate_thrust_cmd_ = np.array(4)
        self.roll_pitch_yawrate_thrust_cmd_pub_ = rospy.Publisher("/mav_roll_pitch_yawrate_thrust_cmd", \
            RollPitchYawrateThrust, queue_size=1)
        self.roll_pitch_yaw_thrust_cmd_ = np.array(4) 
        self.roll_pitch_yaw_thrust_cmd_pub_ = rospy.Publisher("/mav_roll_pitch_yaw_thrust_cmd", \
            AttitudeTarget, queue_size=1)

        self.mpc_traj_plan_vis_pub_ = rospy.Publisher("/mpc/trajectory_plan_vis", Marker, queue_size=1)

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
        try:
            for iStage in range(0, self.mpc_N_):
                self.traj_pos_ref_[0, iStage] = traj_msg.points[iStage].transforms[0].translation.x
                self.traj_pos_ref_[1, iStage] = traj_msg.points[iStage].transforms[0].translation.y
                self.traj_pos_ref_[2, iStage] = traj_msg.points[iStage].transforms[0].translation.z
                self.traj_vel_ref_[0, iStage] = traj_msg.points[iStage].velocities[0].linear.x
                self.traj_vel_ref_[1, iStage] = traj_msg.points[iStage].velocities[0].linear.y
                self.traj_vel_ref_[2, iStage] = traj_msg.points[iStage].velocities[0].linear.z
        except:
            rospy.logwarn('Received commanded trajectory incorrect! Will try to hover')
            self.traj_pos_ref_ = np.tile(self.mav_state_current_[0:3].reshape((-1, 1)), (1, self.mpc_N_))
            self.traj_vel_ref_ = np.tile(np.array([0.0, 0.0, 0.0]).reshape((-1, 1)), (1, self.mpc_N_))

    def set_mpc_ref(self, mode):
        if mode == 'track':  # trajectory tracking
            self.mpc_pos_ref_ = self.traj_pos_ref_
            self.mpc_vel_ref_ = self.traj_vel_ref_
        elif mode == 'hover':  # hovering
            self.mpc_pos_ref_ = np.tile(self.mav_state_current_[0:3].reshape((-1, 1)), (1, self.mpc_N_))
            self.mpc_vel_ref_ = np.tile(np.array([0.0, 0.0, 0.0]).reshape((-1, 1)), (1, self.mpc_N_))
        elif mode == 'home':  # flying to origin
            self.mpc_pos_ref_ = np.tile(np.array([0.0, 0.0, 1.0]).reshape((-1, 1)), (1, self.mpc_N_))
            self.mpc_vel_ref_ = np.tile(np.array([0.0, 0.0, 0.0]).reshape((-1, 1)), (1, self.mpc_N_))
        else:
            rospy.logwarn('Tracking mode is not correctly set!')
        self.mpc_u_ref_ = np.tile(np.array([0.0, 0.0, 1.0*g]).reshape((-1, 1)), (1, self.mpc_N_))

    def reset_acados_solver(self):
        # initial condition
        self.mpc_solver_.constraints_set(0, 'lbx', self.mav_state_current_)
        self.mpc_solver_.constraints_set(0, 'ubx', self.mav_state_current_)
        # initialize plan
        for iStage in range(0, self.mpc_N_):
            self.mpc_solver_.set(iStage, 'x', self.mav_state_current_)
            self.mpc_solver_.set(iStage, 'u', np.array([0.0, 0.0, 1.0*g]))

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
            self.mpc_success_ = False 
            rospy.logwarn("MPC infeasible, will try again.")
            # solve again
            self.reset_acados_solver()
            solver_status_alt = self.mpc_solver_.solve()
            if solver_status_alt != 0:  # if infeasible again
                self.mpc_feasible_ = False
                self.mpc_success_ = False
                rospy.logwarn("MPC infeasible again.")
                return
            else:
                self.mpc_feasible_ = True
                self.mpc_success_ = True 
        else:
            self.mpc_feasible_ = True
            self.mpc_success_ = True 

        solver_time = (rospy.get_rostime() - time_before_solver).to_sec() * 1000.0
        # rospy.loginfo('MPC computation time is: %s ms.', solver_time)

        # obtain solution
        for iStage in range(0, self.mpc_N_):
            self.mpc_x_plan_[:, iStage] = self.mpc_solver_.get(iStage, 'x')
            self.mpc_u_plan_[:, iStage] = self.mpc_solver_.get(iStage, 'u')
        self.mpc_x_next_ = self.mpc_x_plan_[:, 1]
        self.mpc_u_now_ = self.mpc_u_plan_[:, 0]

    def calculate_roll_pitch_yawrate_thrust_cmd(self):
        # if odom and traj command received
        time_now = rospy.Time.now()
        if (time_now-self.odom_received_time_).to_sec() > self.odom_time_out_:
            rospy.logwarn('Odometry time out! Will try to make the MAV hover.')
            self.mpc_feasible_ = False  # will not run mpc if odometry not received
            self.mpc_success_ = False 
        elif (time_now-self.traj_received_time_).to_sec() > self.traj_time_out_ \
                and self.tracking_mode_ == 'track':
            rospy.logwarn('Trajectory command time out! Will try to make the MAV hover.')
            self.set_mpc_ref('hover')
            self.run_acados_solver()
        else:
            self.set_mpc_ref(self.tracking_mode_)
            self.run_acados_solver()

        # control commands
        if self.mpc_success_ is True:
            roll_cmd = self.mpc_u_now_[0]
            pitch_cmd = self.mpc_u_now_[1]
            thrust_cmd = self.mpc_u_now_[2]*self.mass_/self.thrust_scale_
        else:
            roll_cmd = 0.0
            pitch_cmd = 0.0
            thrust_cmd = 1.0*g*self.mass_/self.thrust_scale_

        # yaw controller
        current_yaw = self.mav_state_current_[8]
        yaw_ref = 0.0   # TODO: change to real-time yaw ref
        yaw_error = yaw_ref - current_yaw

        if np.abs(yaw_error) > np.pi:
            if yaw_error > 0.0:
                yaw_error = yaw_error - 2.0*np.pi 
            else:
                yaw_error = yaw_error + 2.0*np.pi 
        
        yawrate_cmd = self.mpc_form_param_.K_yaw * yaw_error

        # clip
        # roll_cmd = np.clip(roll_cmd, -self.mpc_form_param_.roll_max, self.mpc_form_param_.roll_max)
        # pitch_cmd = np.clip(pitch_cmd, -self.mpc_form_param_.pitch_max, self.mpc_form_param_.pitch_max)
        # yawrate_cmd = np.clip(yawrate_cmd, -self.mpc_form_param_.yawrate_max, self.mpc_form_param_.yawrate_max)
        # thrust_cmd = np.clip(thrust_cmd, self.mpc_form_param_.thrust_min*self.mass_/self.thrust_scale_, \
        #     self.mpc_form_param_.thrust_max*self.mass_/self.thrust_scale_)

        # obtained command
        self.roll_pitch_yawrate_thrust_cmd_ = np.array([roll_cmd, pitch_cmd, yawrate_cmd, thrust_cmd])
        self.roll_pitch_yaw_thrust_cmd_ = np.array([roll_cmd, pitch_cmd, yaw_ref, thrust_cmd])

    def pub_roll_pitch_yawrate_thrust_cmd(self):
        try: 
            cmd_msg = RollPitchYawrateThrust()
            cmd_msg.header.stamp = rospy.get_rostime()
            cmd_msg.roll = self.roll_pitch_yawrate_thrust_cmd_[0]
            cmd_msg.pitch = self.roll_pitch_yawrate_thrust_cmd_[1]
            cmd_msg.yaw_rate = self.roll_pitch_yawrate_thrust_cmd_[2]
            cmd_msg.thrust.x = 0.0
            cmd_msg.thrust.y = 0.0
            cmd_msg.thrust.z = self.roll_pitch_yawrate_thrust_cmd_[3]
            self.roll_pitch_yawrate_thrust_cmd_pub_.publish(cmd_msg)
        except:
            rospy.logwarn('MAV roll_pitch_yawrate_thrust command not published!')

    def pub_roll_pitch_yaw_thrust_cmd(self):
        try: 
            cmd_msg = AttitudeTarget()
            cmd_msg.header.stamp = rospy.get_rostime()
            quat = tf.transformations.quaternion_from_euler(self.roll_pitch_yaw_thrust_cmd_[0], 
                                                            self.roll_pitch_yaw_thrust_cmd_[1], 
                                                            self.roll_pitch_yaw_thrust_cmd_[2])
            cmd_msg.orientation.x = quat[0]
            cmd_msg.orientation.y = quat[1]
            cmd_msg.orientation.z = quat[2]
            cmd_msg.orientation.w = quat[3]
            cmd_msg.thrust = self.roll_pitch_yaw_thrust_cmd_[3]
            self.roll_pitch_yaw_thrust_cmd_pub_.publish(cmd_msg)
        except:
            rospy.logwarn('MAV roll_pitch_yawrate_thrust command not published!')

    def pub_mpc_traj_plan_vis(self):
        try:
            marker_msg = Marker()
            marker_msg.header.frame_id = "map"
            marker_msg.header.stamp = rospy.Time.now()
            marker_msg.type = 8
            marker_msg.action = 0
            # set the scale of the marker
            marker_msg.scale.x = 0.2
            marker_msg.scale.y = 0.2
            marker_msg.scale.z = 0.2
            # set the color
            marker_msg.color.r = 1.0
            marker_msg.color.g = 0.0
            marker_msg.color.b = 0.0
            marker_msg.color.a = 1.0
            # Set the pose of the marker
            marker_msg.pose.position.x =  0.0
            marker_msg.pose.position.y =  0.0
            marker_msg.pose.position.z =  0.0
            marker_msg.pose.orientation.x = 0
            marker_msg.pose.orientation.y = 0
            marker_msg.pose.orientation.z = 0
            marker_msg.pose.orientation.w = 1.0
            # points
            mpc_traj_plan_points = []
            for iStage in range(0, self.mpc_N_):
                point = Point(self.mpc_x_plan_[0, iStage], self.mpc_x_plan_[1, iStage], self.mpc_x_plan_[2, iStage])
                mpc_traj_plan_points.append(point)
            marker_msg.points = mpc_traj_plan_points
            self.mpc_traj_plan_vis_pub_.publish(marker_msg)
        except:
            rospy.logwarn("MPC trajectory plan not published!")


def nmpc_tracker_control():
    # create a node
    rospy.loginfo("Starting NMPC tracking...")
    rospy.init_node("mav_nmpc_tracker_node", anonymous=False)
    hz = 40
    dt = 1.0 / hz
    rate = rospy.Rate(hz)
    rospy.sleep(1.0)

    # fetch param
    # tracking mode
    tracking_mode = rospy.get_param("~tracking_mode")
    rospy.loginfo('The running mode is: %s.', tracking_mode)
    yaw_command_mode = rospy.get_param("~yaw_command_mode")
    rospy.loginfo('The yaw control mode is: %s.', yaw_command_mode)
    # horizon
    mpc_form_param = MPC_Formulation_Param()
    mpc_form_param.dt = rospy.get_param("~dt")
    mpc_form_param.N = rospy.get_param("~N")
    mpc_form_param.Tf = mpc_form_param.N * mpc_form_param.dt
    # mav dynamics
    mpc_form_param.mass = rospy.get_param("~mass")
    mpc_form_param.thrust_scale = rospy.get_param("~thrust_scale")
    mpc_form_param.roll_time_constant = rospy.get_param("~roll_time_constant")
    mpc_form_param.roll_gain = rospy.get_param("~roll_gain")
    mpc_form_param.pitch_time_constant = rospy.get_param("~pitch_time_constant")
    mpc_form_param.pitch_gain = rospy.get_param("~pitch_gain")
    mpc_form_param.drag_coefficient_x = rospy.get_param("~drag_coefficient_x")
    mpc_form_param.drag_coefficient_y = rospy.get_param("~drag_coefficient_y")
    # control bound
    mpc_form_param.roll_max = np.deg2rad(rospy.get_param("~roll_max"))
    mpc_form_param.pitch_max = np.deg2rad(rospy.get_param("~pitch_max"))
    mpc_form_param.thrust_min = rospy.get_param("~thrust_min") * mpc_form_param.mass * g
    mpc_form_param.thrust_max = rospy.get_param("~thrust_max") * mpc_form_param.mass * g
    mpc_form_param.K_yaw = rospy.get_param("~K_yaw")
    mpc_form_param.yawrate_max = np.deg2rad(rospy.get_param("~yawrate_max"))
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
    nmpc_tracker = Mav_Nmpc_Tracker(mpc_form_param, tracking_mode, yaw_command_mode)

    while not rospy.is_shutdown():
        if nmpc_tracker.received_first_odom_ is False:
            rospy.logwarn('Waiting for first Odometry!')
        else:
            nmpc_tracker.calculate_roll_pitch_yawrate_thrust_cmd()
            if nmpc_tracker.yaw_command_mode_ == 'yawrate':
                nmpc_tracker.pub_roll_pitch_yawrate_thrust_cmd()
            elif nmpc_tracker.yaw_command_mode_ == 'yaw':
                nmpc_tracker.pub_roll_pitch_yaw_thrust_cmd()
            else: 
                rospy.logwarn('yaw control mode is not set!')
            nmpc_tracker.pub_mpc_traj_plan_vis()
        rate.sleep()


if __name__ == "__main__":
    nmpc_tracker_control()
