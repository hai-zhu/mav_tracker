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
    def __init__(self, mpc_form_param):
        # MPC formulation settings
        self.mpc_form_param_ = mpc_form_param

        # mass
        self.mass_ = self.mpc_form_param_.mass

        # state
        self.mav_state_current_ = np.zeros(9)
        self.mav_control_current_ = np.array([0.0, 0.0, 0.0, g*self.mass_])

        # MPC settings
        self.mpc_dt_ = self.mpc_form_param_.dt
        self.mpc_N_ = self.mpc_form_param_.N
        self.mpc_Tf_ = self.mpc_N_ * self.mpc_dt_
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
        self.mpc_feasible_ = False

        # MPC solver
        self.mpc_solver_ = acados_mpc_solver_generation(self.mpc_form_param_)

        # ROS subscriber
        self.odom_sub_ = rospy.Subscriber("/mav_sim_odom", Odometry, self.set_odom)
        self.traj_sub_ = rospy.Subscriber("/command/trajectory", MultiDOFJointTrajectory, self.set_traj_ref)

        # ROS publisher
        self.roll_pitch_yawrate_thrust_pub_ = rospy.Publisher("/mav_roll_pitch_yawrate_thrust_cmd",
                                                              RollPitchYawrateThrust, queue_size=1)

        # fly to origin by default
        for iStage in range(0, self.mpc_N_):
            self.mpc_pos_ref_[2, iStage] = 1.0
            self.mpc_u_ref_[2, iStage] = g*self.mass_

    def set_odom(self, odom_msg):
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
        # for iStage in range(0, self.mpc_N_):
        #     self.mpc_pos_ref_[0, iStage] = traj_msg.points[iStage].transforms[0].translation.x
        #     self.mpc_pos_ref_[1, iStage] = traj_msg.points[iStage].transforms[0].translation.y
        #     self.mpc_pos_ref_[2, iStage] = traj_msg.points[iStage].transforms[0].translation.z
        #     self.mpc_vel_ref_[0, iStage] = traj_msg.points[iStage].velocities[0].linear.x
        #     self.mpc_vel_ref_[1, iStage] = traj_msg.points[iStage].velocities[0].linear.y
        #     self.mpc_vel_ref_[2, iStage] = traj_msg.points[iStage].velocities[0].linear.z
        pass

    def reset_acados_solver(self):
        # initial condition
        self.mpc_solver_.constraints_set(0, 'lbx', self.mav_state_current_)
        self.mpc_solver_.constraints_set(0, 'ubx', self.mav_state_current_)
        # initialize plan
        for iStage in range(0, self.mpc_N_):
            self.mpc_solver_.set(iStage, 'x', self.mav_state_current_)
            self.mpc_solver_.set(iStage, 'u', np.array([0.0, 0.0, g*self.mass_]))

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
        yref_e = np.concatenate((self.mpc_pos_ref_[:, self.mpc_N_-1],
                                 self.mpc_vel_ref_[:, self.mpc_N_-1]))
        self.mpc_solver_.set(self.mpc_N_, 'yref', yref_e)

    def calculate_roll_pitch_yawrate_thrust_cmd(self):
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
        if solver_status != 0:          # if infeasible
            self.mpc_feasible_ = False
            print("MPC infeasible, will try again.")
            # solve again
            self.reset_acados_solver()
            solver_status_alt = self.mpc_solver_.solve()
            if solver_status_alt != 0:  # if infeasible again
                self.mpc_feasible_ = False
                print("MPC infeasible again.")
                self.mav_control_current_ = np.array([0.0, 0.0, 0.0, g*self.mass_])
                self.pub_roll_pitch_yawrate_thrust_cmd()
                return
            else:
                self.mpc_feasible_ = True
        else:
            self.mpc_feasible_ = True

        solver_time = (rospy.get_rostime() - time_before_solver).to_sec() * 1000.0

        # obtain solution
        for iStage in range(0, self.mpc_N_):
            self.mpc_x_plan_[:, iStage] = self.mpc_solver_.get(iStage, 'x')
            self.mpc_u_plan_[:, iStage] = self.mpc_solver_.get(iStage, 'u')
        roll_cmd = self.mpc_u_plan_[0, 1]
        pitch_cmd = self.mpc_u_plan_[1, 1]
        thrust_cmd = self.mpc_u_plan_[2, 1]

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


def nmpc_fly_to_origin():
    # create a node
    print("Starting NMPC tracking...")
    rospy.init_node("mav_nmpc_tracker_node", anonymous=False)
    hz = 50
    dt = 1.0 / hz
    rate = rospy.Rate(hz)
    rospy.sleep(0.1)

    # create a nmpc tracker
    mpc_form_param = MPC_Formulation_Param()
    nmpc_tracker = Mav_Nmpc_Tracker(mpc_form_param)

    while not rospy.is_shutdown():
        nmpc_tracker.calculate_roll_pitch_yawrate_thrust_cmd()
        nmpc_tracker.pub_roll_pitch_yawrate_thrust_cmd()
        rate.sleep()


if __name__ == "__main__":
    nmpc_fly_to_origin()
