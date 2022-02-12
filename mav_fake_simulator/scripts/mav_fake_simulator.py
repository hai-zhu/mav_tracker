#!/usr/bin/env python

import numpy as np
import rospy
import tf
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
from mav_msgs.msg import RollPitchYawrateThrust
from mav_dynamics import mav_dynamics_continuous
from integrator import my_RK4

g = 9.8066


class MAV_Fake_Motion_Simulator:
    def __init__(self):
        # MAV mass, dynamics model and parameters
        self.mass_ = 0.0
        self.roll_time_constant_ = 0.0
        self.roll_gain_ = 0.0
        self.pitch_time_constant_ = 0.0
        self.pitch_gain_ = 0.0
        self.drag_coefficient_x_ = 0.0
        self.drag_coefficient_y_ = 0.0
        self.dynamics_model_ = mav_dynamics_continuous

        # MAV starting state in simulation
        self.pos_start_ = np.zeros((3, ))
        self.vel_start_ = np.zeros((3, ))
        self.rpy_start_ = np.zeros((3, ))

        # MAV simulation dt, current time and state
        self.sim_dt_ = 0.01
        self.time_out_ = 0.1
        self.time_now_ = 0.0
        self.pos_ = np.zeros((3,))
        self.vel_ = np.zeros((3,))
        self.rpy_ = np.zeros((3,))
        self.quaternion_ = np.zeros((4,))

        # Flag
        self.hit_ground_ = True

        # MAV control command
        self.roll_pitch_yawrate_thrust_cmd_ = np.zeros((4,))

        # ROS subscriber
        self.roll_pitch_yawrate_thrust_sub_ = rospy.Subscriber("/mav_roll_pitch_yawrate_thrust_cmd",
                                                               RollPitchYawrateThrust,
                                                               self.roll_pitch_yawrate_thrust_callback)
        self.cmd_received_time_ = -self.time_out_
        # ROS publisher
        self.odom_pub_ = rospy.Publisher("/mav_sim_odom", Odometry, queue_size=1)
        self.odom_msg_ = Odometry()
        self.odom_msg_.header.frame_id = "map"
        self.odom_msg_.child_frame_id = "base_link"
        self.cmd_pub_ = rospy.Publisher("/mav_sim_cmd", RollPitchYawrateThrust, queue_size=1)
        self.cmd_msg_ = RollPitchYawrateThrust()
        self.cmd_msg_.header.frame_id = "base_link"

    def reset_sim(self):
        self.time_now_ = 0.0
        self.pos_ = self.pos_start_
        self.vel_ = self.vel_start_
        self.rpy_ = self.rpy_start_
        self.quaternion_ = tf.transformations.quaternion_from_euler(self.rpy_[0], self.rpy_[1], self.rpy_[2])
        self.odom_msg_.header.seq = 0
        self.odom_msg_.header.stamp.secs = 0
        self.odom_msg_.header.stamp.nsecs = 0
        self.cmd_msg_.header.seq = 0
        self.cmd_msg_.header.stamp.secs = 0
        self.cmd_msg_.header.stamp.nsecs = 0

    def set_sim(self, pos, vel, rpy):
        self.pos_start_ = pos
        self.vel_start_ = vel
        self.rpy_start_ = rpy
        self.reset_sim()
        if self.pos_start_[2] <= 0.1:
            self.hit_ground_ = True
            print("Invalid mav initial state!")
        else:
            self.hit_ground_ = False

    def set_param(self, sim_dt, time_out, mass, roll_time_constant, roll_gain,
                  pitch_time_constant, pitch_gain, drag_coefficient_x, drag_coefficient_y):
        self.sim_dt_ = sim_dt
        self.time_out_ = time_out
        self.mass_ = mass
        self.roll_time_constant_ = roll_time_constant
        self.roll_gain_ = roll_gain
        self.pitch_time_constant_ = pitch_time_constant
        self.pitch_gain_ = pitch_gain
        self.drag_coefficient_x_ = drag_coefficient_x
        self.drag_coefficient_y_ = drag_coefficient_y

    def roll_pitch_yawrate_thrust_callback(self, roll_pitch_yawrate_thrust_msg):
        self.roll_pitch_yawrate_thrust_cmd_[0] = roll_pitch_yawrate_thrust_msg.roll
        self.roll_pitch_yawrate_thrust_cmd_[1] = roll_pitch_yawrate_thrust_msg.pitch
        self.roll_pitch_yawrate_thrust_cmd_[2] = roll_pitch_yawrate_thrust_msg.yaw_rate
        self.roll_pitch_yawrate_thrust_cmd_[3] = roll_pitch_yawrate_thrust_msg.thrust.z
        self.cmd_received_time_ = self.time_now_

    def step_sim(self):
        # if time out, try to make the mav hover
        if self.time_now_ - self.cmd_received_time_ >= self.time_out_:
            self.roll_pitch_yawrate_thrust_cmd_.fill(0.0)
            self.roll_pitch_yawrate_thrust_cmd_[3] = \
                self.mass_ * g / np.cos(self.rpy_[0]) / np.cos(self.rpy_[1])
        if self.pos_[2] <= 0.1:
            self.hit_ground_ = True
        if self.hit_ground_:       # hit ground
            print("MAV hits ground!")
        else:
            pos_vel_rpy_now = np.concatenate((self.pos_, self.vel_, self.rpy_))
            roll_pitch_yawrate_thrust_now = self.roll_pitch_yawrate_thrust_cmd_
            param_now = np.array([self.mass_, self.roll_time_constant_, self.roll_gain_,
                                  self.pitch_time_constant_, self.pitch_gain_,
                                  self.drag_coefficient_x_, self.drag_coefficient_y_])
            pos_vel_rpy_next = my_RK4(pos_vel_rpy_now, roll_pitch_yawrate_thrust_now, self.dynamics_model_,
                                      self.sim_dt_, param_now)
            self.pos_ = pos_vel_rpy_next[0:3]
            self.vel_ = pos_vel_rpy_next[3:6]
            self.rpy_ = pos_vel_rpy_next[6:9]
            self.time_now_ += self.sim_dt_

    def pub_sim_cmd(self):
        self.cmd_msg_.header.seq += 1
        self.cmd_msg_.header.stamp.secs = int(np.floor(self.time_now_))
        self.cmd_msg_.header.stamp.nsecs = int((self.time_now_ - np.floor(self.time_now_)) * 1E9)
        self.cmd_msg_.roll = self.roll_pitch_yawrate_thrust_cmd_[0]
        self.cmd_msg_.pitch = self.roll_pitch_yawrate_thrust_cmd_[1]
        self.cmd_msg_.yaw_rate = self.roll_pitch_yawrate_thrust_cmd_[2]
        self.cmd_msg_.thrust.x = 0.0
        self.cmd_msg_.thrust.y = 0.0
        self.cmd_msg_.thrust.z = self.roll_pitch_yawrate_thrust_cmd_[3]
        self.cmd_pub_.publish(self.cmd_msg_)

    def pub_sim_odom(self):
        self.odom_msg_.header.seq += 1
        self.odom_msg_.header.stamp.secs = int(np.floor(self.time_now_))
        self.odom_msg_.header.stamp.nsecs = int((self.time_now_ - np.floor(self.time_now_)) * 1E9)
        self.odom_msg_.pose.pose.position.x = self.pos_[0]
        self.odom_msg_.pose.pose.position.y = self.pos_[1]
        self.odom_msg_.pose.pose.position.z = self.pos_[2]
        self.quaternion_ = tf.transformations.quaternion_from_euler(self.rpy_[0], self.rpy_[1], self.rpy_[2])
        self.odom_msg_.pose.pose.orientation = Quaternion(*self.quaternion_)
        self.odom_msg_.twist.twist.linear.x = self.vel_[0]
        self.odom_msg_.twist.twist.linear.y = self.vel_[1]
        self.odom_msg_.twist.twist.linear.z = self.vel_[2]
        self.odom_pub_.publish(self.odom_msg_)


def run_fake_simulation():
    print("Starting simulation...")
    # Initialize a ros node
    rospy.init_node("mav_motion_simulator_node", anonymous=False)

    # Fetch param
    # simulation
    sim_dt = rospy.get_param("~sim_dt")
    rospy.loginfo("%s is %s", rospy.resolve_name('~sim_dt'), sim_dt)
    time_out = rospy.get_param("~time_out")
    rospy.loginfo("%s is %s", rospy.resolve_name('~time_out'), time_out)
    # mav dynamics
    mass = rospy.get_param("~mass")
    rospy.loginfo("%s is %s", rospy.resolve_name('~mass'), mass)
    roll_time_constant = rospy.get_param("~roll_time_constant")
    rospy.loginfo("%s is %s", rospy.resolve_name('~roll_time_constant'), roll_time_constant)
    roll_gain = rospy.get_param("~roll_gain")
    rospy.loginfo("%s is %s", rospy.resolve_name('~roll_gain'), roll_gain)
    pitch_time_constant = rospy.get_param("~pitch_time_constant")
    rospy.loginfo("%s is %s", rospy.resolve_name('~pitch_time_constant'), pitch_time_constant)
    pitch_gain = rospy.get_param("~pitch_gain")
    rospy.loginfo("%s is %s", rospy.resolve_name('~pitch_gain'), pitch_gain)
    drag_coefficient_x = rospy.get_param("~drag_coefficient_x")
    rospy.loginfo("%s is %s", rospy.resolve_name('~drag_coefficient_x'), drag_coefficient_x)
    drag_coefficient_y = rospy.get_param("~drag_coefficient_y")
    rospy.loginfo("%s is %s", rospy.resolve_name('~drag_coefficient_y'), drag_coefficient_y)
    # initial state
    pos_start = np.array(rospy.get_param("~pos_start"))
    rospy.loginfo("%s is %s", rospy.resolve_name('/pos_start'), pos_start)
    vel_start = np.array(rospy.get_param("~vel_start"))
    rospy.loginfo("%s is %s", rospy.resolve_name('/vel_start'), vel_start)
    rpy_start = np.array(rospy.get_param("~rpy_start"))
    rospy.loginfo("%s is %s", rospy.resolve_name('~rpy_start'), rpy_start)

    # Create a simulator
    mav_simulator = MAV_Fake_Motion_Simulator()
    mav_simulator.set_param(sim_dt, time_out, mass, roll_time_constant, roll_gain,
                            pitch_time_constant, pitch_gain, drag_coefficient_x, drag_coefficient_y)
    mav_simulator.set_sim(pos_start, vel_start, rpy_start)

    # Specify the node frequency
    hz = 1.0/mav_simulator.sim_dt_
    rate = rospy.Rate(hz)
    rospy.sleep(0.1)

    # Start running
    while not rospy.is_shutdown():
        mav_simulator.step_sim()
        mav_simulator.pub_sim_cmd()
        mav_simulator.pub_sim_odom()
        rate.sleep()


if __name__ == "__main__":
    run_fake_simulation()



