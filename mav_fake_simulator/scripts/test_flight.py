#!/usr/bin/env python

import numpy as np
from scipy import signal
import rospy
from mav_msgs.msg import RollPitchYawrateThrust

FLIGHT_TIME = 10.0      # s
PERIOD = 2.0            # s
MASS = 1.56             # kg
G = 9.8066


def test_flight_roll():
    print("Starting test flight...")
    rospy.init_node("mav_test_flight_control_node", anonymous=False)
    hz = 20
    dt = 1.0/hz
    rate = rospy.Rate(hz)
    rospy.sleep(0.1)
    cmd_pub = rospy.Publisher("/mav_roll_pitch_yawrate_thrust_cmd", RollPitchYawrateThrust, queue_size=1)
    cmd_msg = RollPitchYawrateThrust()
    cmd_rad = np.deg2rad(10)
    time_samples = np.arange(0, FLIGHT_TIME+0.15*PERIOD, dt)
    signal_samples = cmd_rad*signal.square(2*np.pi*time_samples/PERIOD)
    idx = int(0.15*PERIOD/dt)
    while not rospy.is_shutdown() and idx < len(signal_samples):
        cmd_msg.header.stamp = rospy.get_rostime()
        cmd_msg.roll = signal_samples[idx]
        cmd_msg.pitch = 0.0
        cmd_msg.yaw_rate = 0.0
        cmd_msg.thrust.x = 0.0
        cmd_msg.thrust.y = 0.0
        cmd_msg.thrust.z = 1.004*MASS * G
        cmd_pub.publish(cmd_msg)
        idx += 1
        rate.sleep()


def test_flight_pitch():
    pass


def test_flight_yaw():
    pass


def test_flight_random():
    pass


if __name__ == "__main__":
    test_flight_roll()


