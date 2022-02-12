import numpy as np

g = 9.8066


def mav_dynamics_continuous(x, u, p):       # Rotation: z-y-x
    # parameter
    mass = p[0]  # kg
    roll_time_constant = p[1]
    roll_gain = p[2]
    pitch_time_constant = p[3]
    pitch_gain = p[4]
    drag_coefficient_x = p[5]
    drag_coefficient_y = p[6]
    # state, m, m/s, rad
    px = x[0]
    py = x[1]
    pz = x[2]
    vx = x[3]
    vy = x[4]
    vz = x[5]
    roll = x[6]
    pitch = x[7]
    yaw = x[8]
    # control
    roll_cmd = u[0]
    pitch_cmd = u[1]
    yawrate_cmd = u[2]
    thrust_cmd = u[3] / mass  # divided by mass
    # drag
    drag_acc_x = np.cos(pitch) * np.cos(yaw) * drag_coefficient_x * thrust_cmd * vx \
                 - np.cos(pitch) * np.sin(yaw) * drag_coefficient_x * thrust_cmd * vy \
                 + np.sin(pitch) * drag_coefficient_x * thrust_cmd * vz
    drag_acc_y = (np.cos(roll)*np.sin(yaw) - np.cos(yaw)*np.sin(pitch)*np.sin(roll))*drag_coefficient_y*thrust_cmd*vx \
                 - (np.cos(roll)*np.cos(yaw) + np.sin(pitch)*np.sin(roll)*np.sin(yaw))*drag_coefficient_y*thrust_cmd*vy \
                 - np.cos(pitch)*np.sin(roll)*drag_coefficient_y*thrust_cmd*vz
    # derivative
    px_dot = vx
    py_dot = vy
    pz_dot = vz
    vx_dot = (np.cos(roll) * np.cos(yaw) * np.sin(pitch) + np.sin(roll) * np.sin(yaw)) * thrust_cmd - drag_acc_x
    vy_dot = (np.cos(roll) * np.sin(pitch) * np.sin(yaw) - np.cos(yaw) * np.sin(roll)) * thrust_cmd - drag_acc_y
    vz_dot = -g + np.cos(pitch) * np.cos(roll) * thrust_cmd
    roll_dot = (roll_gain * roll_cmd - roll) / roll_time_constant
    pitch_dot = (pitch_gain * pitch_cmd - pitch) / pitch_time_constant
    yaw_dot = yawrate_cmd
    return np.array([px_dot, py_dot, pz_dot,
                     vx_dot, vy_dot, vz_dot,
                     roll_dot, pitch_dot, yaw_dot])
