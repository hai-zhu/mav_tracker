clear 
clc 

addpath(genpath(pwd))


%% load bag
bag = rosbag('sim-identification2022-02-19-01-09-16.bag');
% rosbag info 'sim-identification2022-02-19-01-09-16.bag'
bag_pose = select(bag, 'Topic', '/mavros/local_position/pose');
bag_target_att = select(bag, 'Topic', '/mavros/setpoint_raw/target_attitude');
msgs_pose = readMessages(bag_pose, 'DataFormat','struct');
msgs_target_att = readMessages(bag_target_att, 'DataFormat','struct');

%% read to data
% pose
output_pos_rpy = zeros(length(msgs_pose), 7);
for i = 1 : length(msgs_pose)
    output_pos_rpy(i, 1) = stamp_to_seconds(msgs_pose{i}.Header.Stamp);
    output_pos_rpy(i, 2) = msgs_pose{i}.Pose.Position.X;
    output_pos_rpy(i, 3) = msgs_pose{i}.Pose.Position.Y;
    output_pos_rpy(i, 4) = msgs_pose{i}.Pose.Position.Z;
    quat_temp = [msgs_pose{i}.Pose.Orientation.W, ...
                 msgs_pose{i}.Pose.Orientation.X, ...
                 msgs_pose{i}.Pose.Orientation.Y, ...
                 msgs_pose{i}.Pose.Orientation.Z];
    rpy_temp = quat2eul(quat_temp, 'XYZ');
    output_pos_rpy(i, 5) = rpy_temp(1);
    output_pos_rpy(i, 6) = rpy_temp(2);
    output_pos_rpy(i, 7) = rpy_temp(3);
end
% command
input_roll_pitch_thrust = zeros(length(msgs_target_att), 4);
for i = 1 : length(msgs_target_att)
    input_roll_pitch_thrust(i, 1) = stamp_to_seconds(msgs_target_att{i}.Header.Stamp);
    quat_temp = [msgs_target_att{i}.Orientation.W, ...
                 msgs_target_att{i}.Orientation.X, ...
                 msgs_target_att{i}.Orientation.Y, ...
                 msgs_target_att{i}.Orientation.Z];
    rpy_temp = quat2eul(quat_temp, 'XYZ');
    input_roll_pitch_thrust(i, 2) = rpy_temp(1);
    input_roll_pitch_thrust(i, 3) = rpy_temp(2);
    input_roll_pitch_thrust(i, 4) = msgs_target_att{i}.Thrust;
end

save('input_output_data.mat', 'input_roll_pitch_thrust', 'output_pos_rpy')

