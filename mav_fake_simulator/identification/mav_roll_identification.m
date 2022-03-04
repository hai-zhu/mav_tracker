clear 
clc

%% define system identification model
par_init = [-3.8911; 2.9183];   % initial estimate
optional_args = {};
Ts = 0;
id_model = idgrey('mav_roll_pitch_model', par_init, 'c', optional_args, Ts);

%% load and process data
load input_output_data.mat
idx_span = 100 : 10 : 10100;
id_Ts = 0.1;
id_y = output_pos_rpy(idx_span, 5);
id_u = input_roll_pitch_thrust(idx_span, 2);
id_data = iddata(id_y, id_u, id_Ts);
id_data.InputName = {'roll_cmd'};
id_data.InputUnit = {'rad/s'};
id_data.OutputName = {'roll'};
id_data.OutputUnit = {'rad/s'};
id_data.Tstart = 0;
id_data.TimeUnit = 's';

%% identification
id_opt = greyestOptions; 
id_opt.EnforceStability = 1;
id_sys = greyest(id_data, id_model, id_opt);
roll_time_constant = -1.0/id_sys.A
roll_gain = -id_sys.B/id_sys.A
compare(id_data, id_sys, Inf);