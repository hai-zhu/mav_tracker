clear 
clc 

%% define system identification model
Order         = [5 3 8];               % Model orders [ny nu nx].
Parameters    = [0.3; 1.0; 0.3; 1.0; ...
                 0.01; 0.01];          % Initial parameter vector.
InitialStates = [];                    % Initial initial states.
Ts = 0;                                % Time-continuous system.
id_model    = idnlgrey('mav_full_sys_model', Order, Parameters, [], 0);


%% load and process data
load input_output_data.mat
idx_span = 100 : 10 : 10100;
id_Ts = 0.1;
id_y = output_pos_rpy(idx_span, [2,3,4,5,6]);
id_u = input_roll_pitch_thrust(idx_span, [2,3,4]);
for i = 1 : length(id_u)
    id_u(i, 3) = id_u(i, 3) * 20.0 / 1.56;
end
id_data = iddata(id_y, id_u, id_Ts);
id_data.Tstart = 0;
id_data.TimeUnit = 's';
id_opt = nlgreyestOptions; 
id_opt.Display = 'on';
id_opt.SearchOption.MaxIter = 50;
id_sys = nlgreyest(id_data, id_model, id_opt);
