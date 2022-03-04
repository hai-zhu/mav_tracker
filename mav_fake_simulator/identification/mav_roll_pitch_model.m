function [A, B, C, D] = mav_roll_pitch_model(par, T)

    % x: roll (pitch)
    % u: roll_cmd (pitch_cmd)
    % dynamics:
    %   roll_dot = param_roll_tau*roll + param_roll_k*roll_cmd
    % y: roll
    % to translate: 
    %   roll_time_constant = -1/param_roll_tau
    %   roll_gain = -param_roll_k/param_roll_tau
    
    param_roll_tau = par(1);
    param_roll_k = par(2);
    
    A = param_roll_tau;
    B = param_roll_k;
    C = 1;
    D = 0;

end 
