function [adia_mat, alpha_mat] = plot_adiabacity_2(rf, ts, off, b1)

    % analyze adiabatic pulse behavior
    %
    % inputs
    % rf: (# points, 2) 
    % where ( ,1) = amplitude (radians), ( ,2) = phase (raidans)
    % ts: time-interval for one-point (seconds)
    % off: off-resonance freq. devation of B1-field (Hz)
    % b1: B1 scale factor (A.U.)
    %
    % ouputs
    % adia_mat: adiabaticity for each time-step
    % alpha_mat: alpha of effective B-field for each time-step

    gamma = 2*pi*42.577*1e+6; % rad/sec/T
    
    adia_mat = []; % adiabaticity (K)
    Be_mat = []; % | effective B-field |
    dalpha_mat = []; % angular velocty of B_eff (rad/sec)
    
    delta_omega_mat = zeros(1,size(rf,1)-2); % B1 freq. deviation
    alpha_mat = zeros(1,size(rf,1)-2); % angle bettwen B_eff field and z-axis
    
    for ii=2:size(rf,1)-1
        % delta_omega = angle(exp(1i*(rf(ii,2) - rf(ii-1,2)))) / ts; % first-order finite difference (rad/sec)
        delta_omega = angle(exp(1i*(rf(ii,2) - rf(ii-1,2)))) / ts + 2*pi*off;
        alpha_mat(ii-1) = atan((rf(ii,1)*b1/ts)/delta_omega); % angle between B_eff & z-axis
        delta_omega_mat(ii-1) = delta_omega;
    end
    alpha_mat(alpha_mat<0) = alpha_mat(alpha_mat<0) + pi; % [-pi,pi] -> [0, 2*pi]
    
    for ii=2:size(alpha_mat,2)-1

        diff_alpha = (alpha_mat(ii) - alpha_mat(ii-1)) / ts + 1e-8;
        dalpha_mat = [dalpha_mat, diff_alpha]; % angular velocity of B_eff
        Be = sqrt((rf(ii+1,1)*b1/ts/gamma)^2 + (delta_omega_mat(ii)/gamma)^2); % |B_eff|
        % (rad/sec/gyromagnetic_ratio)  +   (rad/sec/gyromagnetic_ratio)  =  Tesla
        adia_mat = [adia_mat, abs((gamma*Be)/diff_alpha)]; % adiabaticity (K)
        Be_mat = [Be_mat, abs(gamma*Be)];
        
    end
    
end

