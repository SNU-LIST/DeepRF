function m = Pulse_step_22_sar(RF_pulse,b1_range,off_range,gamma,time_step,rf_len,iter_num,sar_weight)

    % This script is for sparse reward,
    % with interpolation and symmetry property
    % Outputs
    % rew_arr: reward array, size is RF_pulse
    % num: sorting number for asynchronous parallel exec.    

    num = iter_num;
    
    RF_pulse_new = RF_pulse;

%     if rf_len ~= 0
%         RF_pulse_new = zeros((rf_len/2),2); % increase sampling rate x 2
%         RF_pulse_new(:,1) = interp1(1:size(RF_pulse,1),RF_pulse(:,1),linspace(1,size(RF_pulse,1),rf_len/2),'linear');
%         RF_pulse_new(:,2) = interp1(1:size(RF_pulse,1),RF_pulse(:,2),linspace(1,size(RF_pulse,1),rf_len/2),'linear');
%     end    
%     RF_pulse_new = [RF_pulse_new; flipud(RF_pulse_new)]; % symmetric porperty
    
    max_rf_amp = max(RF_pulse_new(:,1)) / (2*pi*42.577*1e+6*time_step*1e-4);
    rr = 42.577; % MHz/T
    Gz =  40; % mT/m (fixed)
    pos = abs(off_range(1) / rr / Gz); % mm
    m = zeros(size(b1_range,1),size(off_range,1),3);
    
    for jj=1:size(b1_range,1)
        pulse = [[b1_range(jj)*RF_pulse_new(:,1)./max(RF_pulse_new(:,1))*max_rf_amp; 0],[RF_pulse_new(:,2)./pi*180; 0]]';
        gg = ones(size(RF_pulse_new,1),1) * Gz;
        % gg = [gg; -Gz * size(RF_pulse_new,1) / 2]';
        gg = [gg; 0]';
        [Mx, My, Mz] = Bloch_simul(zeros(size(off_range,1),1),zeros(size(off_range,1),1),ones(size(off_range,1),1),1e+10,1e+10,pulse,gg,time_step*1e+3,pos*1e-3,size(off_range,1));
        m(jj,:,1) = Mx(:,end);
        m(jj,:,2) = My(:,end);
        m(jj,:,3) = Mz(:,end);
    end
    
%     max_rad = 2*pi*42.577*1e+6*time_step*0.2*1e-4; % 0.0280
%     max_sar = max_rad^2*rf_len; % maximum allowed SAR = 0.0280 x 256 = 7.1567
%     cur_sar = sum(RF_pulse_new(:,1).^2);
    
%     temp = mean(m,'all') * ones(1,size(RF_pulse,1));
%     mz_arr = temp(1) * ones(1,size(RF_pulse,1));
%     rew_arr = ((1 - temp(1)) + sar_weight * (1 - (cur_sar / max_sar))) * ones(1,size(RF_pulse,1));

end

