function m = visualize_non_selective_pulse_7(RF_pulse, time_step, b1, off)
    
    max_rf_amp = max(RF_pulse(:,1)) / (2*pi*42.577*1e+6*time_step*1e-4);
    rr = 42.577; % MHz/T
    Gz =  40; % mT/m (fixed)
    pos = abs(off / rr / Gz); % mm
    m = zeros(size(RF_pulse,1),3);
    
    [Mx, My, Mz] = Bloch_simul(0,0,1,1e+10,1e+10,...
        [b1*RF_pulse(:,1)./max(RF_pulse(:,1))*max_rf_amp,RF_pulse(:,2)./pi*180]',Gz,time_step*1e+3,...
        pos*1e-3,1);
    
    m = [Mx; My; Mz]';
    
end