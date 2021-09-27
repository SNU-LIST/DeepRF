clear; close all; clc;

addpath('../../Dependencies');

load('./data/inv_du512.txt');

inv1 = inv_du512;

time_step = 5.12e-3/256; % sec
to_gauss = 2 * pi * 42.5775 * 1e+6 * time_step * 1e-4; % rad to gauss
inv1(:,1) = inv1(:,1) * 2 * pi * 42.5775 * time_step * 1e+3;
inv1(:,2) = inv1(:,2) / 180 * pi;

sar1 = sum((inv1(:,1)./to_gauss).^2)*time_step*1e+6;
disp(['SLR SAR: ',num2str(sar1)]);

RF_pulse = inv1;
b1_range = linspace(1.0,1.0,2)';
off_range = linspace(-8000,8000,16001)';
gamma = 1.0;
rf_len = 256;
iter_num = 0;
sar_weight = 0.0;

num = iter_num;
RF_pulse_new = RF_pulse;

max_rf_amp = max(RF_pulse_new(:,1)) / (2*pi*42.577*1e+6*time_step*1e-4);
rr = 42.577; % MHz/T
Gz =  40; % mT/m (fixed)
pos = abs(off_range(1) / rr / Gz); % mm
m1 = zeros(size(b1_range,1),size(off_range,1),259,3);

for jj=1:size(b1_range,1)
    pulse = [[0;b1_range(jj)*RF_pulse_new(:,1)./max(RF_pulse_new(:,1))*max_rf_amp;0],[0;RF_pulse_new(:,2)./pi*180;0]]';
    gg = ones(size(RF_pulse_new,1),1) * Gz;
    gg = [0; gg; 0]';
    [Mx, My, Mz] = Bloch_simul(zeros(size(off_range,1),1),zeros(size(off_range,1),1),ones(size(off_range,1),1),1e+10,1e+10,pulse,gg,time_step*1e+3,pos*1e-3,size(off_range,1));
    m1(jj,:,:,1) = Mx;
    m1(jj,:,:,2) = My;
    m1(jj,:,:,3) = Mz;
end

num = 51164;
load('./data/pulse10000');
inv2 = squeeze(pulse(mod(num,256), :, :))';

inv2(:,1) = (inv2(:,1) + 1.0) / 2.0 * 0.2 * 1e-4 * 2 * pi * 42.5775 * time_step * 1e+6;
inv2(:,2) = inv2(:,2) * pi;

to_gauss = 2 * pi * 42.5775 * 1e+6 * time_step * 1e-4; % rad to gauss
sar2 = sum((inv2(:,1)./to_gauss).^2)*time_step*1e+6;
disp(['DeepRF SAR: ',num2str(sar2)]);

RF_pulse = inv2;

num = iter_num;
RF_pulse_new = RF_pulse;

max_rf_amp = max(RF_pulse_new(:,1)) / (2*pi*42.577*1e+6*time_step*1e-4);
rr = 42.577; % MHz/T
Gz =  40; % mT/m (fixed)
pos = abs(off_range(1) / rr / Gz); % mm
m2 = zeros(size(b1_range,1),size(off_range,1),259,3);

for jj=1:size(b1_range,1)
    pulse = [[0;b1_range(jj)*RF_pulse_new(:,1)./max(RF_pulse_new(:,1))*max_rf_amp;0],[0;RF_pulse_new(:,2)./pi*180;0]]';
    gg = ones(size(RF_pulse_new,1),1) * Gz;
    gg = [0; gg; 0]';
    [Mx, My, Mz] = Bloch_simul(zeros(size(off_range,1),1),zeros(size(off_range,1),1),ones(size(off_range,1),1),1e+10,1e+10,pulse,gg,time_step*1e+3,pos*1e-3,size(off_range,1));
    m2(jj,:,:,1) = Mx;
    m2(jj,:,:,2) = My;
    m2(jj,:,:,3) = Mz;
end


%% change of slice profile (static images)
for ii = 32:32:256
    figure; plot(off_range, squeeze(m1(2,:,ii+1,3)), 'b', 'LineWidth', 1.5); 
    hold on; plot(off_range, squeeze(m2(2,:,ii+1,3)), 'r', 'LineWidth', 1.5); 
    ylim([-1.1 1.5]); xlim([-2000 2000]);
    set(gca,'FontName','Arial','FontSize',20,'LineWidth',1.5);
    xlabel('Frequency (Hz)', 'FontSize', 22); 
    ylabel('Signal (A.U.)', 'FOntSize', 22); 
    legend('\fontsize{18}M\fontsize{14}z \fontsize{18}(SLR)', '\fontsize{18}M\fontsize{14}z \fontsize{18}(DeepRF)', 'Location', 'north', 'NumColumns', 2);
    title(['Time = ',num2str(ii*0.02),' ms'], 'FontSize', 22);
end

