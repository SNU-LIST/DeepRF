%%
clear; close all; clc;

addpath('../../Dependencies');

load('./data/HS_200Hz.mat');

T = 8 * 1e-3; % sec
len = 256; % time points (unit-less)
time_step = T / len; % ms
B1_max = 0.2; % Gauss
steps = 2 * (time_step:time_step:8e-3)./ 8e-3 - 1;

beta = params(2);
max_rf_amp = params(1);
max_freq = params(3);

% HS
am = sech(beta * steps);            
fm = tanh(beta * steps);

RF_pulse_new = zeros(len,2);
RF_pulse_new(:,1) = max_rf_amp * am;
RF_pulse_new(:,2) = -angle(exp(1i * cumsum(2 * pi * max_freq * fm * time_step)));

b1_range = linspace(0.5, 2.0, 151)';
off_range = linspace(-4000, 4000, 2001)';

time_step = 8e-3/256;
max_rf_amp = max(RF_pulse_new(:,1));
rr = 42.577; % MHz/T
Gz =  40; % mT/m (fixed)
pos = abs(off_range(1) / rr / Gz); % mm
m1 = zeros(size(b1_range,1),size(off_range,1));

% SAR calculation
sar1 = sum((RF_pulse_new(:,1)./max(RF_pulse_new(:,1))*max_rf_amp).^2)*time_step*1e+6; % Gauss^2*usec
disp(['HS SAR: ',num2str(sar1)]);

for jj=1:size(b1_range,1)
    [~, ~, Mz] = Bloch_simul(zeros(size(off_range,1),1),zeros(size(off_range,1),1),ones(size(off_range,1),1),1e+10,1e+10,...
        [b1_range(jj)*RF_pulse_new(:,1)./max(RF_pulse_new(:,1))*max_rf_amp,RF_pulse_new(:,2)./pi*180]',Gz,time_step*1e+3,...
        pos*1e-3,size(off_range,1));
    m1(jj,:) = Mz(:,end);
end

load('./data/pulse10000.mat');
[~,ind] = min(loss_arr);
RF_pulse_new2 = squeeze(pulse(ind, :, :))';

max_rad = 2 * pi * 42.5775 * 1e+6 * 8e-3 / 256 * 0.2 * 1e-4;
RF_pulse_new2(:,1) = (RF_pulse_new2(:,1) + 1.0) ./ 2.0 * max_rad;
RF_pulse_new2(:,2) = RF_pulse_new2(:,2) * pi;

time_step = 8e-3/256;
max_rf_amp2 = max(RF_pulse_new2(:,1))  / (2*pi*42.577*1e+6*time_step*1e-4);
rr = 42.577; % MHz/T
Gz =  40; % mT/m (fixed)
pos = abs(off_range(1) / rr / Gz); % mm
m2 = zeros(size(b1_range,1),size(off_range,1));
 
% SAR calculation
sar2 = sum((RF_pulse_new2(:,1)./max(RF_pulse_new2(:,1))*max_rf_amp2).^2)*time_step*1e+6;% Gauss^2*usec
disp(['DeepRF SAR: ',num2str(sar2)]);

for jj=1:size(b1_range,1)
    [~, ~, Mz] = Bloch_simul(zeros(size(off_range,1),1),zeros(size(off_range,1),1),ones(size(off_range,1),1),1e+10,1e+10,...
        [b1_range(jj)*RF_pulse_new2(:,1)./max(RF_pulse_new2(:,1))*max_rf_amp2,RF_pulse_new2(:,2)./pi*180]',Gz,time_step*1e+3,...
        pos*1e-3,size(off_range,1));
    m2(jj,:) = Mz(:,end);
end

disp(['SAR reduction: ',num2str((1-sar2/sar1)*100,'%.4f')]);

%% amplitude shape
figure; plot((time_step:time_step:time_step*double(size(RF_pulse_new,1)))*1e+3, RF_pulse_new(:,1) ./ max(RF_pulse_new(:,1)) * max_rf_amp * 1e+3, 'b', 'LineWidth', 1.5);
hold on; plot((time_step:time_step:time_step*double(size(RF_pulse_new2,1)))*1e+3, RF_pulse_new2(:,1) ./ max(RF_pulse_new2(:,1)) * max_rf_amp2 * 1e+3, 'r', 'LineWidth', 1.5); 
ylim([0 100]); yticks([0:25:100]); xticks([0 2 4 6 8]); xlim([-0.5 8.5]);
legend('HS', 'DeepRF'); set(gca,'FontName','Arial','FontSize',20,'LineWidth',1.5);
ylabel('Amplitude (mG)', 'FontSize', 22); 
xlabel('Time (ms)', 'FontSize', 22);
title('Amplitude shape', 'FontSize', 22);
text(-0.2, 90, ['ENG -',num2str(abs(1-sar2/sar1)*100,'%.0f'),'%'], 'FontSize', 26, 'Color', 'r');


%% phase shape
figure;
ph2 = RF_pulse_new(:,2) / pi * 180;
ph2(48:207) = ph2(48:207) + 360;
ph2 = ph2 - ph2(1);
plot((time_step:time_step:time_step*double(size(RF_pulse_new,1)))*1e+3, ph2, 'b', 'LineWidth', 1.5);
hold on;
ph1 = RF_pulse_new2(:,2)/ pi * 180;
ph1(10:32) = ph1(10:32) - 360; ph1(35:219) = ph1(35:219) - 360; ph1(225:247) = ph1(225:247) - 360; ph1 = ph1 - ph1(1) + 15;
% ph1 = -ph1; ph1(45:212) = ph1(45:212) - 360; ph1 = ph1 - ph1(1) + 5;
% ph1 = -ph1; ph1(1:5) = ph1(1:5) + 360; ph1([35,50,207]) = ph1([35,50,207]) + 360; ph1(252:end) = ph1(252:end) + 360; ph1 = ph1 - ph1(1) + 5;
plot((time_step:time_step:time_step*double(size(RF_pulse_new2,1)))*1e+3, ph1, 'r', 'LineWidth', 1.5);
ylim([-50 450]); xticks([0 2 4 6 8]);  xlim([-0.5 8.5]); yticks(0:100:400);
legend('HS', 'DeepRF'); set(gca,'FontName','Arial','FontSize',20,'LineWidth',1.5);
ylabel('Phase (\circ)', 'FontSize', 22); 
xlabel('Time (ms)', 'FontSize', 22); 
title('Phase shape', 'FontSize', 22);


%% simulated profile (HS)
figure; imagesc(off_range, b1_range, m1, [-1 1]);
xticks([-4000:2000:4000]); ylim([0.5 2.0]); xlim([-4000 4000]);
set(gca,'FontName','Arial','FontSize',20,'LineWidth',1.5);
ylabel('B_1', 'FontSize', 22); 
xlabel('Frequency (Hz)', 'FontSize', 22); 
title('Simulated profile (HS)', 'FontSize', 22); 


%% simulated profile (HS, magnified)
figure; imagesc(off_range, b1_range, m1, [-1 1]);
xticks([-600:300:600]); ylim([0.5 2.0]); xlim([-600 600]);
set(gca,'FontName','Arial','FontSize',20,'LineWidth',1.5);
ylabel('B_1', 'FontSize', 22); 
xlabel('Frequency (Hz)', 'FontSize', 22); 
title('Simulated profile (HS)', 'FontSize', 22); 
rectangle('Position', [-200, 0.52, 400, 1.46], 'EdgeColor', 'r', 'LineWidth', 2.5, 'LineStyle', '--');
text(-150, 1.7, '\fontsize{18}Mean M\fontsize{14}z', 'Color', 'r');
text(-105, 1.85, '-0.906', 'FontSize', 18, 'Color', 'r');


%% simulated profile (DeepRF)
figure; imagesc(off_range, b1_range, m2, [-1 1]);
xticks([-4000:2000:4000]); ylim([0.5 2.0]); xlim([-4000 4000]);
set(gca,'FontName','Arial','FontSize',20,'LineWidth',1.5);
ylabel('B_1', 'FontSize', 22); 
xlabel('Frequency (Hz)', 'FontSize', 22);
title('Simulated profile (DeepRF)', 'FontSize', 22); 


%% simulated profile (DeepRF, magnified)
figure; imagesc(off_range, b1_range, m2, [-1 1]);
xticks([-600:300:600]); ylim([0.5 2.0]); xlim([-600 600]);
set(gca,'FontName','Arial','FontSize',20,'LineWidth',1.5);
ylabel('B_1', 'FontSize', 22); 
xlabel('Frequency (Hz)', 'FontSize', 22);
title('Simulated profile (DeepRF)', 'FontSize', 22); 
rectangle('Position', [-200, 0.52, 400, 1.46], 'EdgeColor', 'r', 'LineWidth', 2.5, 'LineStyle', '--');
text(-150, 1.7, '\fontsize{18}Mean M\fontsize{14}z', 'Color', 'r');
text(-105, 1.85, '-0.904', 'FontSize', 18, 'Color', 'r');


%% mean inversion values
b1_range = linspace(0.5, 2.0, 151)';
off_range = linspace(-200, 200, 401)';

time_step = 8e-3/256;
max_rf_amp = max(RF_pulse_new(:,1));
rr = 42.577; % MHz/T
Gz =  40; % mT/m (fixed)
pos = abs(off_range(1) / rr / Gz); % mm
m1 = zeros(size(b1_range,1),size(off_range,1));

for jj=1:size(b1_range,1)
    [~, ~, Mz] = Bloch_simul(zeros(size(off_range,1),1),zeros(size(off_range,1),1),ones(size(off_range,1),1),1e+10,1e+10,...
        [b1_range(jj)*RF_pulse_new(:,1)./max(RF_pulse_new(:,1))*max_rf_amp,RF_pulse_new(:,2)./pi*180]',Gz,time_step*1e+3,...
        pos*1e-3,size(off_range,1));
    m1(jj,:) = Mz(:,end);
end

disp(['HS Mz: ', num2str(mean(m1,'all'))]);

time_step = 8e-3/256;
max_rf_amp2 = max(RF_pulse_new2(:,1))  / (2*pi*42.577*1e+6*time_step*1e-4);
rr = 42.577; % MHz/T
Gz =  40; % mT/m (fixed)
pos = abs(off_range(1) / rr / Gz); % mm
m2 = zeros(size(b1_range,1),size(off_range,1));

for jj=1:size(b1_range,1)
    [~, ~, Mz] = Bloch_simul(zeros(size(off_range,1),1),zeros(size(off_range,1),1),ones(size(off_range,1),1),1e+10,1e+10,...
        [b1_range(jj)*RF_pulse_new2(:,1)./max(RF_pulse_new2(:,1))*max_rf_amp2,RF_pulse_new2(:,2)./pi*180]',Gz,time_step*1e+3,...
        pos*1e-3,size(off_range,1));
    m2(jj,:) = Mz(:,end);
end

disp(['DeepRF Mz: ', num2str(mean(m2,'all'))]);