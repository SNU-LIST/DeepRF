%% HS
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

am = sech(beta * steps);            
fm = tanh(beta * steps) ./ tanh(beta);

RF_pulse_new = zeros(len,2);
RF_pulse_new(:,1) = max_rf_amp * am;
RF_pulse_new(:,2) = -angle(exp(1i * cumsum(2 * pi * max_freq * fm * time_step)));

time_step = 8e-3/256;
B1 = 1.0;
OFF = 0;

pul_len = size(RF_pulse_new,1);
[a,~] = plot_adiabacity_2(RF_pulse_new, time_step, OFF, B1);
c = visualize_non_selective_pulse_7(RF_pulse_new, time_step, B1, OFF);
c = c(2:end,:);
tt = linspace(0,pul_len,3);

figure; plot(3:pul_len-2, a, 'k', 'LineWidth', 1.5); ylim([0 40]);
set(gca,'FontName','Arial','FontSize',20,'LineWidth',1.5);
xlabel('Times (ms)', 'FontSize', 22); 
ylabel('Adiabaticity (K)', 'FontSize', 22);
t = linspace(0,pul_len,3); xticks(tt); xticklabels([0,4,8]); hold on;
xlim([0 pul_len]); yticks(0:20:40);
[av, ~] = min(a);
hold on; plot(3:pul_len-2, av*ones(1,252), 'r--', 'LineWidth', 1.5);
legend('K', 'K = 3.6');
title('Adiabaticity (HS)', 'FontSize', 26);

figure;
scatter3(c(:,1),c(:,2),c(:,3),40,linspace(1,10,pul_len),'filled')
xlim([-1 1]); ylim([-1 1]); zlim([-1 1]);
set(gca,'FontName','Arial','FontSize',20,'LineWidth',1.5);
view(45+90,45);
xlabel('M_x', 'FontSize', 22); 
ylabel('M_y', 'FontSize', 22); 
zlabel('M_z', 'FontSize', 22);
title('Spin trajectory (HS)', 'FontSize', 26);


%% DeepRF
clear; clc;

load('./data/pulse10000.mat');
RF_pulse_new = best_RF';

max_rad = 2 * pi * 42.5775 * 1e+6 * 8e-3 / 256 * 0.2 * 1e-4;
RF_pulse_new(:,1) = (RF_pulse_new(:,1) + 1.0) ./ 2.0 * max_rad;
RF_pulse_new(:,2) = RF_pulse_new(:,2) * pi;

time_step = 8e-3/256;
B1 = 1.0;
OFF = 0;

pul_len = size(RF_pulse_new,1);
[a,~] = plot_adiabacity_2(RF_pulse_new, time_step, OFF, B1);
c = visualize_non_selective_pulse_7(RF_pulse_new, time_step, B1, OFF);
c = c(2:end,:);

figure; plot(3:pul_len-2, a, 'k', 'LineWidth', 1.5); ylim([0 40]);
tt = linspace(0,pul_len,3); xticks(tt); 
set(gca,'FontName','Arial','FontSize',20,'LineWidth',1.5);
xlabel('Times (ms)', 'FontSize', 22); 
ylabel('Adiabaticity (K)', 'FontSize', 22);
t = linspace(0,pul_len,3); xticks(tt); xticklabels([0,4,8]); hold on;
xlim([0 pul_len]); yticks(0:20:40); 
av = 3.6;
hold on; plot(3:pul_len-2, av*ones(1,252), 'r--', 'LineWidth', 1.5);
legend('K', 'K = 3.6');
title('Adiabaticity (DeepRF)', 'FontSize', 26);

figure;
scatter3(c(:,1),c(:,2),c(:,3),40,linspace(1,10,pul_len),'filled')
xlim([-1 1]); ylim([-1 1]); zlim([-1 1]);
set(gca,'FontName','Arial','FontSize',20,'LineWidth',1.5);
view(45+90,45);
xlabel('M_x', 'FontSize', 22); 
ylabel('M_y', 'FontSize', 22); 
zlabel('M_z', 'FontSize', 22);
title('Spin trajectory (DeepRF)', 'FontSize', 26);


%% HS
clear; clc; 

load('./data/HS_200Hz.mat');

T = 8 * 1e-3; % sec
len = 256; % time points (unit-less)
time_step = T / len; % ms
B1_max = 0.2; % Gauss
steps = 2 * (time_step:time_step:8e-3)./ 8e-3 - 1;

beta = params(2);
max_rf_amp = params(1);
max_freq = params(3);

am = sech(beta * steps);            
fm = tanh(beta * steps) ./ tanh(beta);

RF_pulse_new = zeros(len,2);
RF_pulse_new(:,1) = max_rf_amp * am;
RF_pulse_new(:,2) = -angle(exp(1i * cumsum(2 * pi * max_freq * fm * time_step)));

time_step = 8e-3/256;
B1 = 1.5;
OFF = 150;

pul_len = size(RF_pulse_new,1);
[a,~] = plot_adiabacity_2(RF_pulse_new, time_step, OFF, B1);
tt = linspace(0,pul_len,3);
c = visualize_non_selective_pulse_7(RF_pulse_new, time_step, B1, OFF);
c = c(2:end,:);

figure; plot(3:pul_len-2, a, 'k', 'LineWidth', 1.5); ylim([0 40]);
set(gca,'FontName','Arial','FontSize',20,'LineWidth',1.5);
xlabel('Times (ms)', 'FontSize', 22); 
ylabel('Adiabaticity (K)', 'FontSize', 22);
t = linspace(0,pul_len,3); xticks(tt); xticklabels([0,4,8]); hold on;
xlim([0 pul_len]); yticks(0:20:40); 
[av, ai] = min(a);
hold on; plot(3:pul_len-2, av*ones(1,252), 'r--', 'LineWidth', 1.5);
legend('K', 'K = 3.1');
title('Adiabaticity (HS)', 'FontSize', 26);

figure;
scatter3(c(:,1),c(:,2),c(:,3),40,linspace(1,10,pul_len),'filled')
xlim([-1 1]); ylim([-1 1]); zlim([-1 1]);
set(gca,'FontName','Arial','FontSize',20,'LineWidth',1.5);
view(45+90,45);
xlabel('M_x', 'FontSize', 22); 
ylabel('M_y', 'FontSize', 22); 
zlabel('M_z', 'FontSize', 22);
title('Spin trajectory (HS)', 'FontSize', 26);


%% DeepRF
clear; clc; 

load('./data/pulse10000.mat');
RF_pulse_new = best_RF';

max_rad = 2 * pi * 42.5775 * 1e+6 * 8e-3 / 256 * 0.2 * 1e-4;
RF_pulse_new(:,1) = (RF_pulse_new(:,1) + 1.0) ./ 2.0 * max_rad;
RF_pulse_new(:,2) = RF_pulse_new(:,2) * pi;

time_step = 8e-3/256;
B1 = 1.5;
OFF = 150;

pul_len = size(RF_pulse_new,1);
[a,b] = plot_adiabacity_2(RF_pulse_new, time_step, OFF, B1);
c = visualize_non_selective_pulse_7(RF_pulse_new, time_step, B1, OFF);
c = c(2:end,:);
tt = linspace(0,pul_len,3); 

av = 3.1;
figure; plot(3:pul_len-2, a, 'k', 'LineWidth', 1.5); ylim([0 40]);
set(gca,'FontName','Arial','FontSize',20,'LineWidth',1.5);
xlabel('Times (ms)', 'FontSize', 22); 
ylabel('Adiabaticity (K)', 'FontSize', 22);
t = linspace(0,pul_len,3); xticks(tt); xticklabels([0,4,8]); hold on;
xlim([0 pul_len]); yticks(0:20:40); 
hold on; plot(3:pul_len-2, av*ones(1,252), 'r--', 'LineWidth', 1.5);
legend('K', 'K = 3.1');
title('Adiabaticity (DeepRF)', 'FontSize', 26);

figure;
scatter3(c(:,1),c(:,2),c(:,3),40,linspace(1,10,pul_len),'filled')
xlim([-1 1]); ylim([-1 1]); zlim([-1 1]);
set(gca,'FontName','Arial','FontSize',20,'LineWidth',1.5);
view(45+90,45);
xlabel('M_x', 'FontSize', 22); 
ylabel('M_y', 'FontSize', 22); 
zlabel('M_z', 'FontSize', 22);
title('Spin trajectory (DeepRF)', 'FontSize', 26);



