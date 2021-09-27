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

n_f_p = find(~(off_range+250));
f_p = find(~(off_range-250));
n_f_s = find(~(off_range+(250+336)));
f_s = find(~(off_range-(250+336)));
n_f_b = find(~(off_range+(250+336/2)));
f_b = find(~(off_range-(250+336/2)));

m1 = SLR_inv_full_simul(RF_pulse,b1_range,off_range,gamma,time_step,rf_len,iter_num,sar_weight);
temp = sum(m1(2,n_f_b:f_b,3))/(f_b - n_f_b + 1);
disp(['SLR bandwidth: ',num2str(temp)]);
temp = sum(m1(2,n_f_p:f_p,3))/(f_p - n_f_p + 1);
disp(['SLR passband: ',num2str(temp)]);
temp = sum(m1(2,n_f_s:f_s,3))/(f_s - n_f_s + 1);
disp(['SLR BW+trans: ',num2str(temp)]);
temp2 = abs(1-m1(2,1:n_f_s,3));
temp3 = abs(1-m1(2,f_s:end,3));
disp(['SLR stopband ripple: ',num2str(max(temp2)),' ',num2str(max(temp3))]);

load('./data/pulse10000');
[~, ind] = min(loss_arr);
inv2 = squeeze(pulse(ind, :, :))';

inv2(:,1) = (inv2(:,1) + 1.0) / 2.0 * 0.2 * 1e-4 * 2 * pi * 42.5775 * time_step * 1e+6;
inv2(:,2) = inv2(:,2) * pi;

to_gauss = 2 * pi * 42.5775 * 1e+6 * time_step * 1e-4; % rad to gauss
sar2 = sum((inv2(:,1)./to_gauss).^2)*time_step*1e+6;
disp(['DeepRF SAR: ',num2str(sar2)]);

RF_pulse = inv2;

m2 = SLR_inv_full_simul(RF_pulse,b1_range,off_range,gamma,time_step,rf_len,iter_num,sar_weight);
temp = sum(m2(2,n_f_b:f_b,3))/(f_b - n_f_b + 1);
disp(['DeepRF bandwidth: ',num2str(temp)]);
temp = sum(m2(2,n_f_p:f_p,3))/(f_p - n_f_p + 1);
disp(['DeepRF passband: ',num2str(temp)]);
temp = sum(m2(2,n_f_s:f_s,3))/(f_s - n_f_s + 1);
disp(['DeepRF BW+trans: ',num2str(temp)]);
temp2 = abs(1-m2(2,1:n_f_s,3));
temp3 = abs(1-m2(2,f_s:end,3));
disp(['DeepRF stopband ripple: ',num2str(max(temp2)),' ',num2str(max(temp3))]);

disp(['SAR reduction: ',num2str((1-sar2/sar1)*100,'%.4f')]);


%% simulated profile
figure; plot(off_range,squeeze(m1(2,:,3)),'b','LineWidth',1.5); 
hold on; plot(off_range,squeeze(m2(2,:,3)),'r','LineWidth',1.5); 
xlim([off_range(1) off_range(end)]); xticks(-8000:4000:8000);
ylim([-1.1 1.1]); 
set(gca,'FontName','Arial','FontSize',20,'LineWidth',1.5);
xlabel('Frequency (Hz)', 'FontSize', 22); 
ylabel('Signal (A.U.)', 'FontSize', 22); 
title('Simulated slice profile', 'FontSize', 22); 
legend('\fontsize{16}M\fontsize{12}z\fontsize{16} (SLR)', '\fontsize{16}M\fontsize{12}z\fontsize{16} (DeepRF)', 'Location', 'southeast');


%% simulated profile (magnified)
figure; plot(off_range,squeeze(m1(2,:,3)),'b','LineWidth',1.5); 
hold on; plot(off_range,squeeze(m2(2,:,3)),'r','LineWidth',1.5); 
xlim([off_range(n_f_s) off_range(f_s)]); 
xticks(int64(linspace(off_range(n_f_s), off_range(f_s), 5))); 
ylim([-1.1 1.1]); 
set(gca,'FontName','Arial','FontSize',20,'LineWidth',1.5);
xlabel('Frequency (Hz)', 'FontSize', 22); 
ylabel('Signal (A.U.)', 'FontSize', 22); 
title('Simulated slice profile (zoom-in)', 'FontSize', 22); 
legend('\fontsize{18}M\fontsize{14}z\fontsize{18} (SLR)', '\fontsize{18}M\fontsize{14}z\fontsize{18} (DeepRF)', 'Location', 'north');

text(-730 * (586/1570), -0.4, '\fontsize{18}Mean M\fontsize{14}z\fontsize{18} over BW ', 'FontSize', 18);
text(-550 * (586/1570), -0.6, '-0.81', 'FontSize', 18, 'Color', 'b');
text(-80 * (586/1570), -0.6, 'vs.', 'FontSize', 18);
text(200 * (586/1570), -0.6, '-0.81', 'FontSize', 18, 'Color', 'r');


%% amplitude plot
time_step = 5.12/256;
figure; plot(time_step:time_step:time_step*256,inv1(:,1)./to_gauss*1e+3,'b','LineWidth',1.5);
hold on; plot(time_step:time_step:time_step*256,inv2(:,1)./to_gauss*1e+3,'r','LineWidth',1.5);
xlim([0-0.4, 5.12+0.4]); xticks(0:1:5); ylim([0 200]);
set(gca,'FontName','Arial','FontSize',20,'LineWidth',1.5);
ylabel('Amplitude (mG)', 'FontSize', 22); 
xlabel('Time (ms)', 'FontSize', 22);
title('Amplitude shape', 'FontSize', 22);
legend('SLR', 'DeepRF', 'FontSize', 20);
text(-0.2, 180, ['ENG -',num2str((1-sar2/sar1)*100,'%.0f'),'%'], 'FontSize', 26, 'Color', 'r');


%% phase plot
figure; plot(time_step:time_step:time_step*256,inv1(:,2)./pi*180,'b','LineWidth',1.5);
hold on; plot(time_step:time_step:time_step*256,angle(exp(1i*inv2(:,2)))./pi*180,'r','LineWidth',1.5);
xlim([0-0.4, 5.12+0.4]); xticks(0:1:5); ylim([-230 330]);
yticks(-180:90:180);
set(gca,'FontName','Arial','FontSize',20,'LineWidth',1.5);
ylabel('Phase (\circ)', 'FontSize', 22); 
xlabel('Time (ms)', 'FontSize', 22);
title('Phase shape', 'FontSize', 22);
legend('SLR', 'DeepRF', 'FontSize', 20);


%% stopband ripple analysis
figure; plot(off_range, log10(squeeze(abs(1-m1(2,:,3)))), 'b', 'LineWidth', 1.5);
hold on; plot(off_range, log10(squeeze(abs(1-m2(2,:,3)))), 'r', 'LineWidth', 1.5);
xlim([-8000 8000]);
set(gca,'FontName','Arial','FontSize',20,'LineWidth',1.5);
yticks(-8:2:0); title('\fontsize{22}Log plot of |1-M\fontsize{18}z\fontsize{22}|');
yticklabels({'10^{-8}','10^{-6}','10^{-4}','10^{-2}','10^0'});
ylabel('Signal (A.U.)', 'FontSize', 22); xlabel('Frequency (Hz)', 'FontSize', 22);
ylim([-8 0]); xticks(-8000:4000:8000);
legend('SLR', 'DeepRF', 'FontSize', 18);

r1 = max(max(abs(1-m1(2,1:7371,3))),max(abs(1-m1(2,8631:end,3))))*100;
r2 = max(max(abs(1-m2(2,1:7371,3))),max(abs(1-m2(2,8631:end,3))))*100;

text(-6500, -1.0, 'Max. ripple', 'FontSize', 18);
text(-7000, -1.8, [num2str(r1,'%0.1f'),'%'], 'FontSize', 18, 'Color', 'b');
text(-4700, -1.8, 'vs.', 'FontSize', 18);
text(-3400, -1.8, [num2str(r2,'%0.1f'),'%'], 'FontSize', 18, 'Color', 'r');

%% passband ripple analysis
figure; plot(off_range, log10(squeeze(abs(1+m1(2,:,3)))), 'b', 'LineWidth', 1.5);
hold on; plot(off_range, log10(squeeze(abs(1+m2(2,:,3)))), 'r', 'LineWidth', 1.5);
xlim([off_range(n_f_b), off_range(f_b)]);
set(gca,'FontName','Arial','FontSize',20,'LineWidth',1.5);
yticks(-5:0); title('\fontsize{22}Log plot of |1+M\fontsize{18}z\fontsize{22}|', 'FontSize', 22);
yticklabels({'10^{-5}','10^{-4}','10^{-3}','10^{-2}','10^1', '10^0'});
ylabel('Signal (A.U.)', 'FontSize', 22); xlabel('Frequency (Hz)', 'FontSize', 22);
ylim([-5 log10(abs(1+m1(2,n_f_b,3)))]); xticks(linspace(off_range(n_f_b), off_range(f_b), 5));
legend('SLR', 'DeepRF', 'FontSize', 18, 'Location', 'South');

rr1 = max(abs(1+m1(2,7801:8201,3)))*100;
rr2 = max(abs(1+m2(2,7801:8201,3)))*100;

text(-120, -0.6, 'Max. ripple', 'FontSize', 18);
text(-150, -1.1, [num2str(rr1,'%0.1f'),'%'], 'FontSize', 18, 'Color', 'b');
text(-30, -1.1, 'vs.', 'FontSize', 18);
text(40, -1.1, [num2str(rr2,'%0.1f'),'%'], 'FontSize', 18, 'Color', 'r');


