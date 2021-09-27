clear; clc; close all;

addpath('../../Dependencies');

load('./data/exc_iteration91_sar4.3636.mat');

RF_pulse_new = zeros(length(w1),2);
RF_pulse_new(:,1) = abs(w1);
RF_pulse_new(:,2) = (w1<0)*pi;
    
exc = RF_pulse_new;
exc2 = exc;

time_step = 2.56e-3 / 256;
to_gauss = 2 * pi * 42.5775 * 1e+6 * time_step * 1e-4; % rad to gauss
sar2 = sum((exc(:,1)./to_gauss).^2)*time_step*1e+6;
disp(['OC SAR: ',num2str(sar2)]);

RF_pulse = exc;
b1_range = linspace(1.0,1.0,2)';
off_range = linspace(-32000,32000,64001)';
gamma = 1.0;
rf_len = 256;
iter_num = 0;
sar_weight = 0.0;

n_f_p = find(~(off_range+1000));  % -1000 Hz
f_p = find(~(off_range-1000));  % 1000 Hz
n_f_s = find(~(off_range+1570));  % -1570 Hz
f_s = find(~(off_range-1570));  % 1570 Hz
n_f_b = find(~(off_range+(1000+570/2))); % -1000-570/2 Hz
f_b = find(~(off_range-(1000+570/2))); % 1000+570/2 Hz

m2 = SLR_exc_full_simul(RF_pulse,b1_range,off_range,gamma,time_step,rf_len,iter_num,sar_weight);
temp = sqrt(sum(m2(2,n_f_b:f_b,1))^2+sum(m2(2,n_f_b:f_b,2))^2)/(f_b - n_f_b + 1);
disp(['OC bandwidth: ',num2str(temp,'%.4f')]);
temp = sum(sqrt((m2(2,n_f_b:f_b,1)).^2+(m2(2,n_f_b:f_b,2)).^2))/(f_b - n_f_b + 1);
disp(['OC bandwidth2: ',num2str(temp,'%.4f')]);
temp = sqrt(sum(m2(2,n_f_p:f_p,1))^2+sum(m2(2,n_f_p:f_p,2))^2)/(f_p - n_f_p + 1);
disp(['OC passband: ',num2str(temp,'%.4f')]);
temp = sqrt(sum(m2(2,n_f_s:f_s,1))^2+sum(m2(2,n_f_s:f_s,2))^2)/(f_s - n_f_s + 1);
disp(['OC BW+trans: ',num2str(temp,'%.4f')]);
temp2 = sqrt((m2(2,1:n_f_s,1)).^2+(m2(2,1:n_f_s,2)).^2);
temp3 = sqrt((m2(2,f_s:end,1)).^2+(m2(2,f_s:end,2)).^2);
disp(['OC stopband ripple: ',num2str(max(temp2)),' ',num2str(max(temp3))]);
temp4 = mean(m2(2,n_f_p:f_p,2));
disp(['OC My signal: ',num2str(temp4,'%.4f')]);
temp5 = mean(m2(2,n_f_p:f_p,1));
disp(['OC Mx signal: ',num2str(temp5,'%.4f')]);


RF_pulse_new = zeros(length(slr),2);
RF_pulse_new(:,1) = abs(slr);
RF_pulse_new(:,2) = (slr<0)*pi;
    
exc = RF_pulse_new;
exc1 = exc;

to_gauss = 2 * pi * 42.5775 * 1e+6 * time_step * 1e-4; % rad to gauss
sar1 = sum((exc(:,1)./to_gauss).^2)*time_step*1e+6;
disp(['SLR SAR: ',num2str(sar1)]);

RF_pulse = exc;

m1 = SLR_exc_full_simul(RF_pulse,b1_range,off_range,gamma,time_step,rf_len,iter_num,sar_weight);
temp = sqrt(sum(m1(2,n_f_b:f_b,1))^2+sum(m1(2,n_f_b:f_b,2))^2)/(f_b - n_f_b + 1);
disp(['SLR bandwidth: ',num2str(temp,'%.4f')]);
temp = sum(sqrt((m1(2,n_f_b:f_b,1)).^2+(m1(2,n_f_b:f_b,2)).^2))/(f_b - n_f_b + 1);
disp(['SLR bandwidth2: ',num2str(temp,'%.4f')]);
temp = sqrt(sum(m1(2,n_f_p:f_p,1))^2+sum(m1(2,n_f_p:f_p,2))^2)/(f_p - n_f_p + 1);
disp(['SLR passband: ',num2str(temp,'%.4f')]);
temp = sqrt(sum(m1(2,n_f_s:f_s,1))^2+sum(m1(2,n_f_s:f_s,2))^2)/(f_s - n_f_s + 1);
disp(['SLR BW+trans: ',num2str(temp,'%.4f')]);
temp2 = sqrt((m1(2,1:n_f_s,1)).^2+(m1(2,1:n_f_s,2)).^2);
temp3 = sqrt((m1(2,f_s:end,1)).^2+(m1(2,f_s:end,2)).^2);
disp(['SLR stopband ripple: ',num2str(max(temp2)),' ',num2str(max(temp3))]);
temp4 = mean(m1(2,n_f_p:f_p,2));
disp(['SLR My signal: ',num2str(temp4,'%.4f')]);
temp5 = mean(m1(2,n_f_p:f_p,1));
disp(['SLR Mx signal: ',num2str(temp5,'%.4f')]);

disp(['SAR reduction: ',num2str((1-sar2/sar1)*100,'%.4f')]);

load('./data/pulse10000');
[~, ind] = min(loss_arr);
exc = squeeze(pulse(ind, :, :))';

time_step = 2.56e-3 / 256;
max_rad = 2 * pi * 42.5775 * 1e+6 * time_step * 0.2 * 1e-4;
exc(:, 1) = (exc(:, 1) + 1.0) / 2.0 * max_rad;
exc(:, 2) = exc(:, 2) * pi;
exc3 = exc;

to_gauss = 2 * pi * 42.5775 * 1e+6 * time_step * 1e-4; % rad to gauss
sar3 = sum((exc3(:,1)./to_gauss).^2)*time_step*1e+6;
disp(['DeepRF SAR: ',num2str(sar3)]);

RF_pulse = exc3;
b1_range = linspace(1.0,1.0,2)';
off_range = linspace(-32000,32000,64001)';
gamma = 1.0;
rf_len = 256;
iter_num = 0;
sar_weight = 0.0;

n_f_p = find(~(off_range+1000));  % -1000 Hz
f_p = find(~(off_range-1000));  % 1000 Hz
n_f_s = find(~(off_range+1570));  % -1570 Hz
f_s = find(~(off_range-1570));  % 1570 Hz
n_f_b = find(~(off_range+(1000+570/2))); % -1000-570/2 Hz
f_b = find(~(off_range-(1000+570/2))); % 1000+570/2 Hz

m3 = SLR_exc_full_simul(RF_pulse,b1_range,off_range,gamma,time_step,rf_len,iter_num,sar_weight);
temp = sqrt(sum(m3(2,n_f_b:f_b,1))^2+sum(m3(2,n_f_b:f_b,2))^2)/(f_b - n_f_b + 1);
disp(['DeepRF bandwidth: ',num2str(temp,'%.4f')]);
temp = sum(sqrt((m3(2,n_f_b:f_b,1)).^2+(m3(2,n_f_b:f_b,2)).^2))/(f_b - n_f_b + 1);
disp(['DeepRF bandwidth2: ',num2str(temp,'%.4f')]);
temp = sqrt(sum(m3(2,n_f_p:f_p,1))^2+sum(m3(2,n_f_p:f_p,2))^2)/(f_p - n_f_p + 1);
disp(['DeepRF passband: ',num2str(temp,'%.4f')]);
temp = sqrt(sum(m3(2,n_f_s:f_s,1))^2+sum(m3(2,n_f_s:f_s,2))^2)/(f_s - n_f_s + 1);
disp(['DeepRF BW+trans: ',num2str(temp,'%.4f')]);
temp2 = sqrt((m3(2,1:n_f_s,1)).^2+(m3(2,1:n_f_s,2)).^2);
temp3 = sqrt((m3(2,f_s:end,1)).^2+(m3(2,f_s:end,2)).^2);
disp(['DeepRF stopband ripple: ',num2str(max(temp2)),' ',num2str(max(temp3))]);
temp4 = mean(m3(2,n_f_p:f_p,2));
disp(['DeepRF My signal: ',num2str(temp4,'%.4f')]);
temp5 = mean(m3(2,n_f_p:f_p,1));
disp(['DeepRF Mx signal: ',num2str(temp5,'%.4f')]);

disp(['SAR reduction: ',num2str((1-sar3/sar1)*100,'%.4f')]);



%% simulated profile
figure; plot(off_range, squeeze(sqrt(m1(2,:,1).^2+m1(2,:,2).^2)), 'k', 'LineWidth', 1.5);
hold on; plot(off_range, squeeze(sqrt(m2(2,:,1).^2+m2(2,:,2).^2)), 'b', 'LineWidth', 1.5);
hold on; plot(off_range, squeeze(sqrt(m3(2,:,1).^2+m3(2,:,2).^2)), 'r', 'LineWidth', 1.5);
xlim([off_range(n_f_s) off_range(f_s)]);
legend('\fontsize{14}|M\fontsize{10}xy\fontsize{14}| (SLR)', '\fontsize{14}|M\fontsize{10}xy\fontsize{14}| (OC)', '\fontsize{14}|M\fontsize{10}xy\fontsize{14}| (DeepRF)', 'NumColumns', 1, 'Location', 'northeast');
set(gca,'FontName','Arial','FontSize',20,'LineWidth',1.5);
yticks([-0.4, 0, 0.5 ,1.0]); ylim([-0.1 1.3]);
xlim([-20000 20000]); xticks(-20000:10000:20000);
xlabel('Frequency (Hz)', 'FontSize', 22); 
ylabel('Signal (A.U.)', 'FontSize', 22);
title('Simulated slice profile', 'FontSize', 22);


%% simulated profiel (magnified)
figure; plot(off_range, squeeze(sqrt(m1(2,:,1).^2+m1(2,:,2).^2)), 'k', 'LineWidth', 1.5);
ph1 = squeeze(angle(m1(2,:,1)+1i*m1(2,:,2)));
hold on; plot(off_range, (ph1 - ph1(32000)) ./ pi, 'k--', 'LineWidth', 1.5);
hold on; plot(off_range, squeeze(sqrt(m2(2,:,1).^2+m2(2,:,2).^2)), 'b', 'LineWidth', 1.5);
ph2 = squeeze(angle(m2(2,:,1)+1i*m2(2,:,2)));
ph2_ = (ph2 - ph2(32000)) ./ pi;
ph2_(ph2_>1) = ph2_(ph2_>1) - 2;
ph2_(ph2_<-1) = ph2_(ph2_<-1) + 2;
hold on; plot(off_range, ph2_, 'b--', 'LineWidth', 1.5);
hold on; plot(off_range, squeeze(sqrt(m3(2,:,1).^2+m3(2,:,2).^2)), 'r', 'LineWidth', 1.5);
ph2 = squeeze(angle(m3(2,:,1)+1i*m3(2,:,2)));
ph2_ = (ph2 - ph2(32000)) ./ pi;
ph2_(ph2_>1) = ph2_(ph2_>1) - 2;
ph2_(ph2_<-1) = ph2_(ph2_<-1) + 2;
hold on; plot(off_range, ph2_, 'r--', 'LineWidth', 1.5);
yticks([-0.4, 0, 0.5, 1.0]); ylim([-0.5 1.9]);
xlim([off_range(n_f_s) off_range(f_s)]);
xticks(linspace(off_range(n_f_s), off_range(f_s), 5));
legend('\fontsize{14}|M\fontsize{10}xy\fontsize{14}| (SLR)', '\fontsize{14}\angleM\fontsize{10}xy \fontsize{14}(SLR)', ...
    '\fontsize{14}|M\fontsize{10}xy\fontsize{14}| (OC)', '\fontsize{14}\angleM\fontsize{10}xy \fontsize{14}(OC)', ...
    '\fontsize{14}|M\fontsize{10}xy\fontsize{14}| (DeepRF)', '\fontsize{14}\angleM\fontsize{10}xy \fontsize{14}(DeepRF)',...
    'NumColumns', 2, 'Location', 'north', 'Orientation', 'horizontal');
set(gca,'FontName','Arial','FontSize',20,'LineWidth',1.5); 
xlabel('Frequency (Hz)', 'FontSize', 22); 
ylabel('Signal (A.U.)', 'FontSize', 22); 
title('Simulated slice profile (zoom-in)', 'FontSize', 22); 

bw1 = sqrt(sum(m1(2,n_f_b:f_b,1))^2+sum(m1(2,n_f_b:f_b,2))^2)/(f_b - n_f_b + 1);
bw2 = sqrt(sum(m2(2,n_f_b:f_b,1))^2+sum(m2(2,n_f_b:f_b,2))^2)/(f_b - n_f_b + 1);
bw3 = sqrt(sum(m3(2,n_f_b:f_b,1))^2+sum(m3(2,n_f_b:f_b,2))^2)/(f_b - n_f_b + 1);

text(-1050, 0.7, 'Mean magnitude over BW ', 'FontSize', 18);
text(-700-120, 0.5, num2str(bw1,'%0.2f'), 'FontSize', 18, 'Color', 'k');
text(-280-120, 0.5, 'vs.', 'FontSize', 18);
text(-20-120, 0.5, num2str(bw2,'%0.2f'), 'FontSize', 18, 'Color', 'b');
text(400-120, 0.5, 'vs.', 'FontSize', 18);
text(660-120, 0.5, num2str(bw3,'%0.2f'), 'FontSize', 18, 'Color', 'r');


%% amplitude plot
time_step = 2.56/256;
figure; plot(time_step:time_step:time_step*256,exc1(:,1)./to_gauss*1e+3,'k','LineWidth',1.5);
hold on; plot(time_step:time_step:time_step*256,exc2(:,1)./to_gauss*1e+3,'b','LineWidth',1.5);
hold on; plot(time_step:time_step:time_step*256,exc3(:,1)./to_gauss*1e+3,'r','LineWidth',1.5);
ylim([0 200]); xlim([0-0.2, 2.56+0.2]); xticks(0:0.5:2.5);
legend('SLR','OC','DeepRF'); 
set(gca,'FontName','Arial','FontSize',20,'LineWidth',1.5);
ylabel('Amplitude (mG)', 'FontSize', 22);
xlabel('Time (ms)', 'FontSize', 22);
title('Amplitude shape', 'FontSize', 22);
text(-0.1, 180, ['ENG -',num2str((1-sar2/sar1)*100,'%.0f'),'%'], 'FontSize', 26, 'Color', 'b');
text(-0.1, 155, ['SAR -',num2str((1-sar3/sar1)*100,'%.0f'),'%'], 'FontSize', 26, 'Color', 'r');


%% phase plot
figure; plot(time_step:time_step:time_step*256,exc1(:,2)./pi*180,'k','LineWidth',1.5);
hold on; plot(time_step:time_step:time_step*256,-angle(exp(-1i*exc2(:,2)))./pi*180,'b','LineWidth',1.5);
hold on; plot(time_step:time_step:time_step*256,angle(exp(-1i*exc3(:,2)))./pi*180,'r','LineWidth',1.5);
xlim([0-0.2, 2.56+0.2]); xticks(0:0.5:2.5);
ylim([-230 320]); yticks(-180:90:180);
legend('SLR','OC','DeepRF','NumColumns',3,'Location','north'); 
set(gca,'FontName','Arial','FontSize',20,'LineWidth',1.5);
ylabel('Phase (\circ)', 'FontSize', 22);
xlabel('Time (ms)', 'FontSize', 22);
title('Phase shape', 'FontSize', 22);


%% stopband ripple analysis
figure; plot(off_range, log10(squeeze(sqrt(m1(2,:,1).^2+m1(2,:,2).^2))),'k', 'LineWidth', 1.5);
hold on; plot(off_range, log10(squeeze(sqrt(m2(2,:,1).^2+m2(2,:,2).^2))),'b', 'LineWidth', 1.5);
hold on; plot(off_range, log10(squeeze(sqrt(m3(2,:,1).^2+m3(2,:,2).^2))),'r', 'LineWidth', 1.5);
xlim([off_range(1) off_range(end)]); 
legend('SLR', 'OC','DeepRF','FontSize', 16);
set(gca,'FontName','Arial','FontSize',20,'LineWidth',1.5);
yticks(-6:2:0);  ylim([-6 0]);
title('\fontsize{22}Log plot of |M\fontsize{18}xy\fontsize{22}|');
yticklabels({'10^{-6}', '10^{-4}', '10^{-2}', '1'});
ylabel('Signal (A.U.)', 'FontSize', 22); xlabel('Frequency (Hz)', 'FontSize', 22);
xlim([-20000 20000]);

sr1 = max(max(sqrt(m1(2,1:30387,1).^2+m1(2,1:30387,2).^2)),max(sqrt(m1(2,33615:end,1).^2+m1(2,33615:end,2).^2)));
sr2 = max(max(sqrt(m2(2,1:30387,1).^2+m2(2,1:30387,2).^2)),max(sqrt(m2(2,33615:end,1).^2+m2(2,33615:end,2).^2)));
sr3 = max(sqrt(m1(2,33615:end,1).^2+m1(2,33615:end,2).^2));

text(-6500*2.5-500, -0.5, 'Max. ripple', 'FontSize', 18);
text(-7000*2.5-2000, -1.3+0.2, [num2str(sr1*100,'%0.1f'),'%,'], 'FontSize', 18, 'Color', 'k');
text(-3400*2.5-5000, -1.3+0.2, [num2str(sr2*100,'%0.1f'),'%,'], 'FontSize', 18, 'Color', 'b');
text(-3400*2.5+1000, -1.3+0.2, [num2str(sr3*100,'%0.1f'),'%'], 'FontSize', 18, 'Color', 'r');


%% passband ripple analysis
figure; plot(off_range, log10(abs(1-squeeze(sqrt(m1(2,:,1).^2+m1(2,:,2).^2)))), 'k', 'LineWidth', 1.5);
hold on; plot(off_range, log10(abs(1-squeeze(sqrt(m2(2,:,1).^2+m2(2,:,2).^2)))), 'b', 'LineWidth', 1.5);
hold on; plot(off_range, log10(abs(1-squeeze(sqrt(m3(2,:,1).^2+m3(2,:,2).^2)))), 'r', 'LineWidth', 1.5);
xlim([off_range(n_f_b) off_range(f_b)]); legend('SLR', 'OC', 'DeepRF', 'Location', 'southeast');
set(gca,'FontName','Arial','FontSize',20,'LineWidth',1.5);
yticks(-8:2:0); title('\fontsize{22}Log plot of |1-M\fontsize{18}xy\fontsize{22}|'); ylim([-8 0]);
yticklabels({'10^{-8}', '10^{-6}', '10^{-4}', '10^{-2}', '10^{0}'});
ylabel('Signal (A.U.)', 'FontSize', 22); xlabel('Frequency (Hz)', 'FontSize', 22);
xticks(int64(linspace(off_range(n_f_b), off_range(f_b), 5)));

pr1 = max(abs(1-sqrt(m1(2,31153:32849,1).^2+m1(2,31153:32849,2).^2)));
pr2 = max(abs(1-sqrt(m2(2,31153:32849,1).^2+m2(2,31153:32849,2).^2)));
pr3 = max(abs(1-sqrt(m2(2,31153:32849,1).^2+m2(2,31153:32849,2).^2)));

text(-350, -0.4, 'Max. ripple', 'FontSize', 18);
text(-600-100, -1.2, [num2str(pr1*100,'%0.1f'),'%'], 'FontSize', 18, 'Color', 'k');
text(-220-100, -1.2, 'vs.', 'FontSize', 18);
text(-20-100, -1.2, [num2str(pr2*100,'%0.1f'),'%'], 'FontSize', 18, 'Color', 'b');
text(360-100, -1.2, 'vs.', 'FontSize', 18);
text(560-100, -1.2, [num2str(pr3*100,'%0.1f'),'%'], 'FontSize', 18, 'Color', 'r');

