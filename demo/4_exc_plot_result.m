clear; close all; clc;

d1 = dir('../logs/exc-v51_exc_refinement');
d2 = dir(['../logs/exc-v51_exc_refinement/',d1(end).name,'/arrays']);
t = struct2table(d2);
st = sortrows(t, 'date');

load(['../logs/exc-v51_exc_refinement/',d1(end).name,'/arrays/',char(st.name(end-2))]);
disp(['load ',char(st.name(end-2))]);
[~, ind] = min(loss_arr);
exc = squeeze(pulse(ind, :, :))';

time_step = 2.56e-3 / 256;
max_rad = 2 * pi * 42.5775 * 1e+6 * time_step * 0.2 * 1e-4;
exc(:, 1) = (exc(:, 1) + 1.0) / 2.0 * max_rad;
exc(:, 2) = exc(:, 2) * pi;
exc2 = exc;

to_gauss = 2 * pi * 42.5775 * 1e+6 * time_step * 1e-4; % rad to gauss
sar2 = sum((exc(:,1)./to_gauss).^2)*time_step*1e+6;
disp(['DeepRF SAR: ',num2str(sar2),' mG^2 sec']);

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

load('../data/conv_rf/SLR_exc.txt');

exc = SLR_exc;
exc(:,1) = exc(:,1) * 2 * pi * 42.5775 * time_step * 1e+3;
exc(:,2) = exc(:,2) / 180 * pi;

exc1 = exc;

to_gauss = 2 * pi * 42.5775 * 1e+6 * time_step * 1e-4; % rad to gauss
sar1 = sum((exc(:,1)./to_gauss).^2)*time_step*1e+6;
disp(['SLR SAR: ',num2str(sar1),' mG^2 sec']);

RF_pulse = exc;

m1 = SLR_exc_full_simul(RF_pulse,b1_range,off_range,gamma,time_step,rf_len,iter_num,sar_weight);

disp(['SAR reduction: ',num2str((1-sar2/sar1)*100,'%.1f'),'%']);


%% simulated profile
figure; plot(off_range, squeeze(sqrt(m1(2,:,1).^2+m1(2,:,2).^2)), 'b', 'LineWidth', 1.5);
hold on; plot(off_range, squeeze(sqrt(m2(2,:,1).^2+m2(2,:,2).^2)), 'r', 'LineWidth', 1.5);
xlim([off_range(n_f_s) off_range(f_s)]);
legend('\fontsize{16}|M\fontsize{12}xy\fontsize{16}| (SLR)', '\fontsize{16}|M\fontsize{12}xy\fontsize{16}| (DeepRF)', 'NumColumns', 2, 'Location', 'north');
set(gca,'FontName','Arial','FontSize',20,'LineWidth',1.5);
yticks([-0.4, 0, 0.5 ,1.0]); ylim([-0.1 1.3]);
xlim([-20000 20000]); xticks(-20000:10000:20000);
xlabel('Frequency (Hz)', 'FontSize', 22); 
ylabel('Signal (A.U.)', 'FontSize', 22);
title('Simulated slice profile', 'FontSize', 22);


%% simulated profiel (magnified)
figure; plot(off_range, squeeze(sqrt(m1(2,:,1).^2+m1(2,:,2).^2)), 'b', 'LineWidth', 1.5);
ph1 = squeeze(angle(m1(2,:,1)+1i*m1(2,:,2)));
hold on; plot(off_range, (ph1 - ph1(32000)) ./ pi, 'b--', 'LineWidth', 1.5);
hold on; plot(off_range, squeeze(sqrt(m2(2,:,1).^2+m2(2,:,2).^2)), 'r', 'LineWidth', 1.5);
ph2 = squeeze(angle(m2(2,:,1)+1i*m2(2,:,2)));
ph2_ = (ph2 - ph2(32000)) ./ pi;
ph2_(ph2_>1) = ph2_(ph2_>1) - 2;
ph2_(ph2_<-1) = ph2_(ph2_<-1) + 2;
hold on; plot(off_range, ph2_, 'r--', 'LineWidth', 1.5);
yticks([-0.4, 0, 0.5, 1.0]); ylim([-0.5 1.7]);
xlim([off_range(n_f_s) off_range(f_s)]);
xticks(linspace(off_range(n_f_s), off_range(f_s), 5));
legend('\fontsize{16}|M\fontsize{12}xy\fontsize{16}| (SLR)', '\fontsize{16}\angleM\fontsize{12}xy \fontsize{16}(SLR)', '\fontsize{16}|M\fontsize{12}xy\fontsize{16}| (DeepRF)', '\fontsize{16}\angleM\fontsize{12}xy \fontsize{16}(DeepRF)', 'NumColumns', 2, 'Location', 'north');
set(gca,'FontName','Arial','FontSize',20,'LineWidth',1.5); 
xlabel('Frequency (Hz)', 'FontSize', 22); 
ylabel('Signal (A.U.)', 'FontSize', 22); 
title('Simulated slice profile (zoom-in)', 'FontSize', 22); 


%% amplitude plot
time_step = 2.56/256;
figure; plot(time_step:time_step:time_step*256,exc1(:,1)./to_gauss*1e+3,'b','LineWidth',1.5);
hold on; plot(time_step:time_step:time_step*256,exc2(:,1)./to_gauss*1e+3,'r','LineWidth',1.5);
ylim([0 200]); xlim([0-0.2, 2.56+0.2]); xticks(0:0.5:2.5);
legend('SLR','DeepRF'); 
set(gca,'FontName','Arial','FontSize',20,'LineWidth',1.5);
ylabel('Amplitude (mG)', 'FontSize', 22);
xlabel('Time (ms)', 'FontSize', 22);
title('Amplitude shape', 'FontSize', 22);
text(-0.1, 180, ['SAR -',num2str((1-sar2/sar1)*100,'%.0f'),'%'], 'FontSize', 26, 'Color', 'r');


%% phase plot
figure; plot(time_step:time_step:time_step*256,exc1(:,2)./pi*180,'b','LineWidth',1.5);
hold on; plot(time_step:time_step:time_step*256,angle(exp(-1i*exc2(:,2)))./pi*180,'r','LineWidth',1.5);
xlim([0-0.2, 2.56+0.2]); xticks(0:0.5:2.5);
ylim([-230 330]); yticks(-180:90:180);
legend('SLR','DeepRF'); 
set(gca,'FontName','Arial','FontSize',20,'LineWidth',1.5);
ylabel('Phase (\circ)', 'FontSize', 22);
xlabel('Time (ms)', 'FontSize', 22);
title('Phase shape', 'FontSize', 22);


%% stopband ripple analysis
figure; plot(off_range, log10(squeeze(sqrt(m1(2,:,1).^2+m1(2,:,2).^2))), 'LineWidth', 1.5);
hold on; plot(off_range, log10(squeeze(sqrt(m2(2,:,1).^2+m2(2,:,2).^2))), 'LineWidth', 1.5);
xlim([off_range(1) off_range(end)]); 
legend('SLR', 'DeepRF');
set(gca,'FontName','Arial','FontSize',20,'LineWidth',1.5);
yticks(-8:2:0); 
title('\fontsize{22}Log plot of |M\fontsize{18}xy\fontsize{22}|');
yticklabels({'10^{-8}', '10^{-6}', '10^{-4}', '10^{-2}', '1'});
ylabel('Signal (A.U.)', 'FontSize', 22); xlabel('Frequency (Hz)', 'FontSize', 22);
xlim([-20000 20000]);


%% passband ripple analysis
figure; plot(off_range, log10(abs(1-squeeze(sqrt(m1(2,:,1).^2+m1(2,:,2).^2)))), 'LineWidth', 1.5);
hold on; plot(off_range, log10(abs(1-squeeze(sqrt(m2(2,:,1).^2+m2(2,:,2).^2)))), 'LineWidth', 1.5);
xlim([off_range(n_f_b) off_range(f_b)]); legend('SLR', 'DeepRF', 'Location', 'southeast');
set(gca,'FontName','Arial','FontSize',20,'LineWidth',1.5);
yticks(-10:2:0); title('\fontsize{22}Log plot of |1-M\fontsize{18}xy\fontsize{22}|');
yticklabels({'10^{-10}', '10^{-8}', '10^{-6}', '10^{-4}', '10^{-2}', '10^{0}'});
ylabel('Signal (A.U.)', 'FontSize', 22); xlabel('Frequency (Hz)', 'FontSize', 22);
xticks(int64(linspace(off_range(n_f_b), off_range(f_b), 5)));

