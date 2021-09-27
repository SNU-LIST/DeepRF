%% slice_selective_excitation

clear; clc;

load('./data/slice_selective_excitation');

disp('Slice-selective excitation RF: ');
disp(['RF ENG reduction (%): ', num2str(mean(sar),'%0.1f'), ' +/- ', num2str(std(sar),'%0.1f')]);
disp(['Mean magnitude: ', num2str(mean(bw),'%0.2f'), ' +/- ', num2str(std(bw),'%0.2f')]);
disp(['Maximum passband ripple (%): ', num2str(mean(pass),'%0.1f'), ' +/- ', num2str(std(pass),'%0.1f')]);
disp(['Maximum stopband ripple (%): ', num2str(mean(stop),'%0.1f'), ' +/- ', num2str(std(stop),'%0.1f')]);


%% slice_selective_inversion

clear; clc;

load('./data/slice_selective_inversion');

disp('Slice-selective inversion RF: ');
disp(['RF ENG reduction (%): ', num2str(mean(sar),'%0.1f'), ' +/- ', num2str(std(sar),'%0.1f')]);
disp(['Mean Mz over BW: ', num2str(mean(bw),'%0.2f'), ' +/- ', num2str(std(bw),'%0.2f')]);
disp(['Maximum passband ripple (%): ', num2str(mean(pass),'%0.1f'), ' +/- ', num2str(std(pass),'%0.1f')]);
disp(['Maximum stopband ripple (%): ', num2str(mean(stop),'%0.1f'), ' +/- ', num2str(std(stop),'%0.1f')]);


%% B1_insensitive_volume_inversion

clear; clc;

load('./data/B1_insensitive_volume_inversion');

disp('B1-insensitive volume-inversion RF: ');
disp(['RF ENG reduction (%): ', num2str(mean(sar),'%0.1f'), ' +/- ', num2str(std(sar),'%0.1f')]);
disp(['Mean Mz: ', num2str(mean(bw),'%0.2f'), ' +/- ', num2str(std(bw),'%0.2f')]);


%% B1_insensitive_selective_inversion

clear; clc;

load('./data/B1_insensitive_selective_inversion');

disp('B1-insensitive selective-inversion RF: ');
disp(['RF ENG reduction (%): ', num2str(mean(sar),'%0.1f'), ' +/- ', num2str(std(sar),'%0.1f')]);
disp(['Mean Mz: ', num2str(mean(bw),'%0.2f'), ' +/- ', num2str(std(bw),'%0.2f')]);


