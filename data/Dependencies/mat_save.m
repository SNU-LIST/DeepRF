%% 5.12 ms (250 Hz + 336.6 Hz)
clear; clc; close all;

load('inv_du512.txt');

inv2 = inv_du512;

% mT to rad
time_step = 5.12e-3/256; % sec
inv2(:,1) = inv2(:,1) * 2 * pi * 42.5775 * time_step * 1e+3;
inv2(:,2) = inv2(:,2) / 180 * pi;
to_gauss = 2 * pi * 42.5775 * 1e+6 * time_step * 1e-4; % rad to gauss
result = zeros(size(inv2));
result(:,1) = 2*(inv2(:,1)/to_gauss/0.2)-1;
result(:,2) = inv2(:,2) / pi;
result = result';
save inv_du512 result

%% 5.12 ms (250 Hz + 336.6 Hz) -- equi-ripple
clear; clc; close all;

load('inv_du512_equi.txt');

inv2 = inv_du512_equi;

% mT to rad
time_step = 5.12e-3/256; % sec
inv2(:,1) = inv2(:,1) * 2 * pi * 42.5775 * time_step * 1e+3;
inv2(:,2) = inv2(:,2) / 180 * pi;
to_gauss = 2 * pi * 42.5775 * 1e+6 * time_step * 1e-4; % rad to gauss
result = zeros(size(inv2));
result(:,1) = 2*(inv2(:,1)/to_gauss/0.2)-1;
result(:,2) = inv2(:,2) / pi;
result = result';
save inv_du512_equi result

%% 5.12 ms (250 Hz + 336.6 Hz) -- min_phase
clear; clc; close all;

load('inv_du512_min.txt');

inv2 = inv_du512_min;

% mT to rad
time_step = 5.12e-3/256; % sec
inv2(:,1) = inv2(:,1) * 2 * pi * 42.5775 * time_step * 1e+3;
inv2(:,2) = inv2(:,2) / 180 * pi;
to_gauss = 2 * pi * 42.5775 * 1e+6 * time_step * 1e-4; % rad to gauss
result = zeros(size(inv2));
result(:,1) = 2*(inv2(:,1)/to_gauss/0.2)-1;
result(:,2) = inv2(:,2) / pi;
result = result';
save inv_du512_min result

%% 7.68 ms (250 Hz + 224.4 Hz)
clear; clc; close all;

load('inv_du768.txt');

inv2 = inv_du768;

% mT to rad
time_step = 7.68e-3/256; % sec
inv2(:,1) = inv2(:,1) * 2 * pi * 42.5775 * time_step * 1e+3;
inv2(:,2) = inv2(:,2) / 180 * pi;
to_gauss = 2 * pi * 42.5775 * 1e+6 * time_step * 1e-4; % rad to gauss
result = zeros(size(inv2));
result(:,1) = 2*(inv2(:,1)/to_gauss/0.2)-1;
result(:,2) = inv2(:,2) / pi;
result = result';
save inv_du768 result

%% 10.24 ms (250 Hz + 168.3 Hz)
clear; clc; close all;

load('inv_du1024.txt');

inv2 = inv_du1024;

% mT to rad
time_step = 10.24e-3/256; % sec
inv2(:,1) = inv2(:,1) * 2 * pi * 42.5775 * time_step * 1e+3;
inv2(:,2) = inv2(:,2) / 180 * pi;
to_gauss = 2 * pi * 42.5775 * 1e+6 * time_step * 1e-4; % rad to gauss
result = zeros(size(inv2));
result(:,1) = 2*(inv2(:,1)/to_gauss/0.2)-1;
result(:,2) = inv2(:,2) / pi;
result = result';
save inv_du1024 result

%% 12.8 ms (250 Hz + 134.6 Hz)
clear; clc; close all;

load('inv_du1280.txt');

inv2 = inv_du1280;

% mT to rad
time_step = 12.8e-3/256; % sec
inv2(:,1) = inv2(:,1) * 2 * pi * 42.5775 * time_step * 1e+3;
inv2(:,2) = inv2(:,2) / 180 * pi;
to_gauss = 2 * pi * 42.5775 * 1e+6 * time_step * 1e-4; % rad to gauss
result = zeros(size(inv2));
result(:,1) = 2*(inv2(:,1)/to_gauss/0.2)-1;
result(:,2) = inv2(:,2) / pi;
result = result';
save inv_du1280 result

%% exc

load('exc_du256.txt');

% mT to rad
time_step = 2.56e-3/256; % sec
exc(:,1) = exc(:,1) * 2 * pi * 42.5775 * time_step * 1e+3;
exc(:,2) = exc(:,2) / 180 * pi;

exc2 = exc;

to_gauss = 2 * pi * 42.5775 * 1e+6 * time_step * 1e-4; % rad to gauss
result = zeros(size(exc));
result(:,1) = 2*(exc2(:,1)/to_gauss/0.2)-1;
result(:,2) = exc(:,2) / pi;
result = result';
save exc_du256 result;

%% exc

load('exc_du256_equi.txt');

exc = exc_du256_equi;

% mT to rad
time_step = 2.56e-3/256; % sec
exc(:,1) = exc(:,1) * 2 * pi * 42.5775 * time_step * 1e+3;
exc(:,2) = exc(:,2) / 180 * pi;

exc2 = exc;

to_gauss = 2 * pi * 42.5775 * 1e+6 * time_step * 1e-4; % rad to gauss
result = zeros(size(exc));
result(:,1) = 2*(exc2(:,1)/to_gauss/0.2)-1;
result(:,2) = exc(:,2) / pi;
result = result';
save exc_du256_equi result;



%% inv sigpy
clear; clc; close all;

load('inv_pm_4.2803_256_0.01_0.01.mat');

inv2 = zeros(256,2);

% mT to rad
time_step = 5.12e-3/256; % sec
to_gauss = 2 * pi * 42.5775 * 1e+6 * time_step * 1e-4; % rad to gauss
inv2(:,1) = abs(pulse');
inv2(:,2) = angle(pulse');
result = zeros(size(inv2));
result(:,1) = 2*(inv2(:,1)/to_gauss/0.2)-1;
result(:,2) = inv2(:,2) / pi;
result = result';
save inv_pm result



%% exc sigpy

load('ex_pm_6.5792_256_0.01_0.01.mat');

% mT to rad
time_step = 2.56e-3/256; % sec
exc = zeros(256,2);
exc(:,1) = abs(pulse');
exc(:,2) = angle(pulse');

exc2 = exc;

to_gauss = 2 * pi * 42.5775 * 1e+6 * time_step * 1e-4; % rad to gauss
result = zeros(size(exc));
result(:,1) = 2*(exc2(:,1)/to_gauss/0.2)-1;
result(:,2) = exc(:,2) / pi;
result = result';
save exc_pm result;







%% inversion matpulse min phase equi-ripple
clear; clc; close all;

load('inv_matpulse_min_phase_pm.txt');

inv2 = inv_matpulse_min_phase_pm;

% mT to rad
time_step = 5.12e-3/256; % sec
inv2(:,1) = inv2(:,1) * 2 * pi * 42.5775 * time_step * 1e+3;
inv2(:,2) = inv2(:,2) / 180 * pi;
to_gauss = 2 * pi * 42.5775 * 1e+6 * time_step * 1e-4; % rad to gauss
result = zeros(size(inv2));
result(:,1) = 2*(inv2(:,1)/to_gauss/0.2)-1;
result(:,2) = inv2(:,2) / pi;
result = result';
save inv_du512_min_pm result




