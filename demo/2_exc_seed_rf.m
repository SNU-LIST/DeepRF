%%
clear; close all; clc;

d = dir('../logs/exc_generation');

num = 256;

pulses = [];
rews = [];

for i=1:length(d)-2
    if isfolder(['../logs/exc_generation/',d(i+2).name])
        d1 = dir(['../logs/exc_generation/',d(i+2).name,'/arrays']);
        if length(d1) > 2
            disp(['# of DRL run: ',num2str(i),'/50']);
            load(['../logs/exc_generation/',d(i+2).name,'/arrays/pulse300.mat']);
            [sorted,ind] = sort(rew_list, 'descend');
            rews = [rews, sorted(1:num)];
            pulses = [pulses; rf_list(ind(1:num),:,:)];
        end
    end
end

[sorted,ind] = sort(rews, 'descend');
result = pulses(ind(~isnan(sorted)), :, :);
result = result(1:num, :, :);

m = interp1(linspace(1,128,32),result(:, :, 1)',1:128)';
p = interp1(linspace(1,128,32),result(:, :, 2)',1:128)';
m = cat(2, m, fliplr(m));
p = cat(2, p, fliplr(p));
result = cat(3, m, p);

save('../logs/exc_generation/seed_rfs', 'result')




