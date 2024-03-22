clear all
close all

% Back Solve Data
threads = 8;
n = [3000, 6000, 9000, 12000, 15000, 18000, 21000, 24000, 27000, 30000];
t_secs_serial = [0.130, 0.662, 1.49, 2.82, 4.29 6.30, 9.73, 17.28, 14.5, 18.3];
t_secs_static = [0.114, 0.236, 0.414, 0.87, 0.98, 1.43, 2.29, 3.42, 2.65, 3.37];
t_secs_dynamic = [1.93, 8.39, 18.3];

% Plot Parallel Efficiency as a function of n
parallel_efficiency_static = t_secs_serial ./ (t_secs_static * threads);
figure(2)
plot(n, parallel_efficiency_static, '-o');
xlabel('Matrix Size n');
ylabel('Parallel Efficiency (%)');
title('Parallel Efficiency vs Matrix Size');
grid on;