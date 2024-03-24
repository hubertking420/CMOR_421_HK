clear all
close all

threads = [8, 12, 16, 20, 24];
t_secs_serial = [2.51, 2.51, 2.51, 2.52, 2.51];
t_secs_nested = [0.323, 0.338, 0.337, 0.360, 0.363];
t_secs_collapse = [0.305, 0.233, 0.232, 0.262, 0.319];



% Plot Parallel Efficiency as a function of n
parallel_efficiency_nested = t_secs_serial ./ (t_secs_nested .* threads);
parallel_efficiency_collapse = t_secs_serial ./ (t_secs_collapse .* threads);

figure(1)
hold on
plot(threads, parallel_efficiency_nested, '-o');
plot(threads, parallel_efficiency_collapse, '-*');
xlabel('Number of Threads');
ylabel('Parallel Efficiency (%)');
title('Parallel Efficiency vs. Number of Threads');
legend('#omp pragma parallel for', '#omp pragma parallel for collapse(2)');
grid on;
hold off

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
title('Parallel Efficiency vs. Matrix Size');
grid on;