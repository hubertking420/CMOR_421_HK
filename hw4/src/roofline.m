clear all;
close all;

% Data calculation
[ci_s_1, flops_s_s_1] = ci(50.293*10^9, 98.357*10^9, 0.382, 12582906);
[ci_s_2, flops_s_s_2] = ci(59.779*10^9, 61.150*10^9, 0.324, 12582906);
[ci_r_0, flops_s_r_0] = ci(13.765*10^9, 86.187*10^6, 1.422, 4177920);
[ci_r_1, flops_s_r_1] = ci(21.088*10^9, 120.56*10^6, 0.927, 4177920);
[ci_r_2, flops_s_r_2] = ci(34.805*10^9, 121.53*10^6, 0.548, 4186112);

% Constants for roofline
peak_performance = 8.74e12;  % 8.74 TFLOPs
peak_bandwidth = 480e9;  % 480 GB/s

% Define the range of arithmetic intensities
I = logspace(-2, 2, 100);  % from 0.01 to 100 FLOPs/Byte

% Calculate performance for each intensity
P = min(peak_performance, peak_bandwidth * I);

% Plotting the roofline model for reduction
figure;
loglog(I, P, 'b', 'LineWidth', 2);
hold on;
loglog(I, repmat(peak_performance, length(I), 1), 'r--', 'LineWidth', 2);
% Annotations for stencil and reduction values
loglog(ci_s_1, flops_s_s_1, 'ro', 'MarkerFaceColor', 'r');  % version 1
loglog(ci_s_2, flops_s_s_2, 'go', 'MarkerFaceColor', 'g');  % version 2
% Labels and legend
xlabel('Computational Intensity (FLOPs/Byte)');
ylabel('Performance (GFLOPs)');
title('Roofline Model for Stencil');
legend('\beta\timesI', '\pi', 'Version 1', 'Version 2', 'Location', 'SouthEast');
hold off;

% Plotting the roofline model for stencil
figure;
loglog(I, P, 'b', 'LineWidth', 2);
hold on;
loglog(I, repmat(peak_performance, length(I), 1), 'r--', 'LineWidth', 2);
% Annotations for stencil and reduction values
loglog(ci_r_0, flops_s_r_0, 'ro', 'MarkerFaceColor', 'r');  % version 0
loglog(ci_r_1, flops_s_r_1, 'go', 'MarkerFaceColor', 'g');  % version 1
loglog(ci_r_2, flops_s_r_2, 'bo', 'MarkerFaceColor', 'b');  % version 2
% Labels and legend
xlabel('Computational Intensity (FLOPs/Byte)');
ylabel('Performance (GFLOPs)');
title('Roofline Model for Reduction');
legend('\beta\timesI', '\pi', 'Version 0', 'Version 1', 'Version 2', 'Location', 'SouthEast');
hold off;

% Function definition
function [ci, flops_s] = ci(dram_r, dram_w, time, flops)
    time_s = time * 10^-3;  % Convert ms to seconds
    flops_s = flops / time_s;  % Sustained FLOPs/s
    ci = flops_s / (dram_r + dram_w);  % Computational Intensity
end