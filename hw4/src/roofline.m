close all
clear all
ci_stencil_1 = ci(50.293*10^9, 98.357*10^9, 0.382, 12582906);
ci_stencil_2 = ci(59.779*10^9, 61.150*10^9, 0.324, 12582906);
ci_red_0 = ci(13.765*10^9, 86.187*10^6, 1.422, 4177920);
ci_red_1 = ci(21.088*10^9, 120.56*10^6, 0.927, 4177920);
ci_red_2 = ci(34.805*10^9, 121.53*10^6, 0.548, 4186112);
function [ci] = ci(dram_r, dram_w, time, flops)
    time = time*10^-3;
    ci = flops/(time*(dram_r+dram_w));
end