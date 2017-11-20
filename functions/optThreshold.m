function [threshold, classError] = optThreshold(feature, labels)

len = 150;
t_temp = linspace(min(feature), max(feature), len);

cl_e = zeros(1,len);

for t = 1:len
    cl_e(t) = clError(feature, labels, t_temp(t));
end

[classError,t_ideal]  = min(cl_e);
threshold = t_temp(t_ideal);
