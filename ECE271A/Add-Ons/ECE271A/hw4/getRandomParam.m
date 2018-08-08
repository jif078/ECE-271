function [priorInit, muInit, varInit] = getRandomParam(di, c)
priorInit = ones(1,c) * 1 / c;
muInit = rand(c, di);
temp = rand(c,di);
temp(temp < 0.001) = 0.001;
varInit = cell(c,1);
for i = 1:c
    varInit{i} = diag(temp(i, :));
end
end
