function [piFinal, muFinal, varFinal] = learning2(di, c, dataset)
% di dimension, c classes,  
%[priorInit, muInit, varInit] = getRandomParam(di, c);
piInit = rand(1, c);
piInit = piInit / sum(piInit);
range = max(dataset,[],1) - min(dataset,[],1);
rand_times_range = bsxfun(@times, rand(c, di), range);
muInit = bsxfun(@plus, rand_times_range, min(dataset,[],1));

varInit = cell(c, 1);
for j = 1:c
    temp0 = rand(1,di);
    temp0(temp0 < 0.001) = 0.001;
    varInit{j} = diag(temp0);
end

dataset = dataset(:, 1:di);

pi = piInit;
mu = muInit;
var = varInit;

[row column] = size(dataset);
% init the h matrix

for iters = 1 : 1000
    muprev = mu;
    piprev = pi;
    varprev = var;
    % compute the hij first E-step
    fprintf('  EM Iteration %d\n', iters);
    h = zeros(row, c);
    pdf = zeros(row, c);
    % do h i,j for every one, the i is rows and j is components 
    for j = 1:c
        pdf(:, j) = gaussianND(dataset, muprev(j, :), varprev{j});
    end
    pdf_w = bsxfun(@times, pdf, piprev);
    h = bsxfun(@rdivide, pdf_w, sum(pdf_w, 2));
    % for each of the clusters
    % pi can compute together
    pi = sum(h, 1) / row;
    
    for j = 1:c
        %compute the mu = sum of weighted value / sum(h(:,j))
        val = h(:, j)' * dataset;
        mu(j, :) = val ./ sum(h(:, j), 1);
        %mu(j, :) = weightedAverage(h(:, j), dataset);
        
        sigma_k = zeros(1, di);
        Xm = bsxfun(@minus, dataset, mu(j, :));
        for i = 1 : row
            sigma_k = sigma_k + h(i,j) .* (Xm(i, :) .* Xm(i, :));
        end
        vardiag = sigma_k ./ sum(h(:,j));
        % check the diagnal
        vardiag(vardiag < 0.001) = 0.001;
        var{j} = diag(vardiag);
    end
    %m step
%     if max(abs(pi - piprev)./abs(piprev)) < 0.01
%         break;
%     end

%     if max(max(abs(mu - muprev)./abs(muprev))) < 0.01
%         break;
%     end
%compute the log likelihood
    lambdacur = sum(log(pdf * pi'));
    if iters ~= 1
        residual = abs(lambdacur - lambdapre) / abs(lambdacur);
        if residual < 0.005
            break;
        end
    end
    lambdapre = lambdacur;
end

piFinal = piprev;
muFinal = muprev;
varFinal = varprev;
