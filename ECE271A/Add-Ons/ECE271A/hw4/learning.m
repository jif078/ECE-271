function [piFinal, muFinal, varFinal] = learning(di, c, priorInit, muInit, varInit, dataset)
% di dimension, c classes,  
%[priorInit, muInit, varInit] = getRandomParam(di, c);
dataset = dataset(:, 1:di);
piprev = priorInit;
muprev = muInit;
varprev = varInit;

picur = piprev;
mucur = muprev;
varcur = varprev;

[row column] = size(dataset);
% init the h matrix
h = zeros(row, c);

for times = 1 : 30
    % compute the hij first E-step
    for i = 1 : row
        
        for j = 1:c
            %mvnpdf -- Multivariate normal probability density function
            h(i, j) = mvnpdf(dataset(i, :), muprev(j, :), diag(varprev(j, :))) * piprev(j);
        end
        h(i, :) = h(i, :) / sum(h(i, :));
    end
    
    % compute the M-step, parameters
    for j = 1:c
        sumh = sum(h(:, j));
        mucur(j, :) = h(:, j)'*dataset(:, :)./sumh;
        picur(j) = sumh / row;
%         varcur(j, :) = zeros(1, di);
%         for i2 = 1:row
%             varcur(j,:) = varcur(j,:) + sqrt(h(i2, j) * ((dataset(i2, :) - mucur(j,:)).^2));
%         end
%         varcur(j,:) = varcur(j,:) / sumh;
    end
    
    for j=1:c
       sum_1=0;
       sum_2=0;
       for i=1: size(dataset,1)
           sum_1=sum_1+h(i,j)*(dataset(i,:)-mucur(j,:)).^2;
           sum_2=sum_2+h(i,j);
       end
       varcur(j, :) = sum_1 / sum_2;
       varcur(varcur < 0.002) = 0.002;
   end
    if max(max(abs(mu - muprev)./abs(muprev))) < 0.01
        break;
    end
    muprev = mucur;
    piprev = picur;
    varprev = varcur;
end

piFinal = piprev;
muFinal = muprev;
varFinal = varprev;

