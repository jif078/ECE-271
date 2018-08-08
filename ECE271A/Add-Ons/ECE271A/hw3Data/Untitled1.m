fprintf('Loading dataset.\n\n');

%(a) conider the training setD1 and strategy1. For each class, compute the
%covariance sigma of the class conditional, and the posterior mean mu1, ans
%covariance sigma. Next, compute the parameters of the predictive
%distribution for each classed.

[rowBG columnBG] = size(D1_BG);
[rowFG columnFG] = size(D1_FG);
priorBG = rowBG / (rowBG + rowFG);
priorFG = rowFG / (rowBG + rowFG);
%mu
muBG = sum(D1_BG)/rowBG;
muFG = sum(D1_FG)/rowFG;

%covariance of the class conditional
varBG = cov(D1_BG);
varFG = cov(D1_FG);

%%put read img here
falseness = zeros(9,1);

for j = 1:9
    varBG0 = alpha(j) * W0;
    varFG0 = alpha(j) * W0;
% for the prior we have know 
% mu0_BG and mu0_FG the posterior mu0
% variance is sigma = diag(alpha * weight)
    
    varBG0 = diag(varBG0);
    varFG0 = diag(varFG0);

    weight1BG = varBG0 * inv(varBG0 + (varBG / rowBG) );
    weight2BG = (varBG / rowBG) * inv(varBG0 + (varBG / rowBG));
 
    mu1BG = weight1BG * transpose(muBG) + weight2BG * transpose(mu0_BG);
    mu1BG = transpose(mu1BG);
 %something wrong here we should not multiply the rowFG agaoun
    weight1FG = varFG0 * inv(varFG0 + (varFG / rowFG) );
    weight2FG = (varFG / rowFG) * inv(varFG0 + (varFG / rowFG));

    mu1FG = weight1FG * transpose(muFG) + weight2FG * transpose(mu0_FG);
    mu1FG = transpose(mu1FG);
%  varBG1 = inv( inv(varBG0) + inv(varBG) * rowBG );
%  varFG1 = inv( inv(varFG0) + inv(varFG) * rowFG );
%  
    varBG1 = varBG0 * inv(varBG0 + (varBG / rowBG) ) * (varBG / rowBG);  
    varFG1 = varFG0 * inv(varFG0 + (varFG / rowFG) ) * (varFG / rowFG);
 

% for the parameters of X|T posterior
    muXDBG = mu1BG;
    muXDFG = mu1FG;
    varXDBG = varBG + varBG1;
    varXDFG = varFG + varFG1;

%now do the beyes decision

    d = zeros(255,270);
    %2pi^d-- the d is dimension
    alphaBG = log(((2 * pi)^64) * det(varXDBG)) - 2*log(priorBG);
    alphaFG = log(((2 * pi)^64) * det(varXDFG)) - 2*log(priorFG);
    % alphaBG = log(((2 * pi)^64) * det(varBG0)) - 2*log(priorBG0);
    % alphaFG = log(((2 * pi)^64) * det(varFG0)) - 2*log(priorFG0);
    gBG = zeros(1,65224);
    gFG = zeros(1,65224);
    g2 = zeros(1,65224);
for count = 1:65224

    %gFG(count) = 1/(1+exp(dxy(a(count, :), muXDFG, varXDFG) - dxy(a(count, :), muXDBG, varXDBG) + alphaFG - alphaBG));
    %FG(count) = 1/(1+exp(dxy(a(count, :), muXDFG, varXDFG) - dxy(a(count, :), muXDBG, varXDBG) + alphaFG - alphaBG));
    gBG(count) = 1/(1+exp(dxy(a(count, :), muXDBG, varXDBG) - dxy(a(count, :), muXDFG, varXDFG) + alphaBG - alphaFG));
    %g2(count) = exp(-0.5 * );
    %gFG(count) = 1/(1+exp(dxy(a(count, :), mu2, sig2) - dxy(a(count, :), mu1, sig1) + alphaFG - alphaBG));
    if(gBG(count) < 0.5)
        d(rem(count,248)+1, floor(count/248)+1) = 1;
    end
end
    figure;
    Cmask = mat2gray(d);
    imshow(Cmask);


    [A2 B2] = imread('cheetah_mask.bmp');
    A2 = A2/255;
    falseness(j,1) = sum(sum(xor(A2, d))) / (255*277);
end
figure;
semilogx(alpha, falseness);
xlabel('alpha');
ylabel('probability of error');
set(gca,'XGrid','on');
set(gca,'YGrid','on');
%1 - 0.782


%% for ML

%ML solution onider the training setD1 and strategy1. For each class, compute the
%covariance sigma of the class conditional, and the posterior mean mu1, ans
%covariance sigma. Next, compute the parameters of the predictive
%distribution for each classed.

[rowBG columnBG] = size(D1_BG);
[rowFG columnFG] = size(D1_FG);
priorBG = rowBG / (rowBG + rowFG);
priorFG = rowFG / (rowBG + rowFG);
%mu1
mu1 = sum(D1_BG)/rowBG;
mu2 = sum(D1_FG)/rowFG;

sig1 = cov(D1_BG);
sig2 = cov(D1_FG);

d = zeros(255,270);
%2pi^d-- the d is dimension
alphaBG = log(((2 * pi)^64) * det(sig1)) - 2*log(priorBG);
alphaFG = log(((2 * pi)^64) * det(sig2)) - 2*log(priorFG);

gBG = zeros(1,65224);
%gFG = zeros(1,65224);
%g2 = zeros(1,65224);
for count = 1:65224
    gBG(count) = 1/(1+exp(dxy(a(count, :), mu1, sig1) - dxy(a(count, :), mu2, sig2) + alphaBG - alphaFG));
    %g2(count) = exp(-0.5 * );
    %gFG(count) = 1/(1+exp(dxy(a(count, :), mu2, sig2) - dxy(a(count, :), mu1, sig1) + alphaFG - alphaBG));
    if(gBG(count) < 0.5)
        d(rem(count,248)+1, floor(count/248)+1) = 1;
    end
end

%g2 = zeros(1,65224);
% for count = 1:65224
%     gBG(count) = 1/(1+exp(dxy(a(count, :), mu0_BG, varBG0) - dxy(a(count, :), mu0_FG, varFG0) + alphaBG - alphaFG));
%     %g2(count) = exp(-0.5 * );
%     %gFG(count) = 1/(1+exp(dxy(a(count, :), mu2, sig2) - dxy(a(count, :), mu1, sig1) + alphaFG - alphaBG));
%     if(gBG(count) < 0.5)
%         d(rem(count,248)+1, floor(count/248)+1) = 1;
%     end
% end
figure;
Cmask = mat2gray(d);
imshow(Cmask);


[A2 B2] = imread('cheetah_mask.bmp');
A2 = A2/255;
falsenessML = sum(sum(xor(A2, d))) / (255*277);

%% MAP solution
fprintf('Loading dataset.\n\n');

%(a) conider the training setD1 and strategy1. For each class, compute the
%covariance sigma of the class conditional, and the posterior mean mu1, ans
%covariance sigma. Next, compute the parameters of the predictive
%distribution for each classed.

[rowBG columnBG] = size(D1_BG);
[rowFG columnFG] = size(D1_FG);
priorBG = rowBG / (rowBG + rowFG);
priorFG = rowFG / (rowBG + rowFG);
%mu
muBG = sum(D1_BG)/rowBG;
muFG = sum(D1_FG)/rowFG;

%covariance of the class conditional
varBG = cov(D1_BG);
varFG = cov(D1_FG);

%%put read img here

falsenessMAP = zeros(9,1);

for j = 1:9
    varBG0 = alpha(j) * W0;
    varFG0 = alpha(j) * W0;
% for the prior we have know 
% mu0_BG and mu0_FG the posterior mu0
% variance is sigma = diag(alpha * weight)
    
    varBG0 = diag(varBG0);
    varFG0 = diag(varFG0);

    weight1BG = varBG0 * inv(varBG0 + (varBG / rowBG) );
    weight2BG = (varBG / rowBG) * inv(varBG0 + (varBG / rowBG));
    mu1BG = weight1BG * transpose(muBG) + weight2BG * transpose(mu0_BG);
    mu1BG = transpose(mu1BG);
 %something wrong here we should not multiply the rowFG again
    weight1FG = varFG0 * inv(varFG0 + (varFG / rowFG) );
    weight2FG = (varFG / rowFG) * inv(varFG0 + (varFG / rowFG));
    mu1FG = weight1FG * transpose(muFG) + weight2FG * transpose(mu0_FG);
    mu1FG = transpose(mu1FG);
% for the parameters of X|T posterior

%now do the beyes decision

    d = zeros(255,270);
    %2pi^d-- the d is dimension
    alphaBG = log(((2 * pi)^64) * det(varBG)) - 2*log(priorBG);
    alphaFG = log(((2 * pi)^64) * det(varFG)) - 2*log(priorFG);
    % alphaBG = log(((2 * pi)^64) * det(varBG0)) - 2*log(priorBG0);
    % alphaFG = log(((2 * pi)^64) * det(varFG0)) - 2*log(priorFG0);
    gBG = zeros(1,65224);
    gFG = zeros(1,65224);
    g2 = zeros(1,65224);
for count = 1:65224

    %gFG(count) = 1/(1+exp(dxy(a(count, :), muXDFG, varXDFG) - dxy(a(count, :), muXDBG, varXDBG) + alphaFG - alphaBG));
    %FG(count) = 1/(1+exp(dxy(a(count, :), muXDFG, varXDFG) - dxy(a(count, :), muXDBG, varXDBG) + alphaFG - alphaBG));
    gBG(count) = 1/(1+exp(dxy(a(count, :), mu1BG, varBG) - dxy(a(count, :), mu1FG, varFG) + alphaBG - alphaFG));
    %g2(count) = exp(-0.5 * );
    %gFG(count) = 1/(1+exp(dxy(a(count, :), mu2, sig2) - dxy(a(count, :), mu1, sig1) + alphaFG - alphaBG));
    if(gBG(count) < 0.5)
        d(rem(count,248)+1, floor(count/248)+1) = 1;
    end
end
    figure;
    Cmask = mat2gray(d);
    imshow(Cmask);


    [A2 B2] = imread('cheetah_mask.bmp');
    A2 = A2/255;
    falsenessMAP(j,1) = sum(sum(xor(A2, d))) / (255*277);
end
figure;
semilogx(alpha, falsenessMAP);
xlabel('alpha');
ylabel('probability of error');
set(gca,'XGrid','on');
set(gca,'YGrid','on');


%% plot all 3 solutions
figure;
falsenessML2 = ones(1,9) * falsenessML;
semilogx(alpha, falseness, '-r',alpha, falsenessML2, '-g',  alpha, falsenessMAP, '-b' );
xlabel('alpha');
ylabel('probability of error');
set(gca,'XGrid','on');
set(gca,'YGrid','on');
