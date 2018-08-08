fprintf('Loading dataset.\n\n');

%  Load data TrainsmapleDCT_BG & TrainsampleDCT_FG
load ('TrainingSamplesDCT_8_new.mat');

%(a) conider the training setD1 and strategy1. For each class, compute the
%covariance sigma of the class conditional, and the posterior mean mu1, ans
%covariance sigma. Next, compute the parameters of the predictive
%distribution for each classed.

[rowBG columnBG] = size(D1_BG);
[rowFG columnFG] = size(D1_FG);
priorBG = rowBG / (rowBG + rowFG);
priorFG = rowFG / (rowBG + rowFG);
%mu1
muBG = sum(D1_BG)/rowBG;
muFG = sum(D1_FG)/rowFG;

%covariance of the class conditional
varBG = cov(D1_BG);
varFG = cov(D1_FG);

%%put read img here
[A, B] = imread('cheetah.bmp');
% A is the degree of the pixels 
%B is the color map from 0-149 degrees
A2 = im2double(A);

% a is the matrix of sliding windows
a = zeros(65224, 64)

for i = 1:(270-7)  %colomns
    for j = 1:(255-7) %rows
        temp = A2(j:j+7, i:i+7)
        temp = dct2(temp);
        a((i-1)*248+j, :) = tras264(temp);
    end
end

falseness2 = zeros(9,1);


for j = 1:9
varBG0 = alpha(j) * W0;
varFG0 = alpha(j) * W0;
% for the prior we have know 
% mu0_BG and mu0_FG the posterior mu0
% variance is sigma = diag(alpha * weight)

varBG0 = diag(varBG0);
varFG0 = diag(varFG0);



  weight1BG = rowBG * varBG0 / (rowBG * varBG0 + varBG);
  weight2BG = varBG / (rowBG * varBG0 + varBG);
%   weight1BG = varBG0 / (varBG0 + varBG);
%  weight2BG = varBG / (varBG0 + varBG);
  
%    weight1BG = 1;
%  weight2BG = 0;
 

 mu1BG = weight1BG * transpose(muBG) + weight2BG * transpose(mu0_BG);
 %something wrong here we should not multiply the rowFG agaoun
  weight1FG = rowFG * varFG0 / (rowFG * varFG0 + varFG);
  weight2FG = varFG / (rowFG * varFG0 + varFG);
 
%   weight1FG = varFG0 / (varFG0 + varFG);
%  weight2FG = varFG / (varFG0 + varFG);
 
%    weight1FG = 1;
%  weight2FG = 0;
%  
 
 mu1FG = weight1FG * transpose(muFG) + weight2FG * transpose(mu0_FG);
 
 mu1BG = transpose(mu1BG);
 mu1FG = transpose(mu1FG);
 
  varBG1 = inv( inv(varBG0) + inv(varBG) * rowBG );
  varFG1 = inv( inv(varFG0) + inv(varFG) * rowFG );
%  
%  varBG1 = inv( inv(varBG0)  + inv(varBG) );
%  varFG1 = inv( inv(varFG0)  + inv(varFG) );
 

% for the parameters of X|T posterior
% muXDBG = transpose(mu1BG);
% muXDFG = transpose(mu1FG);
% varXDBG = varBG + varBG1;
% varXDFG = varFG + varFG1;

%now do the beyes decision

d = zeros(255,270);
%2pi^d-- the d is dimension
 alphaBG = log(((2 * pi)^64) * det(varBG0)) - 2*log(priorBG);
 alphaFG = log(((2 * pi)^64) * det(varFG0)) - 2*log(priorFG);
% alphaBG = log(((2 * pi)^64) * det(varBG0)) - 2*log(priorBG0);
% alphaFG = log(((2 * pi)^64) * det(varFG0)) - 2*log(priorFG0);
gBG = zeros(1,65224);
gFG = zeros(1,65224);
g2 = zeros(1,65224);
for count = 1:65224

    %gFG(count) = 1/(1+exp(dxy(a(count, :), muXDFG, varXDFG) - dxy(a(count, :), muXDBG, varXDBG) + alphaFG - alphaBG));
    %FG(count) = 1/(1+exp(dxy(a(count, :), muXDFG, varXDFG) - dxy(a(count, :), muXDBG, varXDBG) + alphaFG - alphaBG));
    gBG(count) = 1/(1+exp(dxy(a(count, :), mu1BG, varBG1) - dxy(a(count, :), mu1FG, varFG1) + alphaBG - alphaFG));
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
falseness2(j,1) = sum(sum(xor(A2, d))) / (255*277);
end
%1 - 0.782

