
fprintf('Loading dataset.\n\n');

%(a) conider the training setD1 and strategy1. For each class, compute the
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


%read the image
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
d = zeros(255,270);
%2pi^d-- the d is dimension
alphaBG = log(((2 * pi)^64) * det(sig1)) - 2*log(priorBG);
alphaFG = log(((2 * pi)^64) * det(sig2)) - 2*log(priorFG);

% alphaBG = log(((2 * pi)^64) * det(varBG0)) - 2*log(priorBG);
% alphaFG = log(((2 * pi)^64) * det(varFG0)) - 2*log(priorFG);


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

%ML method falsenness 14.29%