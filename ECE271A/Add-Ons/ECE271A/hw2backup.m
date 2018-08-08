fprintf('Loading movie ratings dataset.\n\n');

%  Load data TrainsmapleDCT_BG & TrainsampleDCT_FG
load ('TrainingSamplesDCT_8_new.mat');

%problem a: the prior probabilities not solved, here used old one
[rowBG columnBG] = size(TrainsampleDCT_BG);
[rowFG columnFG] = size(TrainsampleDCT_FG);
priorBG = rowBG / (rowBG + rowFG);
priorFG = rowFG / (rowBG + rowFG);

mu1 = sum(TrainsampleDCT_BG)/1053;
mu1ext = repmat(mu1,1053,1);
mu2 = sum(TrainsampleDCT_FG)/250;
mu2ext = repmat(mu2, 250,1);

sigma1 = sum((TrainsampleDCT_BG - mu1ext).^2);
sigma2 = sum((TrainsampleDCT_FG - mu2ext).^2);

x = -1.2:0.05:4.3;
a = size(x,2);
y1 = zeros(64, a);
y2 = zeros(64, a);
for count = 1:34
    y1(count, :) = normpdf(x,mu1(count), sigma1(count));
    y2(count, :) = normpdf(x,mu2(count), sigma2(count));
    plot(x,y1(count, :),'-b',x,y2(count, :),'-r');
    figure;
end

%we need to calculate the covariance this old way is not right;
%sig1 = diag(sigma1);
%sig2 = diag(sigma2);
dimension = 64;
sig1 = zeros(dimension);
sig2 = zeros(dimension);
% for i = 1:dimension
%     for j = 1:dimension
%         sig1(i,j) = cov(TrainsampleDCT_BG(i, :), TrainsampleDCT_BG(j, :));
%     end
% end
sig1 = cov(TrainsampleDCT_BG);
sig2 = cov(TrainsampleDCT_FG);


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
gBG = zeros(1,65224);
gFG = zeros(1,65224);
g2 = zeros(1,65224);
for count = 1:65224
    gBG(count) = 1/(1+exp(dxy(a(count, :), mu1, sig1) - dxy(a(count, :), mu2, sig2) + alphaBG - alphaFG));
    %g2(count) = exp(-0.5 * );
    gFG(count) = 1/(1+exp(dxy(a(count, :), mu2, sig2) - dxy(a(count, :), mu1, sig1) + alphaFG - alphaBG));
    if(gBG(count) < 0.5)
        d(rem(count,248)+1, floor(count/248)+1) = 1;
    end
end
figure;
Cmask = mat2gray(d);
imshow(Cmask);


[A2 B2] = imread('cheetah_mask.bmp');
A2 = A2/255;
falseness = sum(sum(xor(A2, d))) / (255*277);

    

