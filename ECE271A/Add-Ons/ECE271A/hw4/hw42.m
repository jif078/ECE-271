fprintf('Loading cheeta data.\n\n');

%  Load data TrainsmapleDCT_BG & TrainsampleDCT_FG
load ('TrainingSamplesDCT_8_new.mat');

%problem a: the prior probabilities not solved, here used old one
[rowBG columnBG] = size(TrainsampleDCT_BG);
[rowFG columnFG] = size(TrainsampleDCT_FG);
dimensions = [1, 2, 4, 8, 16, 32, 40, 48, 56, 64];
numberOfDimensions = 10;
%generate init point randomly
c = 8; %number of components
di = 2 %number of dimensions

%for background
randInit = rand(c, di);
% initial mu
muBG = randInit;
% random init covariance
covarianceBG = zeros(di, di, c);
for i = 1 : c
    covarianceBG(:, :, i) = diag(rand(1,di));
end
% initial prior
priornumBG = ones(1,c);
priorBG = priornumBG / c;

%init splited dataset
datasetBG = cell(1,c);
for i = 1 : c
    datasetBG{i} = muBG(i, :); 
end
for i = 1 : rowBG
    % E step, get the
    assign = 0;
    dmin = inf;
    for j = 1 : c
        g = dxy(muBG(j, :), TrainsampleDCT_BG(i,1 : di), covarianceBG(:, :, j)) + log(det(covarianceBG(:, :, j))) - 2 * log(priorBG(j));
        if g < dmin
            dmin = g;
            assign = j;
        end
    end
    % M step, update the values of numbers of the component, dataset of the component, prior, mu and variance 
    j = assign;
    priornumBG(j) = priornumBG(j) + 1;
    datasetBG{j} = [datasetBG{j}; TrainsampleDCT_BG(i, 1:di)];
    priorBG = priornumBG / sum(priornumBG); % may be right
    muBG(j, :) = ((priornumBG(j) - 1) * muBG(j, :) + TrainsampleDCT_BG(i, 1:di)) / priornumBG(j);
    covarianceBG(:, :, j) = diag(var(datasetBG{j}));
end

%for foreground
randInit = rand(c, di);
% initial mu
muFG = randInit;
% random init covariance
covarianceFG = zeros(di, di, c);
for i = 1 : c
    covarianceFG(:, :, i) = diag(rand(1,di));
end
% initial prior
priornumFG = ones(1,c);
priorFG = priornumFG / c;

%init splited dataset
datasetFG = cell(1,c);
for i = 1 : c
    datasetFG{i} = muFG(i, :); 
end
for i = 1 : rowFG
    % E step, get the
    assign = 0;
    dmin = inf;
    for j = 1 : c
        g = dxy(muFG(j, :), TrainsampleDCT_FG(i,1 : di), covarianceFG(:, :, j)) + log(det(covarianceFG(:, :, j))) - 2 * log(priorFG(j));
        if g < dmin
            dmin = g;
            assign = j;
        end
    end
    % M step, update the values of numbers of the component, dataset of the component, prior, mu and variance 
    j = assign;
    priornumFG(j) = priornumFG(j) + 1;
    datasetFG{j} = [datasetFG{j}; TrainsampleDCT_FG(i, 1:di)];
    priorFG = priornumFG / sum(priornumFG); % may be right
    muFG(j, :) = ((priornumFG(j) - 1) * muFG(j, :) + TrainsampleDCT_FG(i, 1:di)) / priornumFG(j);
    covarianceFG(:, :, j) = diag(var(datasetFG{j}));
end

%part II: do the classification

% first: read the pic and change it
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

