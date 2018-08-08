fprintf('Loading cheeta data.\n\n');

%  Load data TrainsmapleDCT_BG & TrainsampleDCT_FG
load ('TrainingSamplesDCT_8_new.mat');

%problem a: the prior probabilities not solved, here used old one
[rowBG columnBG] = size(TrainsampleDCT_BG);
[rowFG columnFG] = size(TrainsampleDCT_FG);

%generate init point randomly
c = 8; %number of components
randInit = rand(c, columnBG);
% initial mu
mu = randInit;
% initial covariance and all zero
covariance = zeros(columnBG, columnBG, c);
% initial prior
priornum = ones(1,c);
prior = priornum / c;
for i = 1 : c
    dataset(1, :, i) = mu(i, :); 
end
for i = 1 : rowBG
    % E step, get the
    assign = 0;
    dmin = inf;
    for j = 1 : c
        g = dxy(mu(j, :), TrainsampleDCT_BG(i,:), covariance(:, :, j)) + log(det(covariance(:, :, j))) - 2 * log(prior(j));
        if temp < dmin
            dmin = temp;
            assign = j;

        end
    end
    % M step, update the values of numbers of the component, dataset of the component, prior, mu and variance 
    j = assign;
    priornum(j) = priornum(j) + 1;
    dataset(priornum(j), :, j) = TrainsampleDCT_BG(i, :);
    prior = priornum / sum(priornum); % may be right
    mu(j, :) = ((priornum(j) - 1) * mu(j, :) + TrainsampleDCT_BG(i, :)) / priornum(j);
    covariance(:, :, j) = diag(var(dataset(:, :, j)));
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

% second: count the PBG and PFG
PBG = zeros(1,65224);
PFG = zeros(1,65224);
d = zeros(255,270);

for count = 1:65224
    PBG = CountProb(a(count, :), muBG{n1}, varBG{n1}, priorBG{n1}, c);
    PFG = CountProb(a(count, :), muFG{n2}, varFG{n2}, priorFG{n2}, c)
    if(PBG(count) < PFG(count))
        
        d(rem(count,248)+1, floor(count/248)+1) = 1;
    end
end
figure;
Cmask = mat2gray(d);
imshow(Cmask);


[A2 B2] = imread('cheetah_mask.bmp');
A2 = A2/255;
falseness = sum(sum(xor(A2, d))) / (255*277);
