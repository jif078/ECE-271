%ECE 271A statistical learning 
%computer problem1: segment the "cheetah" image into two components,
%cheetah and grassland(background)

%
fprintf('Loading movie ratings dataset.\n\n');

%  Load data TrainsmapleDCT_BG & TrainsampleDCT_FG
load ('TrainingSamplesDCT_8.mat');

%problem a: the prior probabilities
[rowBG columnBG] = size(TrainsampleDCT_BG);
[rowFG columnFG] = size(TrainsampleDCT_FG);
priorBG = rowBG / (rowBG + rowFG);
priorFG = rowFG / (rowBG + rowFG);
%prior is the Pbg Pfg



%problem b: 
XBG = zeros(64,1);
XFG = zeros(64,1);
    %get the X of background
for count = 1: rowBG
    [sortNum sortPosition] = sort(TrainsampleDCT_BG(count, :));
    %XBG(count) = sortPosition(63); 
    XBG(sortPosition(63)) = XBG(sortPosition(63)) + 1;
end
    %get the X of frontground
for count = 1: rowFG
    [sortNum sortPosition] = sort(TrainsampleDCT_FG(count, :));
    XFG(sortPosition(63)) = XFG(sortPosition(63)) + 1;
end
XBG = XBG ./ rowBG;
XFG = XFG ./ rowFG;
% then we got the P(x|cheeta) = XFG
%                 P(x|grass)  = XBG
%set1to64 = [1:64];
%set1to64 = transposse(set1to64);
%meanBG = sum(set1to64 .* XBG);
%meanFG = sum(set1to64 .* XFG);
%testing
THold = priorBG / priorFG;

v = zeros(250,1);
for i = 1:250
    [sortNum sortPosition] = sort(TrainsampleDCT_FG(i, :)); 
    v(i) = XFG(sortPosition(63)) / (XBG(sortPosition(63)));
end

v2 = zeros(1053,1);
for i = 1:1053
    [sortNum sortPosition] = sort(TrainsampleDCT_BG(i, :)); 
    v2(i) = XFG(sortPosition(63)) / (XBG(sortPosition(63)));
end

right = 0;
for i = 1:250
    [sortNum sortPosition] = sort(TrainsampleDCT_FG(i, :));
    if((XFG(sortPosition(63))) / (XBG(sortPosition(63))) >= THold)    %there can do some optimization for better
        right = right +1;
    end
end

rightBG = 0
for i = 1:1053
    [sortNum sortPosition] = sort(TrainsampleDCT_BG(i, :));
    if((XFG(sortPosition(63))) / (XBG(sortPosition(63))) < THold)
        rightBG = rightBG +1;
    end
end


%problem3:
%read the image
[A, B] = imread('cheetah.bmp');
% A is the degree of the pixels 
%B is the color map from 0-149 degrees
A2 = im2double(A);
A3 = [A2, A2(:, 264:270)];
A3 = [A3; A3(249:255, :)];

% a is the matrix of sliding windows
%a = zeros(65224, 64);
a= zeros(68850, 64);
% for i = 1:(270-7)  %colomns
%     for j = 1:(255-7) %rows
%         temp = A2(j:j+7, i:i+7)
%         temp = dct2(temp);
%         a((i-1)*248+j, :) = tras264(temp);
%     end
% end
for i = 1:(277-7)  %colomns
    for j = 1:(262-7) %rows
        temp = A3(j:j+7, i:i+7);
        temp = dct2(temp);
        a((i-1)*255+j, :) = tras264(temp);
    end
end
THold = THold;
%vector d is the estimation of the window
d = zeros(255,270);
% for i = 1:65224
%     [sortNum sortPosition] = sort( abs( a(i, :) ) );
%     if((XFG(sortPosition(63))) / (XBG(sortPosition(63))) > THold)    %means this is the foreground
%         d(rem(i,248)+1, floor(i/248)+1) = 1;
%     end
%     if((XFG(sortPosition(63))) / (XBG(sortPosition(63))) == THold && round(rand))
%                 d(rem(i,248)+1, floor(i/248)+1) = 1;
%     end
% end

for i = 1:68850
    [sortNum sortPosition] = sort( abs( a(i, :) ) );
    if((XFG(sortPosition(63))) / (XBG(sortPosition(63))) > THold)    %means this is the foreground
        d(rem(i,255)+1, floor(i/255)+1) = 1;
    end
    if((XFG(sortPosition(63))) / (XBG(sortPosition(63))) == THold && round(rand))
                d(rem(i,255)+1, floor(i/255)+1) = 1;
    end
end

Cmask = mat2gray(d);
imshow(Cmask);


[A2 B2] = imread('cheetah_mask.bmp');
A2 = A2/255;
falseness = sum(sum(xor(A2, d))) / (255*277);

%
%actualFG = sum(sum(A2));
%actualBG = 255*270 - numberOfFG;
%predFG =  
%Precision =
%fFG = 
%fBG = 
