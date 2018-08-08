
%problem3:
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

THold = THold*1.1;
%vector d is the estimation of the window
d = zeros(255,270);
for i = 1:65224
    [sortNum sortPosition] = sort( abs( a(i, :) ) );
    if((XFG(sortPosition(63))) / (XBG(sortPosition(63))) > THold)    %means this is the foreground
        d(rem(i,248)+1, floor(i/248)+1) = 1;
    end
    if((XFG(sortPosition(63))) / (XBG(sortPosition(63))) == THold && round(rand))
                d(rem(i,248)+1, floor(i/248)+1) = 1;
    end
end
Cmask = mat2gray(d);
imshow(Cmask);


[A2 B2] = imread('cheetah_mask.bmp');
A2 = A2/255;
falseness = sum(sum(xor(A2, d))) / (255*277);
numberOfFG = sum(sum(A2));
numberOfBG = 255*270 - numberOfFG;






