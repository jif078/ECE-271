

    alphaBG = log(((2 * pi)^64) * det(varBG0)) - 2*log(priorBG);
    alphaFG = log(((2 * pi)^64) * det(varFG0)) - 2*log(priorFG);

d = zeros(255,270);
for count = 1:65224

    %gFG(count) = 1/(1+exp(dxy(a(count, :), muXDFG, varXDFG) - dxy(a(count, :), muXDBG, varXDBG) + alphaFG - alphaBG));
    %FG(count) = 1/(1+exp(dxy(a(count, :), muXDFG, varXDFG) - dxy(a(count, :), muXDBG, varXDBG) + alphaFG - alphaBG));
    gBG(count) = 1/(1+exp(dxy(a(count, :), mu0_BG, varBG0) - dxy(a(count, :), mu0_FG, varFG0) + alphaBG - alphaFG));
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
    falseness0 = sum(sum(xor(A2, d))) / (255*277);