clear;clc;close all;
%load cheetah mask
msk=imread('cheetah_mask.bmp');
msk=im2double(msk);
%load cheetah
im=imread('cheetah.bmp');
im=im2double(im);
[m,n]=size(im);
coefs=zeros((m-7)*(n-7),64);
for i=1:m-7
    for j=1:n-7
        block=im(i:i+7,j:j+7);
        coef=dct2(block);
        coefs((i-1)*(n-7)+j,:)=zigzag8(coef);
    end
end
%load training data
load('TrainingSamplesDCT_8_new 2.mat')
n_bg=length(TrainsampleDCT_BG);
n_fg=length(TrainsampleDCT_FG);
p_bg=n_bg/(n_bg+n_fg);
p_fg=1-p_bg;
train_bg = TrainsampleDCT_BG;
train_fg = TrainsampleDCT_FG;
C=8;%C is the number of Mixture Gaussian Models
[mu_c_bg,sigma_c_bg,pi_c_bg,iter_bg]=EM(train_bg,C);
[mu_c_fg,sigma_c_fg,pi_c_fg,iter_fg]=EM(train_fg,C);
%predict
mask=zeros((m-7)*(n-7),1);
for j=1:length(coefs)
    % all are from training set
    %G(x,mu,sigma^2)
    if EM_probability(coefs(j,:),mu_c_bg,sigma_c_bg,pi_c_bg) < EM_probability(coefs(j,:),mu_c_fg,sigma_c_fg,pi_c_fg) 
        mask(j)=1;
    end
end
newmask=zeros(m-7,n-7);
for row=1:m-7
    newmask(row,:)=mask(((row-1)*(n-7)+1):row*(n-7))';
end
figure
imshow(newmask,[])
N=0;
for r=1:m-7
    for c=1:n-7
        if msk(r,c)~=newmask(r,c)
            N=N+1;
        end
    end
end
Err_EM=N/255/270;    