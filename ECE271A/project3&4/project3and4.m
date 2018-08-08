clear;clc;close all;
load('TrainingSamplesDCT_subsets_8.mat')
load('Alpha.mat');
%Using Strategy 1(mu_cheetah=1 0 0 ...; mu_grass=3 0 0 ...)
load('Prior_1.mat');
%load('Prior_2.mat');
msk=imread('cheetah_mask.bmp');
msk=im2double(msk);

%ML Estimation on Training Set
%mu_hat|grass
mu_d1bg=mean(D1_BG);

%sigma^2|grass
var_d1bg=cov(D1_BG);

%mu_hat|cheetah
mu_d1fg=mean(D1_FG);

%sigma^2|cheetah
var_d1fg=cov(D1_FG);

%P(grass)
prior_bg=length(D1_BG)/( length(D1_BG)+length(D1_FG) );

%P(cheetah)
prior_fg=1-prior_bg;

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

%For every alpha, compute error rate
for i=1:length(alpha)
% for the prior we have know 
% mu0_BG=3 and mu0_FG=1 
% covariance is sigma = diag(alpha * weight)

%Class BG
    % For grass: Prior: P_mu(mu)=G(mu, mu0_BG, sigma0_bg)
    sigma0_bg=diag(alpha(i)*W0);
    %Compute parameters for P_MU|T(mu|D)=G(mu,mu_n,sigma_n^2)
    %w1=(n*sigma0^2)/[sigma^2+n*sigma0^2]
    weight1_bg=sigma0_bg/(sigma0_bg+var_d1bg/length(D1_BG) );
    %w2=I-w1
    weight2_bg=(var_d1bg/length(D1_BG))/(sigma0_bg+var_d1bg/length(D1_BG) );
    %mu_n=w1*mu_ml+w2*mu0
    mu_nbg=weight1_bg*mu_d1bg'+weight2_bg*mu0_BG';
    %sigma_n^2=(sigma^2*sigma0^2)/(sigma^2+n*sigma_n^2)
    sigma_nbg=(var_d1bg*sigma0_bg)/ (var_d1bg+length(D1_BG)*sigma0_bg);
    %P_X|MU(x|mu)=G(x,mu,sigma^2);P_MU|T(mu|D)=G(mu,mu_n,sigma_n^2)
    %P_X|T(x|D)=Integral{P_X|MU(x|mu) * P_MU|T(mu|D)*(d_mu)}
    %P_X|T(x|D) = G(x,0,sigma^2)(convolution)G(x,mu_n,sigma_n^2)
    %P_X|T(x|D) = G(x,mu_n,sigma^2+sigma_n^2)
    sigma_XT_BG=((sigma_nbg+var_d1bg)+(sigma_nbg+var_d1bg)')/2;
    
%Class FG    
    sigma0_fg=diag(alpha(i)*W0);
    weight1_fg=sigma0_fg/(sigma0_fg+var_d1fg/length(D1_FG) );
    weight2_fg=(var_d1fg/length(D1_FG))/(sigma0_fg+var_d1fg/length(D1_FG) );
    mu_nfg=weight1_fg*mu_d1fg'+weight2_fg*mu0_FG';
    sigma_nfg=(var_d1fg*sigma0_fg)/ (var_d1fg+length(D1_FG)*sigma0_fg);
    sigma_XT_FG=((sigma_nfg+var_d1fg)+(sigma_nfg+var_d1fg)')/2;
    
%Apply BDR for Gaussian Model

%PD
mask=zeros((m-7)*(n-7),1);
for j=1:length(coefs) 
    %Predictive Distribution: average all model 
    %P_X|T(x|D) = G(x,mu_n,sigma^2+sigma_n^2)
    if log(mvnpdf(coefs(j,:),mu_nbg',sigma_XT_BG)*prior_bg) < log(mvnpdf(coefs(j,:),mu_nfg',sigma_XT_FG)*prior_fg) 
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
for r=1:248
    for c=1:263
        if msk(r,c)~=newmask(r,c)
            N=N+1;
        end
    end
end
Err_pd(i)=N/255/270;    

%MAP
mask=zeros((m-7)*(n-7),1);
for j=1:length(coefs)
    %Using MAP, choose mu_n(bg or fg), and ML sigma^2 from training set
    %G(x,mu_n,sigma^2)
    if log(mvnpdf(coefs(j,:),mu_nbg',var_d1bg)*prior_bg) < log(mvnpdf(coefs(j,:),mu_nfg',var_d1fg)*prior_fg) 
        mask(j)=1;
    end
end
newmask=zeros(m-7,n-7);
for row=1:m-7
    newmask(row,:)=mask(((row-1)*(n-7)+1):row*(n-7))';
end

N=0;
for r=1:248
    for c=1:263
        if msk(r,c)~=newmask(r,c)
            N=N+1;
        end
    end
end
Err_map(i)=N/255/270;    

%ML
mask=zeros((m-7)*(n-7),1);
for j=1:length(coefs)
    % all are from training set
    %G(x,mu,sigma^2)
    if log(mvnpdf(coefs(j,:),mu_d1bg,var_d1bg)*prior_bg) < log(mvnpdf(coefs(j,:),mu_d1fg,var_d1fg)*prior_fg) 
        mask(j)=1;
    end
end
newmask=zeros(m-7,n-7);
for row=1:m-7
    newmask(row,:)=mask(((row-1)*(n-7)+1):row*(n-7))';
end

N=0;
for r=1:248
    for c=1:263
        if msk(r,c)~=newmask(r,c)
            N=N+1;
        end
    end
end
Err_ml(i)=N/255/270;    
    
end
figure(10)
plot(alpha,Err_map)
hold on;
plot(alpha,Err_ml)
hold on;
plot(alpha,Err_pd)
grid on;
set(gca,'XScale', 'log')
legend('MAP','ML','Pred')
title('Pred is better when prior sigma0^2 is small, which means you are confident on MU')



