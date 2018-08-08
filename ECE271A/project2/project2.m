clear; close all; clc;
load('TrainingSamplesDCT_8_new 2.mat')
n_bg=length(TrainsampleDCT_BG);
n_fg=length(TrainsampleDCT_FG);
p_bg=n_bg/(n_bg+n_fg);
p_fg=1-p_bg;
train_bg = TrainsampleDCT_BG;
train_fg = TrainsampleDCT_FG;
%{
train_bg(:,1)=0;
train_fg(:,1)=0;

%compute p(x|background)
[bg_max,bg_id]=max(abs(train_bg),[],2);
pdf_bg=zeros(1,64);
for i=1:length(bg_id)
    pdf_bg(bg_id(i))=pdf_bg(bg_id(i))+1/length(bg_id);
end
%compute p(x|foreground)        
[fg_max,fg_id]=max(abs(train_fg),[],2);
pdf_fg=zeros(1,64);
for i=1:length(fg_id)
    pdf_fg(fg_id(i))=pdf_fg(fg_id(i))+1/length(fg_id);
end
%}
%{
for i=1
    mu_bg(i)=mean(train_bg(:,i));
    std_bg(i)=sum((train_bg(:,i)-mu_bg(i)).^2)/n_bg;
    mu_fg(i)=mean(train_fg(:,i));
    std_fg(i)=sum((train_fg(:,i)-mu_fg(i)).^2)/n_fg;    
end
%}
mu_bg=mean(train_bg);
std_bg=std(train_bg);
mu_fg=mean(train_fg);
std_fg=std(train_fg);
%{
figure
for i=1:64
    subplot(8,8,i)
    x=-5:0.01:5;
    y=normpdf(x,mu_bg(i),std_bg(i));
    plot(x,y)
    hold on;
    y=normpdf(x,mu_fg(i),std_fg(i));
    plot(x,y)
    title(i)
end
%}
%all 64 features
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
%Estimate covariance of class cheetah and grass
sigma_bg=cov(train_bg);
sigma_fg=cov(train_fg);
mask=zeros((m-7)*(n-7),1);
%Apply BDR for Gaussian Model
for i=1:length(coefs)
    if log(p_bg)-0.5*log((2*pi)^64*det(sigma_bg))-0.5*(coefs(i,:)-mu_bg)*inv(sigma_bg)*(coefs(i,:)-mu_bg)' < log(p_fg)-0.5*log((2*pi)^64*det(sigma_fg))-0.5*(coefs(i,:)-mu_fg)*inv(sigma_fg)*(coefs(i,:)-mu_fg)' 
       mask(i)=1;
    end
end
newmask=zeros(m-7,n-7);
for row=1:m-7
    newmask(row,:)=mask(((row-1)*(n-7)+1):row*(n-7))';
end
figure
imshow(newmask,[])

%max 8 best features
max8=[1,11,14,23,25,27,32,40];
for i=1:8
    train8_bg(:,i)=train_bg(:,max8(i));
    train8_fg(:,i)=train_fg(:,max8(i));
    coefs8(:,i)=coefs(:,max8(i));
end
%Estimate 8 dimension mu & sigma
mu8_bg=mean(train8_bg);
mu8_fg=mean(train8_fg);
sigma8_bg=cov(train8_bg);
sigma8_fg=cov(train8_fg);
mask8=zeros((m-7)*(n-7),1);
%Apply BDR only on 8 dimension
for i=1:length(coefs8)
    if log(p_bg)-0.5*log((2*pi)^8*det(sigma8_bg))-0.5*(coefs8(i,:)-mu8_bg)*inv(sigma8_bg)*(coefs8(i,:)-mu8_bg)' < log(p_fg)-0.5*log((2*pi)^8*det(sigma8_fg))-0.5*(coefs8(i,:)-mu8_fg)*inv(sigma8_fg)*(coefs8(i,:)-mu8_fg)' 
       mask8(i)=1;
    end
end
newmask8=zeros(m-7,n-7);
for row=1:m-7
    newmask8(row,:)=mask8(((row-1)*(n-7)+1):row*(n-7))';
end
figure
imshow(newmask8,[])
    
    


    
    