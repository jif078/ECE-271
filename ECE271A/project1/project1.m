clear; close all; clc;
load('TrainingSamplesDCT_8.mat')
p_bg=length(TrainsampleDCT_BG)/(length(TrainsampleDCT_BG)+length(TrainsampleDCT_FG));
p_fg=1-p_bg;
train_bg = TrainsampleDCT_BG;
train_bg(:,1)=0;
train_fg = TrainsampleDCT_FG;
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
%
figure
histogram(bg_id,1:64,'Normalization','pdf')
title('pdf of background')
figure
histogram(fg_id,1:64,'Normalization','pdf')
title('pdf of foreground')
%}
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
%
coefs(:,1)=0;
[coefs_max,coefs_id]=max(abs(coefs),[],2);
for i=1:length(coefs_id)
    if pdf_bg(coefs_id(i))*p_bg < pdf_fg(coefs_id(i))*p_fg
       mask(i)=1;
    elseif pdf_bg(coefs_id(i))*p_bg > pdf_fg(coefs_id(i))*p_fg
       mask(i)=0;
    end
end
newmask=zeros(m-7,n-7);
for row=1:m-7
    newmask(row,:)=mask(((row-1)*(n-7)+1):row*(n-7))';
end
figure
imshow(im,[])
figure
imshow(newmask,[])



