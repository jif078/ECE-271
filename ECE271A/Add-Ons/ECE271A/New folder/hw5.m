% load ('TrainingSamplesDCT_8_new.mat');
% 
% %  Load data of the picture
% % first: read the pic and change it
% [A, B] = imread('cheetah.bmp');
% % A is the degree of the pixels 
% %B is the color map from 0-149 degrees
% A2 = im2double(A);
% 
% % a is the matrix of sliding windows
% a = zeros(65224, 64)
% for i = 1:(270-7)  %colomns
%     for j = 1:(255-7) %rows
%         temp = A2(j:j+7, i:i+7);
%         temp = dct2(temp);
%         a((i-1)*248+j, :) = tras264(temp);
%     end
% end
% 
% % read the cheetah_mask and get the A2
% [A2 B2] = imread('cheetah_mask.bmp');
% A2 = A2/255;

di = 64;
c = 8;

    %learn 5 paramset of Background
    priorBG = cell(5,1);
    muBG = cell(5,1);
    varBG = cell(5,1);
    for num = 1:5
        [prior, mu, var] = learning2(di, c, TrainsampleDCT_BG );
        % save the learning
        priorBG{num} = prior;
        muBG{num} = mu;
        varBG{num} = var;
    end

    %learn 5 paramset of Foreground
    priorFG = cell(5,1);
    muFG = cell(5,1);
    varFG = cell(5,1); 
    for num = 1:5
        [prior, mu, var] = learning2(di, c, TrainsampleDCT_FG );           
        % save the learning
        priorFG{num} = prior;
        muFG{num} = mu;
        varFG{num} = var;
    end
    
    %get the classification prior of 64 dimension
    prior_BG = size(TrainsampleDCT_BG, 1) / (size(TrainsampleDCT_BG, 1) + size(TrainsampleDCT_FG, 1));
    prior_FG = size(TrainsampleDCT_FG, 1) / (size(TrainsampleDCT_BG, 1) + size(TrainsampleDCT_FG, 1));
    errorrate = zeros(25, 11);
    %classification evaluation for 64 dimension
    cnt1 = 0;
    for di = [1,2,4,8,16,24,32,40,48,56,64]
        cnt2 = 1;
        cnt1 = cnt1 + 1;
        for n1 = 1:5
            for n2 = 1:5
                %totally 25 classifiers
                % second: count the PBG and PFG
                PBG = zeros(65224,1);
                PFG = zeros(65224,1);
                d = zeros(255,270);
                for j = 1 : c
                PBG = PBG + gaussianND(a(:, 1:di),muBG{n1}(j, 1:di), varBG{n1}{j}(1:di, 1:di)) * priorBG{n1}(j);
                PFG = PFG + gaussianND(a(:, 1:di),muFG{n2}(j, 1:di), varFG{n2}{j}(1:di, 1:di)) * priorFG{n2}(j);
                end
                PBG2 = PBG .* prior_BG;
                PFG2 = PFG .* prior_FG;
                Prob = PBG2 - PFG2;

                % reshape process
                for count = 1:65224
                    if(Prob(count) < 0)
                        d(rem(count,248)+1, floor(count/248)+1) = 1;
                    end
                end
%                 figure;
%                 Cmask = mat2gray(d);
%                 imshow(Cmask);
                falseness = sum(sum(xor(A2, d))) / (255*277);
                errorrate(cnt2, cnt1) = falseness;
                cnt2 = cnt2 + 1;
                fprintf(' the falseness is %d\n', falseness);
            end
        end
    end