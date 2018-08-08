fprintf('Loading cheeta data.\n\n');

%  Load data TrainsmapleDCT_BG & TrainsampleDCT_FG
load ('TrainingSamplesDCT_8_new.mat');

%  Load data of the picture
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


%problem a: the prior probabilities not solved, here used old one
[rowBG columnBG] = size(TrainsampleDCT_BG);
[rowFG columnFG] = size(TrainsampleDCT_FG);

for di = [1, 2, 4, 8, 16, 32, 40, 48, 56, 64]
    for c = [1,2,4]
        %learn 5 paramset of Background
        priorBG = cell(5);
        muBG = cell(5);
        varBG = cell(5);
        for num = 1:5
            [prior, mu, var] = getRandomParam(di, c);
            [prior, mu, var] = learning(di, c, prior, mu, var, TrainsampleDCT_BG );
            % save the learning
            priorBG{num} = prior;
            muBG{num} = mu;
            varBG{num} = var;
        end
        
        %learn 5 paramset of Foreground
        priorFG = cell(5);
        muFG = cell(5);
        varFG = cell(5); 
        for num = 1:5
            [prior, mu, var] = getRandomParam(di, c);
            [prior, mu, var] = learning(di, c, prior, mu, var, TrainsampleDCT_FG );           
            % save the learning
            priorFG{num} = prior;
            muFG{num} = mu;
            varFG{num} = var;
        end
        
        
        
        
    end
end




% Estep count hij
