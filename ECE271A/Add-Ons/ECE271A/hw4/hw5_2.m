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
        temp = A2(j:j+7, i:i+7);
        temp = dct2(temp);
        a((i-1)*248+j, :) = tras264(temp);
    end
end

% read the cheetah_mask and get the A2
[A2 B2] = imread('cheetah_mask.bmp');
A2 = A2/255;

di = 64;
c = 4;

    %learn 5 paramset of background
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
   
    cnt2 = 1;
    for c2 = [1, 2, 4, 8, 16, 32]
        
        %get the parameters all_params is cell for all 6 * 3 cells
        muBG = all_param{cnt2,1};
        varBG = all_param{cnt2,2};
        priorBG = all_param{cnt2,3};
        muFG = all_param{cnt2,4};
        varFG = all_param{cnt2,5};
        priorFG = all_param{cnt2,6};       
        
        
        errorrate = zeros(25, 11);
        cnt = 1;   
        % cut process for all the classes
        %for di2 = [1,2,4,8,16,24,32,40,48,56,64]
        for di2 = [16,24,32,40]
            %cut the parameters
            varBG2 = cell(5);
            muBG2 = cell(5);
            varFG2 = cell(5);
            muFG2 = cell(5);
            for i = 1:5
                varBG2{i} = varBG{i}(:, 1:di2);
                muBG2{i} = muBG{i}(:, 1:di2);
                varFG2{i} = varFG{i}(:, 1:di2);
                muFG2{i} = muFG{i}(:, 1:di2);
            end

            count2 = 1;
            for n1 = 1:5
                for n2 = 1:5
                    %totally 25 classifiers
                    % second: count the PBG and PFG
                    PBG = zeros(1,65224);
                    PFG = zeros(1,65224);
                    d = zeros(255,270);

                    for count = 1:65224
                        % the mu and variance are different, the prior is same
                        % the a should cut the dimension
                        PBG(count) = CountProb(a(count, 1:di2), muBG2{n1}, varBG2{n1}, priorBG{n1}, c);
                        PFG(count) = CountProb(a(count, 1:di2), muFG2{n2}, varFG2{n2}, priorFG{n2}, c);
                        if(PBG(count) * prior_BG < PFG(count) * prior_FG)
                            d(rem(count,248)+1, floor(count/248)+1) = 1;
                        end
                    end
                    %figure;
                    Cmask = mat2gray(d);
                    %imshow(Cmask);
                    err = sum(sum(xor(A2, d))) / (255*277);
                    %fprintf('the err of %d dimension classifier NO.%d is %d \n', di2, count2, err);
                    errorrate(count2,cnt) = err;
                    count2 = count2 + 1;
                end
            end    
            cnt = cnt + 1;  
        end
        all_error{cnt2} = errorrate;
        cnt2 = cnt2 + 1;
    end