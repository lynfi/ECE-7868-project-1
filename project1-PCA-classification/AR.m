%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Project 1 for ECE 7868
%Images Classifier
%Author: L.Z.
%Date: 09/14/2018
%Remark: Running time on Macbook Pro 2015:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc
clear all
%% Set the directory
mfile_name = mfilename('fullpath'); %only work when run the m.file instead of debuging.
[pathstr,name,ext] = fileparts(mfile_name);
cd(pathstr);
cd
addpath('test2')
addpath(genpath('eth80-cropped256'))

%% Read the data and calculate the PCA for each group
num_face = 100;
img_path = 'test2';

%Male's images
for k = 1:floor(num_face/2)
    disp(['Reading images of M ', num2str(k)])
    img_matrix{k} = [];
    if k>=10
        img_list = dir([img_path, '/M-0', num2str(k),'*.bmp']);
    else
        img_list = dir([img_path, '/M-00', num2str(k),'*.bmp']);
    end
    img_num = length(img_list);
    for j = 1:img_num
        image = imread([img_list(j).folder, '/',img_list(j).name]); 
        image = rgb2gray(image);
        img_matrix{k} = [img_matrix{k}; image(:)'];
    end
    %calcuate the PCA for each category
    [coeff{k}, score, latent, tsquared, explained{k}, mu{k}] = pca(double(img_matrix{k}));

end

%Female's images
for k = 51:num_face
    disp(['Reading images of W ', num2str(k-50)])
    img_matrix{k} = [];
    if (k-50)>=10
        img_list = dir([img_path, '/W-0', num2str(k-50),'*.bmp']);
    else
        img_list = dir([img_path, '/W-00', num2str(k-50),'*.bmp']);
    end
    img_num = length(img_list);
    for j = 1:img_num
        image = imread([img_list(j).folder, '/',img_list(j).name]);  
        image = rgb2gray(image);
        img_matrix{k} = [img_matrix{k}; image(:)'];
    end    
    %calcuate the PCA for each category
    [coeff{k}, score, latent, tsquared, explained{k}, mu{k}] = pca(double(img_matrix{k}));
end

%% Leave-one-out Testing in different variance

accuracy = zeros(5,1);
for var = 1:4
    var_percent = var * 2 * 10; %percent of variance
    disp(['percent of variance ', num2str(var_percent)])
    total_sample = 0;   %total_sample
    correctness = 0;    %num of correctness
    img_num = zeros(num_face, 1);  %img num in each category   
    
    num = zeros(num_face, 0);
    for k = 1:num_face
        v = 0;
        for j = 1:length(explained{k}) %num of PCs 
            v = v + explained{k}(j);
            if v>var_percent 
                break 
            end
        end
        num(k) = j;
    end
    
    for k = 1:num_face
        img_num(k) = size(img_matrix{k}, 1);
        for i = 1:img_num(k)
            total_sample = total_sample + 1;
            correctness = correctness + 1;
            train = double(img_matrix{k});
            sample = train(i, :); 
            train(i,:) = [];    %leave-one-out
            [co, sc, la, ts, ex] = pca(train);  %recalcuate the PCA for this face
            
            v = 0;
            for j = 1:length(ex) %num of PCs
                v = v + ex(j);
                if v>var_percent 
                    break 
                end
            end
            subspace_vec = co(:,1:j);
            
            %calculate the distance bt the sample and its projection in
            %the correct group
            projection = (sample - mean(train)) * subspace_vec * subspace_vec' + mean(train);
            dis = sum((sample - projection).^2);

            %calculate the distance bt the sample and its projection in other
            %groups
            for j = 1:num_face
                if (j ~= k)
                    co = coeff{j};
                    ex = explained{j};
                    subspace_vec = co(:,1:num(j));
                    projection = (sample - mu{j}) * subspace_vec * subspace_vec' + mu{j};
                    d = sum((sample - projection).^2);
                    if (d < dis)    %if d<dis then our classification will be incorrect
                        correctness = correctness - 1;
                        break
                    end
                end
            end
        end
    end
    accuracy(var) = correctness / total_sample
end