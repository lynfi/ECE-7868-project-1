%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Project 1 for ECE 7868
%PCA
%Author: L.Z.
%Date: 10/26/2018
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc
clear all

%% Set the directory
mfile_name = mfilename('fullpath');    %only work when run the m.file instead of debuging.
[pathstr,name,ext] = fileparts(mfile_name);
cd(pathstr);
addpath('test2')
addpath(genpath('eth80-cropped256'))

%% Read the data and calculate the PCA for each group
categories = {'apple','car','cow','cup','dog','horse','pear','tomato'}; %{'apple', 'car'};
for k = 1:length(categories)
    category = categories{k};
    disp(['Reading images of ', category])
    img_matrix{k}=[];     
    if (k ~= 2)
        for i = 1:10
            img_path = strcat('eth80-cropped256/', category, int2str(i), '/maps');
            img_list = dir([img_path, '/*.png']);
            img_num = length(img_list);
            for j = 1:img_num
                image = imread([img_list(j).folder, '/',img_list(j).name]);  
                img_matrix{k} = [img_matrix{k}; image(:)'];
            end
        end
    else
        car_list = {1,2,3,5,6,7,9,11,12,14};
        for ii = 1:length(car_list)
            i = car_list{ii};
            img_path = strcat('eth80-cropped256/', category, int2str(i), '/maps');
            img_list = dir([img_path, '/*.png']);
            img_num = length(img_list);
            for j = 1:img_num
                image = imread([img_list(j).folder, '/',img_list(j).name]);
                img_matrix{k} = [img_matrix{k}; image(:)'];
            end
        end
    end

    %calcuate the PCA for each category
    [coeff{k}, score, latent, tsquared, explained{k}, mu{k}] = pca(double(img_matrix{k}));
end



%% NO Leave-one-out Testing in different variance

accuracy = zeros(6,1);
for var = 1:4  
    var_percent = var * 2 * 10; %percent of variance
    disp(['percent of variance ', num2str(var_percent)])
    total_sample = 0;   %total_sample
    correctness = 0;    %num of correctness
    img_num = zeros(length(categories),1);  %img num in each category
    %calculate the PC in each category
    num = zeros(length(categories), 0);
    for k = 1:length(categories)
        v = 0;
        for j = 1:length(explained{k}) %num of PCs 
            v = v + explained{k}(j);
            if v>var_percent 
                break 
            end
        end
        num(k) = j;
    end
    
    for k = 1:length(categories)
        category = categories{k};
        img_num(k) = size(img_matrix{k}, 1);
        progress = 1;   %show the progress
        ex = explained{k};  %percent of variance for each PCs in category k
        co = coeff{k};  %all PCs in category k

        subspace_vec = co(:,1:num(k));
        for i = 1:img_num(k)
            if (i/img_num(k) > (progress * 0.2))
                disp (['Classification of ', category, num2str(progress * 0.2)])
                progress = progress + 1;
            end
            total_sample = total_sample + 1;
            correctness = correctness + 1;
            train = double(img_matrix{k});
            sample = train(i, :); 
            %calculate the distance bt the sample and its projection in
            %the correct group
            projection = (sample - mean(train)) * subspace_vec * subspace_vec' + mean(train);
            dis = sum((sample - projection).^2);

            %calculate the distance bt the sample and its projection in other
            %groups
            for j = 1: length(categories)
                if (j ~= k)
                    v = 0;
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