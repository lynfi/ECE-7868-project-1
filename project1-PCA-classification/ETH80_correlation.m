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
mfile_name = mfilename('fullpath');    %only work when run the m.file instead of debuging.
[pathstr,name,ext] = fileparts(mfile_name);
cd(pathstr);
addpath('test2')
addpath(genpath('eth80-cropped256'))

%% Read the data
categories = {'apple','car','cow','cup','dog','horse','pear','tomato'}; %{'apple', 'car'};
total_sample = 0;   %total_sample
group = []; %record the group of each sample
img_matrix=[];
for k = 1:length(categories)
    category = categories{k};
    disp(['Reading images of ', category])
    if (k ~= 2)
        for i = 1:10
            img_path = strcat('eth80-cropped256/', category, int2str(i), '/maps');
            img_list = dir([img_path, '/*.png']);
            img_num = length(img_list);
            for j = 1:img_num
                image = imread([img_list(j).folder, '/',img_list(j).name]);  
                total_sample = total_sample + 1; 
                group = [group; k];
                img_matrix = [img_matrix; image(:)'];
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
                total_sample = total_sample + 1;
                group = [group; k];
                img_matrix = [img_matrix; image(:)'];
            end
        end
    end
end

%% Calculate the covariance
cov = corrcoef(double(img_matrix)');

%% Classification using the correlation (knn method)

k_categories = {1, 3, 5, 10, 15};
accuracy = zeros(6,1);
for v = 1:5 
    k = k_categories{v}; %the k I c hoose
    disp(['K = ', num2str(k)])
    correctness = 0;    %num of correctness
    for i = 1:total_sample
        cr = cov(i,:);
        cr(i) = -1; %delte it self
        [B, I] = maxk(cr, k);
        g = mode(group(I));
        if g == group(i)
            correctness = correctness + 1;
        end
    end
    accuracy(v) = correctness / total_sample
end




