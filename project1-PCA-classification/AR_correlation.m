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

%% Read the data and calculate correlation coefficients
num_face = 100;
img_path = 'test2';
total_sample = 0;   %total_sample
group = []; %record the group of each sample
img_matrix=[];
%Male's images
disp('Reading images of M ')
for k = 1:floor(num_face/2)    
    if k>=10
        img_list = dir([img_path, '/M-0', num2str(k),'*.bmp']);
    else
        img_list = dir([img_path, '/M-00', num2str(k),'*.bmp']);
    end
    img_num = length(img_list);
    for j = 1:img_num
        image = imread([img_list(j).folder, '/',img_list(j).name]); 
        image = rgb2gray(image);
        total_sample = total_sample + 1; 
        group = [group; k];
        img_matrix = [img_matrix; image(:)'];
    end
end

%Female's images
disp('Reading images of W ')
for k = 51:num_face    
    if (k-50)>=10
        img_list = dir([img_path, '/W-0', num2str(k-50),'*.bmp']);
    else
        img_list = dir([img_path, '/W-00', num2str(k-50),'*.bmp']);
    end
    img_num = length(img_list);
    for j = 1:img_num
        image = imread([img_list(j).folder, '/',img_list(j).name]);  
        image = rgb2gray(image);
        total_sample = total_sample + 1; 
        group = [group; k];
        img_matrix = [img_matrix; image(:)'];
    end    
end

% Calculate the covariance
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