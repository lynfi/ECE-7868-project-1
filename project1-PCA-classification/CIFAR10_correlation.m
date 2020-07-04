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
addpath('cifar-10-batches-mat')

%% Read the data and calculate the PCA for each group
total_sample = 0;   %total_sample
group = []; %record the group of each sample
img_matrix = [];

for i = 1:5
    disp(['Loading dataset ', num2str(i)])
    file_name = ['data_batch_', num2str(i), '.mat'];
    load(file_name);
    labels = labels + 1;
    group = [group; labels];
    img_matrix = [img_matrix; data];
end

load('test_batch.mat')
labels = labels + 1;
total_sample = length(labels);
img_matrix = double(img_matrix');
img_matrix = img_matrix ./ sqrt(sum(img_matrix.^2));
data = double(data');
data = data ./ sqrt(sum(data.^2));

%% Classification using the correlation (knn method)
k_categories = {1, 3, 5, 10, 15};
accuracy = zeros(6,1);
cr = data' * img_matrix;
for v = 1:5
    k = k_categories{v}; %the k I c hoose
    disp(['K = ', num2str(k)])
    correctness = 0;    %num of correctness
    for i = 1:total_sample
        [B, I] = maxk(cr(i,:), k);
        g = mode(group(I));
        if g == labels(i)
            correctness = correctness + 1;
        end
    end
    accuracy(v) = correctness / total_sample
end