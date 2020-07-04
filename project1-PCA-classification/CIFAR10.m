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
for k = 1:10
    img_matrix{k} = []; %img num in each category
end
img_num = zeros(10, 1);
for i = 1:5
    disp(['Loading dataset ', num2str(i)])
    file_name = ['data_batch_', num2str(i), '.mat'];
    load(file_name);
    labels = labels + 1;
    for j = 1:length(labels)
        k = labels(j);
        img_num(k) = img_num(k) + 1;
        img_matrix{k} = [img_matrix{k}; data(j,:)];
    end
end

for k = 1:10
    %calcuate the PCA for each category
    [coeff{k}, score, latent, tsquared, explained{k}, mu{k}] = pca(double(img_matrix{k}));
end

load('test_batch.mat')
labels = labels + 1;

%% Testing in different variance
accuracy = zeros(5,1);
for var = 1:4
    var_percent = var * 2 * 10; %percent of variance
    disp(['percent of variance ', num2str(var_percent)])
    total_sample = 0;   %total_sample
    correctness = 0;    %num of correctness
    %calculate the PC in each category
    num = zeros(10, 0);
    for k = 1:10
        v = 0;
        for j = 1:length(explained{k}) %num of PCs
            v = v + explained{k}(j);
            if v>var_percent
                break
            end
        end
        num(k) = j;
    end
    
    for i = 1:length(labels)
        k = labels(i);
        total_sample = total_sample + 1;
        correctness = correctness + 1;
        sample = double(data(i,:));
        co = coeff{k};
        subspace_vec = co(:,1:num(k));
        
        %calculate the distance bt the sample and its projection in
        %the correct group
        projection = (sample - mu{k}) * subspace_vec * subspace_vec' + mu{k};
        dis = sum((sample - projection).^2);
        
        %calculate the distance bt the sample and its projection in other
        %groups
        for j = 1:10
            if (j ~= k)
                co = coeff{j};
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
    accuracy(var) = correctness / total_sample
end
