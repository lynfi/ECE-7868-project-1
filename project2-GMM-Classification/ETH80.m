%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Project 3 for ECE 7868
%Gaussian Mixture Model
%Author: L.Z.
%Date: 10/26/2018
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc
clear all

%% Set the directory
mfile_name = mfilename('fullpath');    %only work when run the m.file instead of debuging.
[pathstr,name,ext] = fileparts(mfile_name);
cd(pathstr);
addpath(genpath('FastICA_25'))
addpath(genpath('eth80-cropped256'))

%% Read the data
categories = {'apple','car','cow','cup','dog','horse','pear','tomato'}; %{'apple', 'car'};
img = [];
for k = 1:length(categories)
    category = categories{k};
    disp(['Reading images of ', category])
    img_matrix{k}=[];     
    if (k ~= 2)
        for i = 1:10
            img_path = strcat('eth80-cropped256/', category, int2str(i));
            img_list = dir([img_path, '/*.png']);
            img_num = length(img_list);
            for j = 1:img_num
                image = imread([img_list(j).folder, '/',img_list(j).name]);  
                image = rgb2gray(image);
                img_matrix{k} = [img_matrix{k}; image(:)'];
            end
        end
    else
        car_list = {1,2,3,5,6,7,9,11,12,14};
        for ii = 1:length(car_list)
            i = car_list{ii};
            img_path = strcat('eth80-cropped256/', category, int2str(i));
            img_list = dir([img_path, '/*.png']);
            img_num = length(img_list);
            for j = 1:img_num
                image = imread([img_list(j).folder, '/',img_list(j).name]);
                image = rgb2gray(image);
                img_matrix{k} = [img_matrix{k}; image(:)'];
            end
        end
    end
end

%% %% Creating the training set and the testing set
img_train = [];
for k = 1:length(categories)
    r = randperm(410, 80);    %we use 20% of data to test
    test{k} = img_matrix{k}(r, :);
    train{k} = img_matrix{k};
    train{k}(r, :) = [];
    img_train = [img_train; train{k}];
end

%% Gaussian Mixture Model in Original space
% Since the dimension of features is much larger than the number of
% observations, it is infeasible to use the Gaussian Mixture Model
% directly, so I used the PCA at first and retain as much information as
% possible. The number of PCs I choose is 100;

disp('GMModel in Original space')
[coeff, score, latent, tsquared, explained, mu_train] = pca(double(img_train));
num_ORI = 60; %# of PCs we use

Original = coeff(:, 1:num_ORI);  %all PCs we use
ORI = {};
GMModel_ORI = {};
ORI_train = [];
for k = 1:length(categories)
    ORI{k} = (double(train{k}) - mu_train) * Original;  %the projection onto subspace
    GMModel_ORI{k} = fitgmdist(ORI{k}, 4, 'CovarianceType', 'diagonal');  %Gaussian Mixture Model
    ORI_train = [ORI_train; ORI{k}];
end

%prediction
correct_ORI = 0;
total = 0;

test_ORI = {};
for k = 1:length(categories)
    test_ORI{k} = (double(test{k}) - mu_train) * Original;
    for  i = 1:80
        total = total + 1;
        correct_ORI = correct_ORI + 1;
        sample = test_ORI{k}(i, :);
        p1 = pdf(GMModel_ORI{k}, sample);
        for j = 1:length(categories)
            if j ~= k
                p2 = pdf(GMModel_ORI{j}, sample);
                %Compare the Likelihood
                if p1 < p2
                    correct_ORI = correct_ORI - 1;
                    break
                end
            end
        end
    end
end
correct_ORI / total

%% Gaussian Mixture Model in PCA subspace
disp('GMModel in PCA')
num_PCA = 40; %# of PCs we use
PC = coeff(:, 1:num_PCA);  %all PCs we use

for k = 1:length(categories)
    PCA = (double(train{k}) - mu_train) * PC;  %the projection onto subspace
    GMModel_PCA{k} = fitgmdist(PCA, 4, 'CovarianceType', 'diagonal');  %Gaussian Mixture Model
end

%prediction
correct_PCA = 0;
total = 0;

for k = 1:length(categories)
    test_PCA = (double(test{k}) - mu_train) * PC;
    for  i = 1:80
        total = total + 1;
        correct_PCA = correct_PCA + 1;
        sample = test_PCA(i, :);
        p1 = pdf(GMModel_PCA{k}, sample);
        for j = 1:length(categories)
            if j ~= k
                p2 = pdf(GMModel_PCA{j}, sample);
                %Compare the Likelihood
                if p1 < p2
                    correct_PCA = correct_PCA - 1;
                    break
                end
            end
        end
    end
end
correct_PCA / total

%% Gaussian Mixture Model in ICA subspace
% When the dimension of the is large, the running time of FastICA is slow,
% so I have to use the PCA to reduce the dimension at first, then use the
% ICA.
disp('GMModel in ICA')
num_ICA = 30;   %# of ICs we use
[IC A W] = fastica(ORI_train, 'verbose', 'off', 'numOfIC', num_ICA);
[IC, R] = mgs(IC');  %Gram-Smith Orthononalization

for k = 1:length(categories)
    ICA = ORI{k} * IC;  %the projection onto subspace
    GMModel_ICA{k} = fitgmdist(ICA, 4, 'CovarianceType', 'diagonal');  %Gaussian Mixture Model
end

%prediction
correct_ICA = 0;
total = 0;

for k = 1:length(categories)
    test_ICA = test_ORI{k} * IC;
    for  i = 1:80
        total = total + 1;
        correct_ICA = correct_ICA + 1;
        sample = test_ICA(i, :);
        p1 = pdf(GMModel_ICA{k}, sample);
        for j = 1:length(categories)
            if j ~= k
                p2 = pdf(GMModel_ICA{j}, sample);
                %Compare the Likelihood
                if p1 < p2
                    correct_ICA = correct_ICA - 1;
                    break
                end
            end
        end
    end
end
correct_ICA / total

%% Gaussian Miture Model and Em in LDA subspace
disp('GMModel in LDA')
num_LDA = 7;

% In order to used the LDA, I first projected the data
% to PCA space, which has dim = 100, and it can cantain the most information of
% the data. Then I used the LDA in the PCA space and finally and I
% projected it back to the original space.
img_train_LDA =[];
label = [];
for k = 1:length(categories)
    label = [label; k * ones(410-80, 1)];
end

W = LDA(ORI_train, label);
LD = W(:, 1:num_LDA);
[LD, R] = mgs(LD);  %normalization

for k = 1:length(categories)
    train_LDA = ORI{k} * LD;  %the projection onto subspace
    GMModel_LDA{k} = fitgmdist(train_LDA, 4, 'CovarianceType', 'diagonal');  %Gaussian Mixture Model
end

%prediction
correct_LDA = 0;
total = 0;

for k = 1:length(categories)
    test_LDA = test_ORI{k} * LD;
    for  i = 1:80
        total = total + 1;
        correct_LDA = correct_LDA + 1;
        sample = test_LDA(i, :);
        p1 = pdf(GMModel_LDA{k}, sample);
        for j = 1:length(categories)
            if j ~= k
                p2 = pdf(GMModel_LDA{j}, sample);
                %Compare the Likelihood
                if p1 < p2
                    correct_LDA = correct_LDA - 1;
                    break
                end
            end
        end
    end
end
correct_LDA / total

%% Sample from original, PCA, ICA and LDA subspace
%Here I just try to draw a sample from apple, you can change GMModel_PCA if
%you want to draw sample from other groups.
p = GMModel_ORI{1}.ComponentProportion;
r = find(mnrnd(1,p) == 1);
MU = GMModel_ORI{1}.mu(r, :);
SIGMA = diag(GMModel_ORI{1}.Sigma(:, :, r));
R = mvnrnd(MU,SIGMA);
R = R * Original' + mu_train;
subplot(1, 4, 1)
imshow(uint8(reshape(R, [256,256])))
title('Original', 'FontSize', 14)
p = GMModel_PCA{1}.ComponentProportion;
r = find(mnrnd(1,p) == 1);
MU = GMModel_PCA{1}.mu(r, :);
SIGMA = diag(GMModel_PCA{1}.Sigma(:, :, r));
R = mvnrnd(MU,SIGMA);
R = R * PC' + mu_train;
subplot(1, 4, 2)
imshow(uint8(reshape(R, [256,256])));
title('PCA', 'FontSize', 14)
p = GMModel_ICA{1}.ComponentProportion;
r = find(mnrnd(1,p) == 1);
MU = GMModel_ICA{1}.mu(r, :);
SIGMA = diag(GMModel_ICA{1}.Sigma(:, :, r));
R = mvnrnd(MU,SIGMA);
R = R * IC' * Original' + mu_train;
subplot(1, 4, 3)
imshow(uint8(reshape(R, [256,256])));
title('ICA', 'FontSize', 14)
p = GMModel_LDA{1}.ComponentProportion;
r = find(mnrnd(1,p) == 1);
MU = GMModel_LDA{1}.mu(r, :);
SIGMA = GMModel_LDA{1}.Sigma(:, :, r);
R = mvnrnd(MU,SIGMA);
R = R * LD' * Original' + mu_train;
subplot(1, 4, 4)
imshow(uint8(reshape(R, [256,256])));
title('LDA', 'FontSize', 14)