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
addpath(genpath('FastICA_25'))
addpath('cifar-10-batches-mat')

%% Read the data 
for k = 1:10
    train{k} = zeros(5000, 32*32*3);   %train img num in each category
    test{k} = zeros(1000, 32*32*3);    %test img num in each category
end
train_num = zeros(10, 1);
test_num = zeros(10, 1);

for i = 1:5
    disp(['Loading dataset ', num2str(i)])
    file_name = ['data_batch_', num2str(i), '.mat'];
    load(file_name);
    labels = labels + 1;
    for j = 1:length(labels)
        k = labels(j);
        train_num(k) = train_num(k) + 1;
        train{k}(train_num(k), :) = data(j,:);
    end
end

img_train = [];
for i = 1:5
    img_train = [img_train; train{i}];
end

load('test_batch.mat')
labels = labels + 1;
for j = 1:length(labels)
    k = labels(j);
    test_num(k) = test_num(k) + 1;
    test{k}(test_num(k), :) = data(j,:);
end

%% Gaussian Mixture Model in Original space
% Since the dimension of features is much larger than the number of
% observations, it is infeasible to use the Gaussian Mixture Model
% directly, so I used the PCA at first and retain as much information as
% possible. The number of PCs I choose is 100;

disp('GMModel in Original space')
[coeff, score, latent, tsquared, explained, mu_train] = pca(double(img_train));

%%
num_ORI = 30; %# of PCs we use

Original = coeff(:, 1:num_ORI);  %all PCs we use
ORI = {};
GMModel_ORI = {};
ORI_train = [];
for k = 1:10
    ORI{k} = (double(train{k}) - mu_train) * Original;  %the projection onto subspace
    GMModel_ORI{k} = fitgmdist(ORI{k}, 4, 'CovarianceType', 'diagonal');  %Gaussian Mixture Model
    ORI_train = [ORI_train; ORI{k}];
end

%prediction
correct_ORI = 0;
total = 0;

test_ORI = {};
for k = 1:10
    test_ORI{k} = (double(test{k}) - mu_train) * Original;
    for  i = 1:1000
        total = total + 1;
        correct_ORI = correct_ORI + 1;
        sample = test_ORI{k}(i, :);
        p1 = pdf(GMModel_ORI{k}, sample);
        for j = 1:10
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
num_PCA = 15; %# of PCs we use
PC = coeff(:, 1:num_PCA);  %all PCs we use

for k = 1:10
    PCA = (double(train{k}) - mu_train) * PC;  %the projection onto subspace
    GMModel_PCA{k} = fitgmdist(PCA, 4, 'CovarianceType', 'diagonal');  %Gaussian Mixture Model
end

%prediction
correct_PCA = 0;
total = 0;

for k = 1:10
    test_PCA = (double(test{k}) - mu_train) * PC;
    for  i = 1:1000
        total = total + 1;
        correct_PCA = correct_PCA + 1;
        sample = test_PCA(i, :);
        p1 = pdf(GMModel_PCA{k}, sample);
        for j = 1:10
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
num_ICA = 3;   %# of ICs in each group we use
IC = [];
for i = 1:10
    [IC_temp A W] = fastica(ORI{i}, 'verbose', 'off', 'numOfIC', num_ICA);
    IC = [IC; IC_temp];
end
[IC, R] = mgs(IC');  %Gram-Smith Orthononalization

for k = 1:10
    ICA = ORI{k} * IC;  %the projection onto subspace
    GMModel_ICA{k} = fitgmdist(ICA, 4, 'CovarianceType', 'diagonal');  %Gaussian Mixture Model
end

%prediction
correct_ICA = 0;
total = 0;

for k = 1:10
    test_ICA = test_ORI{k} * IC;
    for  i = 1:1000
        total = total + 1;
        correct_ICA = correct_ICA + 1;
        sample = test_ICA(i, :);
        p1 = pdf(GMModel_ICA{k}, sample);
        for j = 1:10
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
num_LDA = 9;

% In order to used the LDA, I first projected the data
% to PCA space, which has dim = 100, and it can cantain the most information of
% the data. Then I used the LDA in the PCA space and finally and I
% projected it back to the original space.
img_train_LDA =[];
label = [];
for k = 1:10
    label = [label; k * ones(5000, 1)];
end

W = LDA(ORI_train, label);
LD = W(:, 1:num_LDA);
[LD, R] = mgs(LD);  %normalization

for k = 1:10
    train_LDA = ORI{k} * LD;  %the projection onto subspace
    GMModel_LDA{k} = fitgmdist(train_LDA, 4, 'CovarianceType', 'diagonal');  %Gaussian Mixture Model
end

%prediction
correct_LDA = 0;
total = 0;

for k = 1:10
    test_LDA = test_ORI{k} * LD;
    for  i = 1:1000
        total = total + 1;
        correct_LDA = correct_LDA + 1;
        sample = test_LDA(i, :);
        p1 = pdf(GMModel_LDA{k}, sample);
        for j = 1:10
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
imshow(uint8(reshape(R, [32,32,3])))
title('Original', 'FontSize', 14)

p = GMModel_PCA{1}.ComponentProportion;
r = find(mnrnd(1,p) == 1);
MU = GMModel_PCA{1}.mu(r, :);
SIGMA = diag(GMModel_PCA{1}.Sigma(:, :, r));
R = mvnrnd(MU,SIGMA);
R = R * PC' + mu_train;
subplot(1, 4, 2)
imshow(uint8(reshape(R, [32,32,3])));
title('PCA', 'FontSize', 14)

p = GMModel_ICA{1}.ComponentProportion;
r = find(mnrnd(1,p) == 1);
MU = GMModel_ICA{1}.mu(r, :);
SIGMA = diag(GMModel_ICA{1}.Sigma(:, :, r));
R = mvnrnd(MU,SIGMA);
R = R * IC' * Original' + mu_train;
subplot(1, 4, 3)
imshow(uint8(reshape(R, [32,32,3])));
title('ICA', 'FontSize', 14)

p = GMModel_LDA{1}.ComponentProportion;
r = find(mnrnd(1,p) == 1);
MU = GMModel_LDA{1}.mu(r, :);
SIGMA = GMModel_LDA{1}.Sigma(:, :, r);
R = mvnrnd(MU,SIGMA);
R = R * LD' * Original' + mu_train;
subplot(1, 4, 4)
imshow(uint8(reshape(R, [32,32,3])));
title('LDA', 'FontSize', 14)