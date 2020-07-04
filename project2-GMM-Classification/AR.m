%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Project 3 for ECE 7868
%Mixture Gaussian Model
%Author: L.Z.
%Date: 10/24/2018
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc
clear all

%% Set the directory
mfile_name = mfilename('fullpath');    %only work when run the m.file instead of debuging.
[pathstr,name,ext] = fileparts(mfile_name);
cd(pathstr);
addpath(genpath('test2'))
addpath(genpath('FastICA_25'))

%% Read the data
num_face = 100;
img_path = 'test2';
    
%Male's images
img_male = zeros(50*26, 165*120);
for k = 1:50
    disp(['Reading images of M ', num2str(k)])
    if k>=10
        img_list = dir([img_path, '/M-0', num2str(k),'*.bmp']);
    else
        img_list = dir([img_path, '/M-00', num2str(k),'*.bmp']);
    end
    img_num = length(img_list);
    for j = 1:img_num
        image = imread([img_list(j).folder, '/',img_list(j).name]); 
        image = rgb2gray(image);
        img_male((k-1)*26+j, :) = image(:)';
    end
end

%Female's images
img_female = zeros(50*26, 165*120);
for k = 1:50
    disp(['Reading images of W ', num2str(k)])
    if k>=10
        img_list = dir([img_path, '/W-0', num2str(k),'*.bmp']);
    else
        img_list = dir([img_path, '/W-00', num2str(k),'*.bmp']);
    end
    img_num = length(img_list);
    for j = 1:img_num
        image = imread([img_list(j).folder, '/',img_list(j).name]);  
        image = rgb2gray(image);
        img_female((k-1)*26+j, :) = image(:)';
    end    
end

%% Creating the training set and the testing set
r = randperm(1300, 260);    %we use 20% of data to test
test_male = img_male(r, :);
train_male = img_male;
train_male(r, :) = [];

r = randperm(1300, 260);    %we use 20% of data to test
test_female = img_female(r, :);
train_female = img_female;
train_female(r, :) = [];

img_train = [train_male; train_female];

%% Gaussian Mixture Model in Original space
% Since the dimension of features is much larger than the number of
% observations, it is infeasible to use the Gaussian Mixture Model
% directly, so I used the PCA at first and retain as much information as
% possible. The number of PCs I choose is 100;

disp('GMModel in Original space')
num_ORI = 100; %# of PCs we use
[coeff, score, latent, tsquared, explained, mu_train] = pca(double(img_train));
Original = coeff(:,1:num_ORI);  %all PCs we use

ORI_male = (double(train_male) - mu_train) * Original;  %the projection onto subspace
GMModel_ORI_male = fitgmdist(ORI_male, 4, 'CovarianceType', 'diagonal');  %Gaussian Mixture Model, #of mixture is 4

ORI_female = (double(train_female) - mu_train) * Original;  %the projection onto subspace
GMModel_ORI_female = fitgmdist(ORI_female, 4, 'CovarianceType', 'diagonal');  %Gaussian Mixture Model, #of mixture is 4

%prediction
correct_ORI = 0;
total = 0;

test_ORI_male = (double(test_male) - mu_train) * Original;
for k = 1:260
    total = total + 1;
    p1 = pdf(GMModel_ORI_male, test_ORI_male(k, :));
    p2 = pdf(GMModel_ORI_female, test_ORI_male(k, :));
    %Compare the Likelihood
    if p1 > p2
        correct_ORI = correct_ORI + 1;
    end
end

test_ORI_female = (double(test_female) - mu_train) * Original;
for k = 1:260
    total = total + 1;
    p1 = pdf(GMModel_ORI_male, test_ORI_female(k, :));
    p2 = pdf(GMModel_ORI_female, test_ORI_female(k, :));
    %Compare the Likelihood
    if p1 < p2
        correct_ORI = correct_ORI + 1;
    end
end
correct_ORI / total

%% Gaussian Mixture Model in PCA subspace
disp('GMModel in PCA')
num_PCA = 10; %# of PCs we use
PC = coeff(:,1:num_PCA);  %all PCs we use

PCA_male = (double(train_male) - mu_train) * PC;  %the projection onto subspace
GMModel_PCA_male = fitgmdist(PCA_male, 4, 'CovarianceType', 'diagonal');  %Gaussian Mixture Model, #of mixture is 4

PCA_female = (double(train_female) - mu_train) * PC;  %the projection onto subspace
GMModel_PCA_female = fitgmdist(PCA_female, 4, 'CovarianceType', 'diagonal');  %Gaussian Mixture Model, #of mixture is 4

%prediction
correct_PCA = 0;
total = 0;

test_PCA_male = (double(test_male) - mu_train) * PC;
for k = 1:260
    total = total + 1;
    p1 = pdf(GMModel_PCA_male, test_PCA_male(k, :));
    p2 = pdf(GMModel_PCA_female, test_PCA_male(k, :));
    %Compare the Likelihood
    if p1 > p2
        correct_PCA = correct_PCA + 1;
    end
end

test_PCA_female = (double(test_female) - mu_train) * PC;
for k = 1:260
    total = total + 1;
    p1 = pdf(GMModel_PCA_male, test_PCA_female(k, :));
    p2 = pdf(GMModel_PCA_female, test_PCA_female(k, :));
    %Compare the Likelihood
    if p1 < p2
        correct_PCA = correct_PCA + 1;
    end
end
correct_PCA / total

%% Gaussian Mixture Model and EM in ICA subspace
% When the dimension of the is large, the running time of FastICA is slow,
% so I have to use the PCA to reduce the dimension at first, then use the
% ICA.

disp('GMModel in ICA')
num_ICA = 30;   %# of ICs we use
[IC A W] = fastica([ORI_male; ORI_female], 'verbose', 'off', 'numOfIC', num_ICA);
[IC, R] = mgs(IC');  %Gram-Smith Orthononalization

ICA_male = ORI_male * IC;
GMModel_ICA_male = fitgmdist(ICA_male, 4, 'CovarianceType', 'diagonal');  %Gaussian Mixture Model, #of mixture is 4

ICA_female = ORI_female * IC;
GMModel_ICA_female = fitgmdist(ICA_female, 4, 'CovarianceType', 'diagonal');  %Gaussian Mixture Model, #of mixture is 4

%prediction
correct_ICA = 0;
total = 0;

test_ICA_male = test_ORI_male * IC;
for k = 1:260
    total = total + 1;
    p1 = pdf(GMModel_ICA_male, test_ICA_male(k, :));
    p2 = pdf(GMModel_ICA_female, test_ICA_male(k, :));
    %Compare the Likelihood
    if p1 > p2
        correct_ICA = correct_ICA + 1;
    end
end

test_ICA_female = test_ORI_female * IC;
for k = 1:260
    total = total + 1;
    p1 = pdf(GMModel_ICA_male, test_ICA_female(k, :));
    p2 = pdf(GMModel_ICA_female, test_ICA_female(k, :));
    %Compare the Likelihood
    if p1 < p2
        correct_ICA = correct_ICA + 1;
    end
end
correct_ICA / total

%% Gaussian Miture Model in LDA subspace
num_LDA = 1;

% In order to used the LDA, I first projected the data
% to PCA space, which has dim = 100, and it can cantain the most information of
% the data. Then I used the LDA in the PCA space and finally and I
% projected it back to the original space.

LDA_male = ORI_male;
LDA_female = ORI_female;
test_LDA_male =  test_ORI_male;
test_LDA_female = test_ORI_female;
img_train_LDA = [LDA_male; LDA_female];

label = [ones(1300-260, 1); 2*ones(1300-260, 1)];
W = LDA(img_train_LDA, label);
LD = W(:, 1);
[LD, R] = mgs(LD);  %normalization

LDA_male = LDA_male * LD;
GMModel_LDA_male = fitgmdist(LDA_male, 4, 'CovarianceType', 'diagonal');  %Gaussian Mixture Model, #of mixture is 4

LDA_female = LDA_female * LD;
GMModel_LDA_female = fitgmdist(LDA_female, 4, 'CovarianceType', 'diagonal');  %Gaussian Mixture Model, #of mixture is 4

%prediction
correct_LDA = 0;
total = 0;

test_LDA_male = test_LDA_male * LD;
for k = 1:260
    total = total + 1;
    p1 = pdf(GMModel_LDA_male, test_LDA_male(k, :));
    p2 = pdf(GMModel_LDA_female, test_LDA_male(k, :));
    %Compare the Likelihood
    if p1 > p2
        correct_LDA = correct_LDA + 1;
    end
end

test_LDA_female = test_LDA_female * LD;
for k = 1:260
    total = total + 1;
    p1 = pdf(GMModel_LDA_male, test_LDA_female(k, :));
    p2 = pdf(GMModel_LDA_female, test_LDA_female(k, :));
    %Compare the Likelihood
    if p1 < p2
        correct_LDA = correct_LDA + 1;
    end
end
correct_LDA / total

%% Sample from original, PCA, ICA and LDA subspace
%Here I just try to draw a sample from male, you can change GMModel_PCA if
%you want to draw sample from other groups.
p = GMModel_ORI_male.ComponentProportion;
r = find(mnrnd(1,p) == 1);
MU = GMModel_ORI_male.mu(r, :);
SIGMA = diag(GMModel_ORI_male.Sigma(:, :, r));
R = mvnrnd(MU,SIGMA);
R = R * Original' + mu_train;
subplot(1, 4, 1)
imshow(uint8(reshape(R, [165,120])))
title('Original', 'FontSize', 14)

p = GMModel_PCA_male.ComponentProportion;
r = find(mnrnd(1,p) == 1);
MU = GMModel_PCA_male.mu(r, :);
SIGMA = diag(GMModel_PCA_male.Sigma(:, :, r));
R = mvnrnd(MU,SIGMA);
R = R * PC' + mu_train;
subplot(1, 4, 2)
imshow(uint8(reshape(R, [165,120])));
title('PCA', 'FontSize', 14)

p = GMModel_ICA_male.ComponentProportion;
r = find(mnrnd(1,p) == 1);
MU = GMModel_ICA_male.mu(r, :);
SIGMA = diag(GMModel_ICA_male.Sigma(:, :, r));
R = mvnrnd(MU,SIGMA);
R = R * IC' * Original' + mu_train;
subplot(1, 4, 3)
imshow(uint8(reshape(R, [165,120])));
title('ICA', 'FontSize', 14)

p = GMModel_LDA_male.ComponentProportion;
r = find(mnrnd(1,p) == 1);
MU = GMModel_LDA_male.mu(r, :);
SIGMA = GMModel_LDA_male.Sigma(:, :, r);
R = mvnrnd(MU,SIGMA);
R = R * LD' * Original' + mu_train;
subplot(1, 4, 4)
imshow(uint8(reshape(R, [165,120])));
title('LDA', 'FontSize', 14)