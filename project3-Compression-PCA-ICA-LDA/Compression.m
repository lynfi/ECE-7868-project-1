%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Project 2 for ECE 7868
%Compare PCA, ICA and LDA
%Author: L.Z.
%Date: 09/30/2018
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc
clear all

%% Set the directory
mfile_name = mfilename('fullpath');    %only work when run the m.file instead of debuging.
[pathstr,name,ext] = fileparts(mfile_name);
cd(pathstr);
addpath(genpath('test2'))
addpath(genpath('FastICA_25'))

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

%% PCA recover
img_num = zeros(num_face, 1);  %img num in each category
num = 10; %# of PCs we used
for k = 1:num_face
    img_num(k) = size(img_matrix{k}, 1);
end

image = img_matrix{k};
image = image(1, :);
projection = zeros(num_face, max(img_num), length(image(:)'));
projection_ICA = projection;
for k = 1:num_face
    co = coeff{k};  %all PCs in category k
    subspace_vec = co(:,1:num);
    img_k = img_matrix{k};
    for i = 1:img_num(k)
        sample = double(img_k(i, :));
        %calculate the projection
        projection(k, i, :) = (sample - mu{k}) * subspace_vec * subspace_vec' + mu{k};
    end
end
figure(1)



%% ICA recover
% Just choose the first 4 faces since it is slow.
num=20;
for k = 1:4
    disp(['Calculating IC for ', num2str(k)]);
    [icasig A W] = fastica(double(img_matrix{k}) - mu{k}, 'verbose', 'off', 'numOfIC', num);
    %Gram-Smith Orthononalization
    [Q, R] = mgs(icasig');
    img_k = double(img_matrix{k}) - mu{k};
    projection_ICA(k, :, :) = img_k * Q * Q' + mu{k};
end


%% LDA recover
% Just choose the first 4 faces and used 4 components.
% In order to used the LDA, I first projected the data
% to PCA space, which has dim = 40, and it can cantain the most information of
% the data. Then I used the LDA in the PCA space and finally and I
% projected it back to the original space.

num = 3;
X = [img_matrix{1};img_matrix{2};img_matrix{3};img_matrix{4}];
X = double(X);
[coeff, score, latent, tsquared, explained, mu] = pca(X);
PCAsubspace = coeff(:, 1:40);
centerX = X - mu;
data = (X - mu) * PCAsubspace;

label = [ones(26,1);2*ones(26,1);3*ones(26,1);4*ones(26,1)];
W = LDA(data, label);
LDAsubspace = W(:, 1:3);
%Gram-Smith Orthononalization
[Q, R] = mgs(LDAsubspace);
projection_LDA = (data * Q * Q') * PCAsubspace' + mu;

%% Plot the results
for k = 1:1
    kk = (k-1)*8;
    img = img_matrix{k};
    subplot(4, 8, kk+1), imshow(reshape(uint8(img(1, :)), [165,120]))
    title('Original', 'FontSize', 14)
    subplot(4, 8, kk+2), imshow(reshape(uint8(projection(k, 1, :)),[165,120]))
    title('PCA', 'FontSize', 14)
    subplot(4, 8, kk+3), imshow(reshape(uint8(projection_ICA(k, 1, :)),[165,120]))
    title('ICA', 'FontSize', 14)
    subplot(4, 8, kk+4), imshow(reshape(uint8(projection_LDA((k-1)*26+1, :)),[165,120]))
    title('LDA', 'FontSize', 14)
    subplot(4, 8, kk+5), imshow(reshape(uint8(img(21+k, :)), [165,120]))
    title('Original', 'FontSize', 14)
    subplot(4, 8, kk+6), imshow(reshape(uint8(projection(k, 21+k, :)),[165,120]))
    title('PCA', 'FontSize', 14)
    subplot(4, 8, kk+7), imshow(reshape(uint8(projection_ICA(k, 21+k, :)),[165,120]))
    title('ICA', 'FontSize', 14)
    subplot(4, 8, kk+8), imshow(reshape(uint8(projection_LDA((k-1)*26+21+k, :)),[165,120]))
    title('LDA', 'FontSize', 14)
end
for k = 2:4
    kk = (k-1)*8;
    img = img_matrix{k};
    subplot(4, 8, kk+1), imshow(reshape(uint8(img(1, :)), [165,120]))
    subplot(4, 8, kk+2), imshow(reshape(uint8(projection(k, 1, :)),[165,120]))
    subplot(4, 8, kk+3), imshow(reshape(uint8(projection_ICA(k, 1, :)),[165,120]))
    subplot(4, 8, kk+4), imshow(reshape(uint8(projection_LDA((k-1)*26+1, :)),[165,120]))
    subplot(4, 8, kk+5), imshow(reshape(uint8(img(21+k, :)), [165,120]))
    subplot(4, 8, kk+6), imshow(reshape(uint8(projection(k, 21+k, :)),[165,120]))
    subplot(4, 8, kk+7), imshow(reshape(uint8(projection_ICA(k, 21+k, :)),[165,120]))
    subplot(4, 8, kk+8), imshow(reshape(uint8(projection_LDA((k-1)*26+21+k, :)),[165,120]))
end