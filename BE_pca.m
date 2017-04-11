%% Use PCA algorithm to reduce the dimension of our input and see
clc; clear; close all;

filename = 'datasets\data_O.csv';
datasets = load(filename);
n = size(datasets, 2);
X_original = datasets(:, 1:n-1);
% we do not do normalization here
X_norm = X_original;
[U, S] = pca(X_norm);
fprintf('Dimension reduction on dataset.\n');
K = 2;    % aimed dimension
Z = projectData(X_norm, U, K);
% X_rec = recoverData(Z, U, K);
scatter(Z(:, 1), Z(:, 2));