function [U, S] = pca(X)
%PCA Run principal component analysis on the dataset X
%   [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
%

[m, n] = size(X);

U = zeros(n);
S = zeros(n);

Sigma = (X' * X)/m;
[U, S, V] = svd(Sigma);

end