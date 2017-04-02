function [nn, loo_error] = BE_predictor_v1_2(X_original, Y_original, breakpoints, range, ratio_train, units, coordinates, root)
% Using tool kit in MATLAB to build BP NN for the Bonding energy
% error_ave = BE_predictor(breakpoints, range, ratio_train, units)
% by Yunxuan Chai(yx_chai@whu.edu.cn), 2017.3.12

%% Data preparation
% clc
% clear

Y = Y_original;
m = size(X_original, 1);
if root
    X_original = sqrt(X_original); %test for sqrt input
end

if coordinates
    X_coordinates = load('energy_coordinate.csv');
    X_original = X_original.* repmat(X_coordinates, m, 1);
end

% Data segregation
num_segments = size(breakpoints, 2);
X_seg_ave = zeros(m, num_segments);
for i = (1:num_segments)
    X_seg_ave(:, i) = mean(...
        X_original(:, (breakpoints(i) + 1 - range/2):(breakpoints(i) + range/2)), 2);
        % When range = 2 -> 100:101
end
X = X_seg_ave;
m_train = ceil(ratio_train*m);
k = rand(1, m); % Dividing data randomly
[~, idx] = sort(k);
X_test = X(idx(1+m_train:m), :)';
Y_test = Y(idx(1+m_train:m))';

loo_error = 0;
for loo_time = (1:m_train)
    X_train = X(idx(1:m_train), :)'; % n * m
    Y_train = Y(idx(1:m_train))'; % 1 * m

    X_loo = X_train(:, loo_time);
    Y_loo = Y_train(:, loo_time);
    X_train(:, loo_time) = [];
    Y_train(:, loo_time) = [];
    % Feature scaling && Normolization
%     [X_norm, X_ps] = mapminmax(X_train);
%     [Y_norm, Y_ps] = mapminmax(Y_train);
    X_norm = X_train;
    Y_norm = Y_train;
    %% Training process using fitnet
    % fprintf('Training nn...\n')
    % nn = fitnet([5, 1]);
    nn = fitnet(units);
    nn.divideParam.trainRatio = 0.7;
    nn.divideParam.valRatio = 0.3;
    nn.divideParam.testRatio = 0;
    nn = train(nn, X_norm, Y_norm);
    
%     X_loo_norm = mapminmax('apply', X_loo, X_ps);
    X_loo_norm = X_loo;
    loo_result = nn(X_loo_norm);
%     Y_loo_output = mapminmax('reverse', loo_result, Y_ps);
    Y_loo_output = loo_result;
    error_abs = abs(Y_loo_output - Y_loo);
    loo_error = loo_error + error_abs/abs(Y_loo);
    % fprintf('count\n');
end
loo_error = loo_error/m_train; % bug fixed, replace mean(), yx_chai, 20170402
%% Test part
% Feature scaling && Normolization of the test set
% X_test_norm = mapminmax('apply', X_test, X_ps);
X_test_norm = X_test;
% Predicting
result = nn(X_test_norm);
% Reverse the feature scaling
% Y_output = mapminmax('reverse', result, Y_ps);
Y_output = result;
error_abs = abs(Y_output - Y_test);
error_re = (error_abs)./abs(Y_test);
error_re_ave = mean(error_re); % using abs here for convenience
% fprintf('Error of test set: %.2f\n', error_re_ave);
end