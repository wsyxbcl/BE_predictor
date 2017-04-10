function [nn, error_re_ave] = BE_predictor_v1_1(X_original, Y_original, breakpoints, range, ratio_train, units, coordinates, root)
% Using tool kit in MATLAB to build BP NN for the Bonding energy
% error_ave = BE_predictor(breakpoints, range, ratio_train, units)
% by Yunxuan Chai(yx_chai@whu.edu.cn), 2017.3.12

% BE_predictor_v1.0
% Add divide func: Change the way that data is divided, from continuous to
% discrete.
% Change the calculation of ave_error to make it more reasonable.
% by yx_chai, 2017.3.18

% BE_predictor_v1.1
% Change the way that data is divided, from 3 segments to 5, and more.
% Add two boolean input 'coordinates' and 'root', where coordinates refer
% to the energy coordinate of input, and root refers to whether apply
% sqrt() to the input data.
% by yx_chai, 2017.3.29

% BE_predictor_v1.1.1
% Change the parameter of fitnet by network's property based in MATLAB.
% by yx_chai, 2017.3.30

% BE_predictor_v1.1.2
% Add multielements feature, change the source of data to satisfy the need
% of multielements training.
% Add LOO feature to the function, enable Leave-One-Out cross validation.
% And the ratio_train feature is kept to enable the test set, test set will
% be replaced by loo_sample
% by yx_chai, 2017.4.1

% BE_predictor_v1.1.2.1
% Bug fixed: in error_re_ave, put the abs to the Y_output step
% by yx_chai, 2017.4.1

% BE_predictor_v1.1.3
% Refined version of v1.1.2, remove some chaos, and remove the feature
% scaling part for our features are already in same dimension


% Load data
% fprintf('Loading data...\n')
% X_original = load('nianwu_data_input_values.csv');
% Y_original = load('nianwu_data_output_values.csv');
Y = Y_original;
m = size(X_original, 1);
if root
    X_original = sqrt(X_original); %test for sqrt input
end

if coordinates
    X_coordinates = load('energy_coordinate.csv');
    X_original = X_original.* repmat(X_coordinates, m, 1);
end
    
% Select a range of data as the input layer
% range = 20;
% X = X_original(:, (half + 1 - range/2):(half + range/2));

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
% Dividing data randomly
k = rand(1, m);
[~, idx] = sort(k);
X_test = X(idx(1+m_train:m), :)';
Y_test = Y(idx(1+m_train:m))';
X_train = X(idx(1:m_train), :)'; % n * m
Y_train = Y(idx(1:m_train))'; % 1 * m
% Feature scaling && Normolization
% [X_norm, X_ps] = mapminmax(X_train);
% [Y_norm, Y_ps] = mapminmax(Y_train);
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
%% Testing part
% Feature scaling && Normolization of the test set
% X_test_norm = mapminmax('apply', X_test, X_ps);
X_test_norm = X_test;
% Predicting
result = nn(X_test_norm);
% Reverse the feature scaling
% Y_output = mapminmax('reverse', result, Y_ps);
Y_output = result;


% Calculate the error
error_abs = abs(Y_output - Y_test);
error_re = (error_abs)./abs(Y_test);
error_re_ave = mean(error_re); % using abs here for convenience
end