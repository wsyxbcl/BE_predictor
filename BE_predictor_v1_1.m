function [nn, error_re_ave, X_ps, Y_ps] = BE_predictor_v1_1(breakpoints, range, ratio_train, units, coordinates, root)
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

%% Data preparation
% clc
% clear

% Load data
% fprintf('Loading data...\n')
X_original = load('nianwu_data_input_values.csv');
Y_original = load('nianwu_data_output_values.csv');
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
% 3 segments
% X_1 = X_original(:, (breakpoints(1) + 1 - range/2):(breakpoints(1) + range/2));
% X_2 = X_original(:, (breakpoints(2) + 1 - range/2):(breakpoints(2) + range/2));
% X_3 = X_original(:, (breakpoints(3) + 1 - range/2):(breakpoints(3) + range/2));

% X = [mean(X_1, 2) mean(X_2, 2) mean(X_3, 2)];

% 5 segments
% X_1 = X_original(:, (breakpoints(1) + 1 - range/2):(breakpoints(1) + range/2));
% X_2 = X_original(:, (breakpoints(2) + 1 - range/2):(breakpoints(2) + range/2));
% X_3 = X_original(:, (breakpoints(3) + 1 - range/2):(breakpoints(3) + range/2));
% X_4 = X_original(:, (breakpoints(4) + 1 - range/2):(breakpoints(4) + range/2));
% X_5 = X_original(:, (breakpoints(5) + 1 - range/2):(breakpoints(5) + range/2));
% X = [mean(X_1, 2) mean(X_2, 2) mean(X_3, 2) mean(X_4, 2) mean(X_5, 2)]; % m0 * n0

size_train = ceil(ratio_train*m);
% Dividing data randomly
k = rand(1, m);
[~, idx] = sort(k);
X_train = X(idx(1:size_train), :)'; % n * m
Y_train = Y(idx(1:size_train))'; % 1 * m
X_test = X(idx(1+size_train:m), :)';
Y_test = Y(idx(1+size_train:m))';

% Feature scaling && Normolization
[X_norm, X_ps] = mapminmax(X_train);
[Y_norm, Y_ps] = mapminmax(Y_train);
% 
%% Training process using newff
% nn = newff(X_norm, Y_norm, 25);
% % nn = newff(X_norm, Y_norm, [1, 1], {'tansig', 'tansig'});
% nn.trainParam.epochs = 1000;
% nn.trainParam.lr = 0.01;
% nn.trainParam.goal = 0.000001;
% % nn.divideFcn = '';
% % nn.trainParam.max_fail=100; 
% nn = train(nn, X_norm, Y_norm);

%% Training process using fitnet
% fprintf('Training nn...\n')
% nn = fitnet([5, 1]);
nn = fitnet(units);
nn.divideParam.trainRatio = 0.9;
nn.divideParam.valRatio = 0.1;
nn.divideParam.testRatio = 0;
nn = train(nn, X_norm, Y_norm);
%% Testing part
% Feature scaling && Normolization of the test set
X_test_norm = mapminmax('apply', X_test, X_ps);
% Predicting
result = nn(X_test_norm);
% Reverse the feature scaling
Y_output = mapminmax('reverse', result, Y_ps);

% Showing the result
% figure(1)
% plot(Y_output, 'og');
% hold on

% plot(Y_test, '- *');
% Calculate the error
error_abs = abs(Y_output - Y_test);
error_re = (error_abs)./(Y_test);
error_re_ave = mean(abs(error_re)); % using abs here for convenience


% figure(2)
% plot(error, '- *')
% ylabel('error', 'fontsize', 12)
% xlabel('sample', 'fontsize', 12)
% fprintf('average error: %f\naverage test case: %f\n', error_ave, mean(Y_test))
% fprintf('relative error: %f\n', error_re_ave)
% Plot the distribution of the weight between the input and the first
% hidden layer
% figure(2)
% plot(nn.iw{1,1});
end