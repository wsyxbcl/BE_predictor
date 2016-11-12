%% Using tool kit in MATLAB to build BP NN for the Bonding energy
%  by Yunxuan Chai(yx_chai@whu.edu.cn), 2016.11.4


%% Data preparation
clc
clear

% Load data
fprintf('Loading data...\n')
X = load('nianwu_data_input_values.csv');
Y = load('nianwu_data_output_values.csv');
m = size(X, 1);
size_train = ceil(0.7*m);
% Dividing data randomly
k = rand(1, m);
[item, idx] = sort(k);
X_train = X(idx(1:size_train), :)';
Y_train = Y(idx(1:size_train))';
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
nn = fitnet([1, 1]);
nn = train(nn, X_norm, Y_norm);
%% Testing part
% Feature scaling && Normolization of the test set
X_test_norm = mapminmax('apply', X_test, X_ps);
% Predicting
result = nn(X_test_norm);
% Reverse the feature scaling
Y_output = mapminmax('reverse', result, Y_ps);

% Showing the result
figure(1)
plot(Y_output, 'og');
hold on

plot(Y_test, '- *');
% 
% figure(2)
% plot(nn.iw{1,1});