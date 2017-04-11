%% Network demo, for visualisation the result
clear;
% clc;
close(figure(1))
%% Data preparation
% X_original = load('nianwu_data_input_values.csv');
% Y_original = load('nianwu_data_output_values.csv');
% m = size(X_original, 1);
% 
% elements = {'H', 'Li', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg',...
%    'Al','Si', 'P', 'S', 'Cl'}
datasets = load('datasets\data_O.csv');
m = size(datasets, 1);
n = size(datasets, 2);
X_original = datasets(:, 1:n-1);
Y_original = datasets(:, n);
%% Coefficients
% breakpoints = [32 68 100 132 168];
% breakpoints = [20 40 60 80 100 120 140 160 180];
breakpoints = (10:10:190);
coordinates = 0;
root = 0;
ratio_train = 1;
range = 10;
units = 3;

[nn, error] = ...
    BE_predictor_v1_1(X_original, Y_original, breakpoints, range, ratio_train, units, coordinates, root);
% fprintf('Error: %f\n', error);
%% Result demo

Y = Y_original';
if root
    X_original = sqrt(X_original); %test for sqrt input
end

if coordinates
    X_coordinates = load('energy_coordinate.csv');
    X_original = X_original.* repmat(X_coordinates, m, 1);
end

num_segments = size(breakpoints, 2);
X_seg_ave = zeros(m, num_segments);
for i = (1:num_segments)
    X_seg_ave(:, i) = mean(...
        X_original(:, (breakpoints(i) + 1 - range/2):(breakpoints(i) + range/2)), 2);
end
X = X_seg_ave';
% X_norm = mapminmax('apply', X, X_ps);
X_norm = X;
result = nn(X_norm);
% Y_output = mapminmax('reverse', result, Y_ps);
Y_output = result;

%% Visualization
figure(1)
plot(Y_output, 'og');
hold on
plot(Y, '- *');

%% Plot weight
plot_weight(breakpoints, nn, 0);