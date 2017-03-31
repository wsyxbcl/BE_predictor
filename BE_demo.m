%% Network demo, for visualisation the result
clear; clc;
%% Coefficients
% breakpoints = [32 68 100 132 168];
% breakpoints = [14 32 84 100 116 132 148 168];
coordinates = 0;
root = 0;
ratio_train = 0.9;
range = 14;
units = 3;

[nn, error, X_ps, Y_ps] = ...
    BE_predictor_v1_1(breakpoints, range, ratio_train, units, coordinates, root);
fprintf('Error: %f\n', error);
%% Result demo
X_original = load('nianwu_data_input_values.csv');
Y_original = load('nianwu_data_output_values.csv');
Y = Y_original';
m = size(X_original, 1);
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
X_norm = mapminmax('apply', X, X_ps);
result = nn(X_norm);
Y_output = mapminmax('reverse', result, Y_ps);

%% Visualization
figure(1)
plot(Y_output, 'og');
hold on
plot(Y, '- *');

%% Plot weight
plot_weight(breakpoints, nn);