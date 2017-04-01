%% Used to creat condition of nn and test it performance in loop
clc; clear;
process = 0; %variable use to track the process
% elements = {'H', 'Li', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg',...
%    'Al','Si', 'P', 'S', 'Cl'}
datasets = load('datasets\data_H.csv');
n = size(datasets, 2);
X_original = datasets(:, 1:n-1);
Y_original = datasets(:, n);
%% test of range and num_units
test_time = 30;
% breakpoint = [48 100 152];
breakpoints = [32 68 100 132 168];
% breakpoints = [20 40 60 80 100 120 140 160 180];
coordinates = 0;
root = 0;
ratio_train = 1;
test_range = (20:4:32); % 1*x vector
test_units = (2:1:10);
time_total = size(test_units, 2) * size(test_range, 2) * test_time;
test_error_range = zeros(test_time, size(test_range, 2)); % num_test * num_range
test_error_all = zeros(size(test_units,2), size(test_range, 2));
for unit = (1:size(test_units, 2))
    for range = (1:size(test_range, 2))
        for t = (1:test_time)
            [~, test_error_range(t, range)] = ...
                BE_predictor_v1_2(X_original, Y_original, breakpoints, test_range(range), ratio_train, test_units(unit), coordinates, root);
            % LOO is used here, so the test_error refers to the validation
            % error actually.
            fprintf('Test: num_units = %d range = %d time = %d process = %.2f\n',...
                test_units(unit), test_range(range) ,t, process/time_total);
            process = process + 1;
        end
    end
    test_error_all(unit, :) = mean(test_error_range, 1);
end
% figure(1)
% plot(test_range, mean(test_error_range, 1));
% mesh(test_range, test_units, test_error_all)