%% Used to creat condition of nn and test it performance in loop

clc; clear;
process = 0; %variable use to track the process
%% test of range and num_units
test_time = 100;
% breakpoint = [48 100 152];
% breakpoints = [32 68 100 132 168];
breakpoints = [20 40 60 80 100 120 140 160 180];
coordinates = 0;
root = 1;
ratio_train = 0.8;
test_range = (4:4:20); % 1*x vector
test_units = (3:1:20);
time_total = size(test_units, 2) * size(test_range, 2) * test_time;
test_error_range = zeros(test_time, size(test_range, 2)); % num_test * num_range
test_error_all = zeros(size(test_units,2), size(test_range, 2));
for unit = (1:size(test_units, 2))
    for range = (1:size(test_range, 2))
        for t = (1:test_time)
            [~, test_error_range(t, range)] = ...
                BE_predictor_v1_1(breakpoints, test_range(range), ratio_train, test_units(unit), coordinates, root);
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