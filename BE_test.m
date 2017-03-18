%% Used to creat condition of nn and test it performance in loop

clc; clear;
%% test of range and num_units
test_time = 100;
breakpoint = [48 100 152];
test_range = (4:2:52); % 1*9 vector
test_units = (2:1:6);
test_error_range = zeros(test_time, size(test_range, 2)); % num_test * num_range
test_error_all = zeros(size(test_units,2), size(test_range, 2));
for unit = (1:size(test_units, 2))
    for range = (1:size(test_range, 2))
        for t = (1:test_time)
            [~, test_error_range(t, range)] = BE_predictor_v1_0(breakpoint, test_range(range), 0.8, test_units(unit));
            fprintf('Test: num_units = %d range = %d time = %d\n', test_units(unit), test_range(range) ,t);
        end
    end
    test_error_all(unit, :) = mean(test_error_range, 1);
end
% figure(1)
% plot(test_range, mean(test_error_range, 1));