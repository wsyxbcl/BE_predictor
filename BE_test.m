%% Used to creat condition of nn and test it performance in loop
clc; clear;
process = 0; %variable use to track the process
element_set = {'H', 'Li', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg',...
    'Al','Si', 'P', 'S', 'Cl'};
for element = element_set;
    filename = strcat('datasets\data_',element,'.csv');
    filename = filename{1};
    datasets = load(filename);
    n = size(datasets, 2);
    X_original = datasets(:, 1:n-1);
    Y_original = datasets(:, n);
    %% test of range and num_units
    test_time = 10;
    % breakpoint = [48 100 152];
    breakpoints = [32 68 100 132 168];
    % breakpoints = [20 40 60 80 100 120 140 160 180];
    coordinates = 0;
    root = 0;
    LOO = 1; % Leave-one-out validation
    ratio_train = 1;
    test_range = (16:4:32); % 1*x vector
    test_units = (2:1:8);
    time_total = size(test_units, 2) * size(test_range, 2) * test_time * size(element_set, 2);
    test_error_range = zeros(test_time, size(test_range, 2)); % num_test * num_range
    test_error_all = zeros(size(test_units,2), size(test_range, 2));
    for unit = (1:size(test_units, 2))
        for range = (1:size(test_range, 2))
            for t = (1:test_time)
                if LOO
                    [~, test_error_range(t, range)] = ...
                        BE_predictor_v1_2(X_original, Y_original, breakpoints, test_range(range), ratio_train, test_units(unit), coordinates, root);
                else
                    [~, test_error_range(t, range)] = ...
                        BE_predictor_v1_1(X_original, Y_original, breakpoints, test_range(range), ratio_train, test_units(unit), coordinates, root);
                end           
                % if LOO is used here, the test_error refers to the validation
                % error actually.
                fprintf('%s num_units = %d range = %d time = %d process = %.2f\n',...
                    filename, test_units(unit), test_range(range) ,t, process/(time_total));
                process = process + 1;
            end
        end
        test_error_all(unit, :) = mean(test_error_range, 1);
    end
    eval(strcat('error_', element{1}, '=test_error_all'))
end
% figure(1)
% plot(test_range, mean(test_error_range, 1));
% mesh(test_range, test_units, test_error_all)

%% mesh errors
j = 1;
for i = element_set
    % mesh error grid
    figure(j)
    eval(strcat('mesh(test_range, test_units, error_',i{1},')'));
    title(i{1});
    % output min of errors and its condition
    eval(strcat('[error_min, I] = min(error_', i{1}, '(:));'));
    eval(strcat('[I_row, I_col] = ind2sub(size(error_', i{1}, '), I);'));
    range_min = test_range(I_col);
    units_min = test_units(I_row);
    fprintf('Min of %s = %.2f when range = %.2f and units = %d.\n', i{1}, ...
        error_min, range_min, units_min);
    j = j + 1;
end