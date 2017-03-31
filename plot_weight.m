function nn = plot_weight(breakpoints, nn)
% function nn = plot_weight(breakpoint, range, ratio_train, units)
% Use input [1 0 0,...], [0 1 0, ...][0 0 1, ...] to generate the weights

%% Training part
% error_re_ave = 1;
% while(error_re_ave >= 0.15)
%     [nn, error_re_ave] = BE_predictor_v1_1(breakpoint, range, ratio_train, units);
% end
% format bank
% fprintf('average relative error: %.2f%%\n', error_re_ave * 100);

%% Plot part
num_weight = size(breakpoints, 2);
segments = (1:num_weight);
X_input = eye(num_weight);
weights = zeros(1, num_weight);
for i = segments
    weights(i) = nn(X_input(:, i)); % each weight corresponds a segment of data
end
figure(2);
plot(segments, weights);
hold on;
