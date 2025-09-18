%% Analytical Solution for Relaxation Time
% Analytical_solution_Relaxation_time2.m: Calculate relaxation time using time-based formulas
% Enhanced with various visualization improvements

clear all;
clc;

%% Load Data
% Load dataset - update path as needed for your environment
data = readmatrix(['H:\Batch_OscillatoryShear_10_Heaviside\' ...
    'C2-SINDy_Coefficients.csv']);

%% Filter Valid Data
% Keep only rows where column 17 values are non-negative
valid_rows = data(:, 17) >= 0;
data = data(valid_rows, :);

%% Extract Parameters and Calculate Dimensionless Numbers
C1 = data(:, 2);    % Coefficient C1
C2 = data(:, 3);    % Coefficient C2
C3 = data(:, 4);    % Coefficient C3
p = data(:, 5);     % Parameter p
ds = data(:, 6);    % Parameter ds
rho = data(:, 7);   % Parameter rho
nH = data(:, 8);    % Parameter nH
T = data(:, 9);     % Parameter T
E = data(:, 11);    % Parameter E
delta = data(:, 17);% Parameter delta

% Calculate dimensionless numbers
Pi_1 = E ./ p;
Pi_2 = nH;  % Originally nH.*ds in commented code
pi3 = log10(Pi_1 .* (Pi_2) .^ (1));

%% Configuration Parameters
mu_0 = 0.3664;          % Reference viscosity
gamma_dot0 = 0.2;       % Reference shear rate

% Target value for relaxation time calculation: -(1 - 2e^(-1)) * mu_0
target_value = -(1 - 2*exp(-1)) * mu_0;

% Time range setup: gamma_dot0*t covers [-1.5, 2]
% Corresponding time range: -1.5/0.2 = -7.5 to 2/0.2 = 10
t = linspace(-3, 8, 30000);  % Maintain sufficient points for accuracy

%% Initialize Storage Arrays
num_cases = length(C1);
relaxation_time_t = zeros(num_cases, 1);    % Relaxation time from numerical method
relaxation_time_t2 = zeros(num_cases, 1);   % Relaxation time from analytical formula

%% Configure Plot
figure;
set(gcf, 'Position', [100, 100, 1800, 800]); 
hold on;
box on;
grid on;

% Axis labels and title (formatted for academic presentation)
xlabel('$\dot{\gamma}_{0}t$', 'FontName', 'Arial', 'FontSize', 20, 'Interpreter', 'latex');
ylabel('\fontname{Arial} \it\tau\rm /\it\sigma', ...
    'FontSize', 20, ...
    'Interpreter', 'tex');
title('Relaxation Time Variation with Normalized Time', 'FontName', 'Arial', 'FontSize', 22);

%% Configure Highlighted Cases
% Store data for special cases to plot them on top
special_tau_hat = cell(3, 1);
special_indices = [13, 180, 459];  % Indices of cases to highlight
special_colors = {[0.6392, 0.0196, 0.2627],  % Color for case 13
                  [0.2275, 0.6, 0.4078],      % Color for case 180
                  [170, 126, 160]/255};       % Color for case 459

%% First Pass: Plot All Cases in Gray (with transparency)
for i = 1:num_cases
    % Extract current coefficients
    c1 = C1(i);
    c2 = C2(i);
    c3 = C3(i);
    
    % Solve quadratic equation C3*A^2 - C2*A + C1 = 0 for steady-state values
    p2 = [c3, -c2, c1];
    r = roots(p2);

    % Filter real roots
    real_roots = r(abs(imag(r)) < 1e-6);
    
    if isempty(real_roots)
        continue;  % Skip cases with no real roots
    end
    
    %% Calculate relaxation time using analytical formula
    if c2 == 0
        % Avoid division by zero
        relaxation_time_t2(i) = NaN;
        warning(sprintf('Case %d has C2=0, cannot calculate relaxation time', i));
        continue;
    end
    
    ratio = (c3 * mu_0) / c2;               % C3*mu0/C2 ratio
    log_arg = 1 - 2*(1 - exp(-1)) * ratio;  % Argument for logarithm
    
    % Check if logarithm argument is valid (>0)
    if log_arg <= 0
        relaxation_time_t2(i) = NaN;
        warning(sprintf('Case %d has invalid logarithm argument (%.4f), cannot calculate', i, log_arg));
        continue;
    end
    
    % Calculate denominator: gamma_dot0 * |C2 - 2*C3*mu0|
    denominator = gamma_dot0 * abs(c2 - 2*c3*mu_0);
    if denominator == 0
        relaxation_time_t2(i) = NaN;
        warning(sprintf('Case %d has zero denominator, cannot calculate', i));
        continue;
    end
    
    % Calculate relaxation time using analytical formula
    relaxation_time_t2(i) = (1 + log(log_arg)) / denominator * gamma_dot0;
    
    %% Calculate tau_hat and find relaxation time numerically
    tau_hat = zeros(size(t));
    % For t <= 0 (corresponding to x-axis [-1.5, 0]): tau_hat = -mu_0
    tau_hat(t <= 0) = -mu_0;
    
    % For t > 0: calculate using original formula
    t_pos = t(t > 0);
    term1_pos = 1 - 2*(1 - (c3*mu_0)/c2) * exp(gamma_dot0 * (c2 - 2*c3*mu_0) * t_pos);
    term2_pos = 1 - 2*(c3*mu_0/c2) * exp(gamma_dot0 * (c2 - 2*c3*mu_0) * t_pos);
    tau_hat(t > 0) = mu_0 * term1_pos ./ term2_pos;
    
    % Find time when tau reaches target value (only in t > 0 region)
    idx = find(t > 0 & tau_hat >= target_value, 1, 'first');
    
    if ~isempty(idx)
        if tau_hat(idx) == target_value
            relaxation_time_t(i) = t(idx) * gamma_dot0;
        else
            if idx > 1
                t1 = t(idx-1); t2 = t(idx);
                y1 = tau_hat(idx-1); y2 = tau_hat(idx);
                % Linear interpolation to find exact crossing time
                relaxation_time_t(i) = (t1 + (target_value - y1)*(t2 - t1)/(y2 - y1)) * gamma_dot0;
            else
                relaxation_time_t(i) = t(idx) * gamma_dot0;
            end
        end
    else
        relaxation_time_t(i) = NaN;
        warning(sprintf('Case %d does not reach target value within time range', i));
    end

    % Skip special cases (will be plotted separately)
    if ismember(i, special_indices)
        continue;
    end
    
    % Plot gray lines with transparency (background cases)
    lineColor = [0.7176, 0.7098, 0.7137, 0.3];  % RGBA with transparency
    plot(t*gamma_dot0, -tau_hat, 'Color', lineColor, 'LineWidth', 2);
end

%% Second Pass: Plot Highlighted Cases (on top)
for i_idx = 1:length(special_indices)
    i = special_indices(i_idx);
    
    % Extract coefficients for highlighted case
    c1 = C1(i);
    c2 = C2(i);
    c3 = C3(i);
    
    % Solve quadratic equation for steady-state values
    p2 = [c3, -c2, c1];
    r = roots(p2);
    real_roots = r(abs(imag(r)) < 1e-6);
    
    if isempty(real_roots)
        continue;
    end
    
    % Calculate tau_hat for highlighted case
    tau_hat = zeros(size(t));
    tau_hat(t <= 0) = -mu_0;
    t_pos = t(t > 0);
    term1_pos = 1 - 2*(1 - (c3*mu_0)/c2) * exp(gamma_dot0 * (c2 - 2*c3*mu_0) * t_pos);
    term2_pos = 1 - 2*(c3*mu_0/c2) * exp(gamma_dot0 * (c2 - 2*c3*mu_0) * t_pos);
    tau_hat(t > 0) = mu_0 * term1_pos ./ term2_pos;
    
    % Store data for highlighted case
    special_tau_hat{i_idx} = tau_hat;
    
    % Plot highlighted case with distinct color
    lineColor = special_colors{i_idx};
    plot(t*gamma_dot0, -tau_hat, 'Color', lineColor, 'LineWidth', 5, ...
        'DisplayName', sprintf('Case %d', i));
end

%% Finalize Plot
% Plot target value line (ensured to be on top)
plot(t*gamma_dot0, ones(size(t))*target_value, 'k--', 'LineWidth', 2, ...
    'DisplayName', 'Target Value: (1-2e^{-1})μ₀');

% Add legend and format axes
legend('Location', 'Best', 'FontName', 'Arial', 'FontSize', 16);
xlim([-0.2, 1.6]);
hold off;
grid off;
set(gca, 'LineWidth', 1.6, 'FontSize', 20, 'FontName', 'Arial');

%% Display Results
fprintf('Relaxation times (time to reach target value) for all cases:\n');
for i = 1:num_cases
    fprintf('Case %d: %.4f\n', i, relaxation_time_t(i));
end