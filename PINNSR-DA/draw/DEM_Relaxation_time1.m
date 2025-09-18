%% Relaxation Time Calculation from DEM Heaviside Simulations
% This script calculates relaxation times using multiple methods from DEM simulations
% and compares results between direct simulation data and analytical solutions.

clear all;
clc;

%% Configuration and Path Setup
% Update these paths according to your file system
root_path = 'H:\Batch_OscillatoryShear_10_Heaviside\C2_cases\';

% Load coefficient data from CSV file
csv_table = readtable(['H:\Batch_OscillatoryShear_10_Heaviside\' ...
    'C2-SINDy_Coefficients.csv']);

%% Initialize Storage Arrays
first_level_dirs = dir(root_path);
first_level_dirs = first_level_dirs([first_level_dirs.isdir]);
first_level_dirs = first_level_dirs(~ismember({first_level_dirs.name}, {'.', '..'}));
num_cases = length(first_level_dirs);

% Parameter storage arrays
C1 = zeros(1, num_cases);
C2 = zeros(1, num_cases);
C3 = zeros(1, num_cases);
p = zeros(1, num_cases);
ds = zeros(1, num_cases);
rho = zeros(1, num_cases);
nH = zeros(1, num_cases);
T = zeros(1, num_cases);
E = zeros(1, num_cases);
delta = zeros(1, num_cases);

% Relaxation time storage (different calculation methods)
relaxation_time_dem = zeros(1, num_cases);       % From direct DEM truncation
relaxation_time_fit = zeros(1, num_cases);       % From exponential fitting
relaxation_time_analytical = zeros(1, num_cases); % From analytical solution

%% Process Each Simulation Case
for i = 1:num_cases
    % Get current directory and extract case name
    current_dir = fullfile(root_path, first_level_dirs(i).name);
    folder_parts = strsplit(first_level_dirs(i).name, '_I');
    folder_name = folder_parts{1};

    % Find matching entry in CSV table
    match_index = find(strcmp(csv_table{:,1}, folder_name), 1);
    
    if ~isempty(match_index)
        % Extract parameters from CSV
        C1(1,i) = csv_table{match_index, 2};
        C2(1,i) = csv_table{match_index, 3};
        C3(1,i) = csv_table{match_index, 4};
        p(1,i) = csv_table{match_index, 5};
        ds(1,i) = csv_table{match_index, 6};
        rho(1,i) = csv_table{match_index, 7};
        nH(1,i) = csv_table{match_index, 8};
        T(1,i) = csv_table{match_index, 9};
        E(1,i) = csv_table{match_index, 11};
        delta(1,i) = csv_table{match_index, 17};
    else
        warning(sprintf('No match found in CSV for folder "%s"', folder_name));
    end

    % Load DEM simulation data
    load(fullfile(current_dir, 'post', 'Tau_Gammat.mat'));
    
    %% Method 1: Relaxation time from direct DEM data truncation
    mu0 = 0.3664;
    target_value = (1 - 2*exp(-1)) * mu0;
    
    % Extract relevant time window (501:1000) and normalize time
    tau_hat_temp = tau_xx ./ P0;
    tau_hat = tau_hat_temp(501:1000, 1);
    t = t(501:1000, 1) - 15;  % Time normalization
    
    % Find first occurrence where tau reaches target value
    idx = find(tau_hat >= target_value, 1, 'first');
    
    if ~isempty(idx)
        if tau_hat(idx) == target_value
            relaxation_time_dem(i) = t(idx) * gamma0;
        else
            if idx > 1
                % Linear interpolation for more precise time
                t1 = t(idx-1); t2 = t(idx);
                y1 = tau_hat(idx-1); y2 = tau_hat(idx);
                relaxation_time_dem(i) = (t1 + (target_value - y1)*(t2 - t1)/(y2 - y1)) * gamma0;
            else
                relaxation_time_dem(i) = t(idx) * gamma0;
            end
        end
    else
        relaxation_time_dem(i) = NaN;
        warning(sprintf('Case %d does not reach target value within time range', i));
    end
    
    %% Method 2: Relaxation time from exponential fitting
    % Fitting function: mu0 - 2*mu0*exp(-gamma0*t/t0)
    fit_function = @(t0, t) mu0 - 2*mu0*exp(-gamma0*t./t0);
    
    % Initial guess and curve fitting
    try
        t0_guess = 0.05;
        t0_fit = lsqcurvefit(fit_function, t0_guess, t, tau_hat);
        relaxation_time_fit(i) = t0_fit;
    catch ME
        relaxation_time_fit(i) = NaN;
        warning(sprintf('Case %d fitting failed: %s', i, ME.message));
    end
    
    %% Method 3: Relaxation time from analytical solution
    c1 = C1(1,i);
    c2 = C2(1,i);
    c3 = C3(1,i);
    
    % Solve quadratic equation C3*A^2 - C2*A + C1 = 0 for steady-state values
    p2 = [c3, -c2, c1];
    r = roots(p2);
    real_roots = r(abs(imag(r)) < 1e-6);
    
    if isempty(real_roots)
        relaxation_time_analytical(i) = NaN;
        continue;
    end
    
    % Select physically meaningful root
    mu_0 = real_roots(2,1);
    target_value = 0.2 * mu_0;

    if c2 == 0
        relaxation_time_analytical(i) = NaN;
        warning(sprintf('Case %d has C2=0, cannot calculate analytical solution', i));
        continue;
    end
    
    % Calculate analytical relaxation time
    ratio = (c3 * mu_0) / c2;
    log_arg = 1 - 2*(1 - exp(-1)) * ratio;
    
    if log_arg <= 0
        relaxation_time_analytical(i) = NaN;
        warning(sprintf('Case %d has invalid logarithm argument (%.4f)', i, log_arg));
        continue;
    end
    
    denominator = gamma0 * abs(c2 - 2*c3*mu_0);
    if denominator == 0
        relaxation_time_analytical(i) = NaN;
        warning(sprintf('Case %d has zero denominator in analytical calculation', i));
        continue;
    end
    
    relaxation_time_analytical(1,i) = (1 + log(log_arg)) / denominator * gamma0;
end

%% Calculate Dimensionless Numbers
Pi_1 = E ./ p;
Pi_2 = nH;  % Originally nH.*ds in commented code

%% Comparison Plot: DEM vs Analytical Relaxation Times
figure('Position', [100, 100, 800, 600]); 

% Determine axis limits
x_min = min(relaxation_time_dem);
x_max = max(relaxation_time_dem);
y_min = min(relaxation_time_analytical);
y_max = max(relaxation_time_analytical);

range_min = min([x_min, y_min]);
range_max = max([x_max, y_max]);

% Configure plot range
axis([range_min, range_max, range_min, range_max]);

% Generate reference line (y = x) and confidence interval
x_vals = linspace(0.02, 0.18, 500);
y_vals = x_vals;
confidence_offset = 0.1 * (range_max - range_min);
y_lower = y_vals - confidence_offset;
y_upper = y_vals + confidence_offset;

% Plot confidence interval
h_conf = fill([x_vals, fliplr(x_vals)], [y_lower, fliplr(y_upper)], ...
    [1, 0.85, 0.85], 'EdgeColor', 'none', 'FaceAlpha', 0.55, ...
    'DisplayName', '95% Confidence Interval for y = x');

hold on;

% Plot data points with color coding
h_data = scatter(relaxation_time_dem, relaxation_time_analytical, 50, ...
    log10(Pi_1 .* (Pi_2) .^ (1)), ...
    'filled', ...
    'MarkerEdgeColor', [0.2, 0.2, 0.2], ...
    'MarkerFaceAlpha', 0.9, ...
    'LineWidth', 0.8, ...
    'DisplayName', 'Data Points (DEM vs Analytical)');

% Add colorbar
c = colorbar;
c.Limits = [3, 6];
tick_labels = arrayfun(@(x) sprintf('%.1f', x), c.Ticks, 'UniformOutput', false);
c.TickLabels = tick_labels;

% Plot reference line y = x
h_ref = plot(x_vals, y_vals, 'Color', [0.79, 0.09, 0.11, 0.9], ...
    'LineStyle', '-', 'LineWidth', 3.2, ...
    'DisplayName', 'Reference Line: y = x');

% Format plot
box on;
legend([h_data, h_ref, h_conf], 'Data Points', 'Reference Line: y = x', ...
    '95% Confidence Interval', 'Location', 'best', ...
    'FontSize', 19, 'FontName', 'Arial');

set(gca, 'FontName', 'Arial', 'FontSize', 19, 'LineWidth', 1.6);
xticks = linspace(0.02, 0.18, 5);
set(gca, 'XTick', xticks, 'YTick', xticks);
xtickformat('%.2f');
ytickformat('%.2f');
xlim([0.02, 0.18]);
ylim([0.02, 0.18]);

xlabel('Relaxation Time (DEM)', 'FontSize', 16, 'FontName', 'Arial');
ylabel('Relaxation Time (Analytical)', 'FontSize', 16, 'FontName', 'Arial');
title('Comparison of Relaxation Time Calculation Methods', 'FontSize', 18, 'FontName', 'Arial');

hold off;

%% Exponential Fit Verification Plot
% Verify the exponential fitting for the last case processed
fit_value = mu0 - 2*mu0.*exp(-gamma0.*t./relaxation_time_fit(end));
figure('Position', [200, 200, 800, 600]);
scatter(t, tau_hat, 30, 'b', 'filled', 'DisplayName', 'DEM Data');
hold on;
plot(t, fit_value, 'r-', 'LineWidth', 2, 'DisplayName', 'Exponential Fit');
xlabel('Time', 'FontSize', 14, 'FontName', 'Arial');
ylabel('Tau Hat', 'FontSize', 14, 'FontName', 'Arial');
title('Exponential Fit Verification', 'FontSize', 16, 'FontName', 'Arial');
legend('Location', 'best');
box on;
hold off;

%% Power Law Fitting: Relaxation Time vs Dimensionless Number
% Prepare data for fitting (exclude zeros)
pi3 = log10(Pi_1 .* (Pi_2) .^ (1));
nonZeroIdx = pi3 ~= 0;

pi3_nonzero = pi3(nonZeroIdx);
relaxation_dem_nonzero = relaxation_time_dem(nonZeroIdx);
relaxation_analytical_nonzero = relaxation_time_analytical(nonZeroIdx);

% Define fitting model: y = a/(x - x0)
fitmodel = fittype('a/(x - x0)', 'independent', 'x', 'dependent', 'y', ...
    'coefficients', {'a', 'x0'});

% Configure fitting options
options = fitoptions(fitmodel);
options.StartPoint = [1, 0];
options.Lower = [-Inf, -Inf];
options.Upper = [Inf, Inf];

% Perform fitting
[fittedcurve, gof] = fit(pi3_nonzero', relaxation_dem_nonzero', fitmodel, options);
a = fittedcurve.a;
x0 = fittedcurve.x0;

fprintf('Fitting parameters: a = %.4f, x0 = %.4f\n', a, x0);
fprintf('R² = %.4f\n', gof.rsquare);

% Generate data for fitted curve
x_fit = linspace(3, max(pi3_nonzero), 500);
y_fit = feval(fittedcurve, x_fit);

% Calculate confidence intervals
[pred_int, ~] = predint(fittedcurve, x_fit, 0.95, 'observation', 'on');

% Create fitting plot
figure('Color', 'w', 'Position', [100, 100, 800, 600]);
hold on;

% Plot confidence interval
h_conf_fit = fill([x_fit, fliplr(x_fit)], [pred_int(:,1)', fliplr(pred_int(:,2)')], ...
    [1, 0.85, 0.85], 'EdgeColor', 'none', 'FaceAlpha', 0.55, ...
    'DisplayName', '95% Confidence Interval');

% Plot data points
h_dem = scatter(pi3_nonzero, relaxation_dem_nonzero, 48, ...
    [0.15, 0.44, 0.71], 'filled', 'MarkerEdgeColor', [0.2, 0.2, 0.2], ...
    'MarkerFaceAlpha', 0.9, 'LineWidth', 0.8, ...
    'DisplayName', 'Results from DEM');

h_analytical = scatter(pi3_nonzero, relaxation_analytical_nonzero, 48, ...
    [0.992, 0.588, 0.251], 'filled', 'Marker', '^', ...
    'MarkerEdgeColor', [0.2, 0.2, 0.2], 'MarkerFaceAlpha', 0.9, ...
    'LineWidth', 0.8, 'DisplayName', 'Results from Analytical Solution');

% Plot fitted curve
h_fit_line = plot(x_fit, y_fit, 'Color', [0.79, 0.09, 0.11, 0.8], ...
    'LineStyle', '-', 'LineWidth', 3.2, 'DisplayName', 'Fitted Curve');

% Format plot
legend([h_dem, h_analytical, h_fit_line, h_conf_fit], ...
    'Results from DEM', 'Results from Analytical Solution', ...
    'Fitted Curve', '95% Confidence Interval', ...
    'Location', 'best', 'FontSize', 19, 'FontName', 'Arial');

set(gca, 'FontName', 'Arial', 'LineWidth', 1.6, 'FontSize', 19);
xlabel('$\log\left(\Pi_1 \cdot \Pi_2\right)$', 'FontSize', 16, 'Interpreter', 'latex');
ylabel('Relaxation Time', 'FontSize', 16, 'FontName', 'Arial');
title('Relaxation Time vs Dimensionless Number', 'FontSize', 18, 'FontName', 'Arial');

xlim([3, 6]);
ylim([0, 0.25]);
xtickformat('%.2f');
ytickformat('%.2f');
box on;
hold off;

%% Log-Log Scale Fitting Plot
% Prepare data for log-log fitting
pi3_log = (Pi_1 .* (Pi_2) .^ (1));
nonZeroIdx_log = pi3_log ~= 0;

pi3_nonzero_log = pi3_log(nonZeroIdx_log);
relaxation_dem_nonzero_log = relaxation_time_dem(nonZeroIdx_log);
relaxation_analytical_nonzero_log = relaxation_time_analytical(nonZeroIdx_log);

% Perform fitting on original scale
[fittedcurve_log, gof_log] = fit(pi3_nonzero_log', relaxation_dem_nonzero_log', fitmodel, options);
a_log = fittedcurve_log.a;
x0_log = fittedcurve_log.x0;

fprintf('Log-log fitting parameters: a = %.4f, x0 = %.4f\n', a_log, x0_log);
fprintf('Log-log R² = %.4f\n', gof_log.rsquare);

% Generate fitted curve data
x_fit_log = linspace(10^3, 10^6, 500);
y_fit_log = feval(fittedcurve_log, x_fit_log);

% Calculate confidence intervals
[pred_int_log, ~] = predint(fittedcurve_log, x_fit_log, 0.95, 'observation', 'on');

% Create log-log plot
figure('Color', 'w', 'Position', [100, 100, 800, 600]);
hold on;
set(gca, 'XScale', 'log', 'YScale', 'log');

% Plot confidence interval
h_conf_log = fill([x_fit_log, fliplr(x_fit_log)], [pred_int_log(:,1)', fliplr(pred_int_log(:,2)')], ...
    [1, 0.85, 0.85], 'EdgeColor', 'none', 'FaceAlpha', 0.55, ...
    'DisplayName', '95% Confidence Interval');

% Plot data points
h_dem_log = scatter(pi3_nonzero_log, relaxation_dem_nonzero_log, 48, ...
    [0.15, 0.44, 0.71], 'filled', 'MarkerEdgeColor', [0.2, 0.2, 0.2], ...
    'MarkerFaceAlpha', 0.9, 'LineWidth', 0.8, ...
    'DisplayName', 'Results from DEM');

h_analytical_log = scatter(pi3_nonzero_log, relaxation_analytical_nonzero_log, 48, ...
    [0.992, 0.588, 0.251], 'filled', 'Marker', '^', ...
    'MarkerEdgeColor', [0.2, 0.2, 0.2], 'MarkerFaceAlpha', 0.9, ...
    'LineWidth', 0.8, 'DisplayName', 'Results from Analytical Solution');

% Format plot
legend([h_dem_log, h_analytical_log, h_conf_log], ...
    'Results from DEM', 'Results from Analytical Solution', ...
    '95% Confidence Interval', 'Location', 'best', ...
    'FontSize', 19, 'FontName', 'Arial');

set(gca, 'FontName', 'Arial', 'LineWidth', 1.6, 'FontSize', 19);
xlabel('$\Pi_1 \cdot \Pi_2$', 'FontSize', 16, 'Interpreter', 'latex');
ylabel('Relaxation Time', 'FontSize', 16, 'FontName', 'Arial');
title('Relaxation Time vs Dimensionless Number (Log-Log Scale)', 'FontSize', 18, 'FontName', 'Arial');

xlim([10^3, 10^6]);
ylim([10^(-2), 2*10^(-1)]);
xtickformat('%.e');
ytickformat('%.e');
box on;
hold off;
