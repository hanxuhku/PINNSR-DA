clear all;
clc;

%% Define data list
data_list = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40];

%% Automatically load data
data = cell(size(data_list));
base_path = 'H:\Batch_OscillatoryShear_5\Heaviside\p10000_ds0.005_rho2500_nH15_T10_';
file_tail = '_Heaviside\post\Tau_Gammat.mat';
% file_tail = '_rho2500_nH15_T10_Ep5e7_mu0.50_I5.00e-04\post\Tau_Gammat.mat';

for i = 1:length(data_list)
    % Construct full file path
    gamma_value = data_list(i);
    file_path = [base_path, 'gam', num2str(gamma_value, '%0.2f'), file_tail];
    
    % Load data
    try
        data{i} = load(file_path);
        fprintf('Successfully loaded: %s\n', file_path);
    catch e
        fprintf('Failed to load: %s\nError: %s\n', file_path, e.message);
        data{i} = []; % Set to empty to indicate loading failure
    end
end

num_curves = length(data);

%%
% ==============================================
% Improved color generation system (only need to define start and end colors)
% ==============================================
% light_color = [0.60, 0.78, 1.00];  % Light blue (RGB)
% dark_color  = [0.00, 0.20, 0.65];  % Dark blue (RGB)

% light_color = [1.00, 0.70, 0.40];   % Light orange (RGB)
% dark_color  = [0.60, 0.00, 0.00];   % Dark red (RGB)

% light_color = [0.85, 0.80, 0.95];   % Light purple (RGB)
% dark_color  = [0.25, 0.00, 0.50];   % Dark purple (RGB)

light_color = [239, 229, 237] / 255;  % Start color: #EFE5ED (light purple)
% dark_color = [137, 78, 123] / 255;    % End color: #894E7B (dark magenta)
% dark_color = [145, 65, 185] / 255;    % End color: #9141B9 (dark magenta)
dark_color = [131, 59, 167] / 255;    % End color: #833BA7 (dark magenta)

% light_color = [0.80, 0.95, 0.80];   % Light green (RGB)
% dark_color  = [0.00, 0.40, 0.00];   % Dark green (RGB)

% light_color = [248, 214, 229] / 255;  % Start color: #F8D6E5 (light pink)
% dark_color = [98, 110, 152] / 255;  % End color: #626E98 (blue-purple)
% 
% light_color = [239, 232, 232] / 255; % Start color: #EFE8E8 (light beige)
% dark_color = [142, 109, 99] / 255;  % End color: #8E6D63 (warm brown)

% Automatically generate gradient color array
colors = zeros(num_curves, 3);
for i = 1:num_curves
    % Calculate interpolation ratio for current color (0~1)
    t = (i - 1) / (num_curves - 1);
    
    % Generate gradient color using linear interpolation
    colors(i, :) = (1 - t) * light_color + t * dark_color;
end
% ==============================================

% Create figure and set dimensions
figure
set(gcf, 'Position', [100, 100, 1100, 650]); 

% Plot all curves
legend_entries = cell(1, num_curves); % Pre-allocate legend entries
for i = 1:num_curves
    if ~isempty(data{i})
        current_data = data{i};
        % Get color corresponding to current curve
        current_color = colors(i, :);
        
        % Extract gamma value for legend
        gamma_value = data_list(i);
        
        % Plot curve and store handle for legend
        h = plot((current_data.t(1:1201,1)-1*current_data.T)*current_data.gamma0, current_data.tau_xx(1:1201,1)./current_data.P0, ...
            'Color', current_color, ...
            'LineWidth', 3.5);
        
        % Store legend entry
        legend_entries{i} = sprintf('$\\dot{\\gamma}_{0}$ = %.2f', gamma_value); 
        
        hold on
    end
end

hold off

% % Set axis labels and title
% xlabel('$\dot{\gamma}_{0}t$', 'FontName', 'Arial', 'FontSize', 20, 'Interpreter', 'latex');
% ylabel('\fontname{Arial} Normalized shear stress \it\tau\rm /\it\sigma', ...
%     'FontSize', 20, ...
%     'Interpreter', 'tex');

% Set axis ranges and ticks
xlim([0 inf])
ylim([-0.5 0.5])

% Set axis tick font and grid properties
set(gca, 'FontName', 'Arial', 'FontSize', 26, 'GridLineStyle', '--', 'GridAlpha', 0.3);
grid off

% Configure legend
if ~isempty(legend_entries)
    legend(legend_entries, 'FontSize', 20, 'Location', 'northwest', 'FontName', 'Arial', 'Interpreter', 'latex');
end

% Configure background and border
set(gca, 'Color', [0.96 0.96 0.96]); % Light gray background
box on

% Optimize layout
set(gcf, 'Color', 'white'); % White figure background
set(gca, 'Position', [0.1 0.15 0.75 0.75]); % Adjust plot area position and size

% ==============================================
% Add Colorbar
% ==============================================
% Create colormap corresponding to curve colors
cmap = zeros(100, 3);
for i = 1:100
    t = (i-1)/99;
    cmap(i,:) = light_color + t * (dark_color - light_color);
end

% Set current colormap
colormap(cmap);

% Create colorbar
c = colorbar;
c.Label.String = '$\dot{\gamma}_{0}$'; % LaTeX formatted label
c.Label.FontSize = 26;
c.Label.FontName = 'Arial';
c.Label.Interpreter = 'latex'; % Set LaTeX interpreter for colorbar label
c.TickLabelInterpreter = 'latex'; % Set LaTeX interpreter for tick labels
set(c, 'FontSize', 26, 'FontName', 'Arial'); % Set tick font properties

% Set colorbar range
caxis([min(data_list), max(data_list)]);
% Set colorbar ticks, displaying 7 ticks
c.Ticks = linspace(min(data_list), max(data_list), 7);
% Custom tick labels with two decimal places
tick_values = c.Ticks;
tick_labels = cell(size(tick_values));
for i = 1:length(tick_values)
    tick_labels{i} = sprintf('%.2f', tick_values(i));
end
c.TickLabels = tick_labels;

% Mark positions corresponding to data points on colorbar
hold on
for i = 1:num_curves
    % Calculate normalized position of current gamma value on colorbar
    pos = (data_list(i) - min(data_list)) / (max(data_list) - min(data_list));
end
hold off
% xlim([0,5]);
