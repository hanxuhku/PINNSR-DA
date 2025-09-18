% Clear workspace and command window
clear all;
clc;

%% Define data list
data_list = [5000, 15000, 20000, 35000, 40000, 45000, 50000, 200000, 300000, 400000, 500000];

%% Automatically load data
data = cell(size(data_list));
base_path = 'H:\Batch_OscillatoryShear_6_2\P\';
file_tail = '_ds0.005_rho2500_nH15_T10_Ep5e7_mu0.50_Gamma0.50\post\Tau_Gammat.mat';

for i = 1:length(data_list)
    % Construct complete file path
    p_value = data_list(i);
    file_path = [base_path, 'p', num2str(p_value), file_tail];
    
    % Load data with error handling
    try
        data{i} = load(file_path);
        fprintf('Successfully loaded: %s\n', file_path);
    catch e
        fprintf('Failed to load: %s\nError: %s\n', file_path, e.message);
        data{i} = []; % Set to empty if loading fails
    end
end

num_curves = length(data);

% ==============================================
% Improved color generation system 
% (only need to define start and end colors)
% ==============================================
light_color = [0.85, 0.80, 0.95];   % Light purple (RGB)
dark_color  = [0.25, 0.00, 0.50];   % Dark purple (RGB)

% Automatically generate gradient color array
colors = zeros(num_curves, 3);
for i = 1:num_curves
    % Calculate interpolation ratio (0~1)
    t = (i - 1) / (num_curves - 1);
    
    % Linear interpolation to generate gradient colors
    colors(i, :) = (1 - t) * light_color + t * dark_color;
end
% ==============================================

% Create figure and set dimensions
figure
set(gcf, 'Position', [100, 100, 900, 500]); 

% Plot all curves
legend_entries = cell(1, num_curves); % Preallocate legend entries
for i = 1:num_curves
    if ~isempty(data{i})
        current_data = data{i};
        % Get color for current curve
        current_color = colors(i, :);
        
        % Extract pressure value for legend
        pressure_value = data_list(i);
        
        % Plot curve and store handle for legend
        h = plot(current_data.t(1:1201,1)-current_data.T, current_data.tau_xx(1:1201,1)./current_data.P0, ...
            'Color', current_color, ...
            'LineWidth', 2.5);
        
        % Store legend entry
        legend_entries{i} = sprintf('P = %d kPa', pressure_value/1000); % Convert to kPa
        
        hold on
    end
end

hold off

% Set axis labels
xlabel('t (s)', 'FontName', 'Arial', 'FontSize', 20);
ylabel('\fontname{Arial} Normalized shear stress \it\tau\rm /\it\sigma', ...
    'FontSize', 20, ...
    'Interpreter', 'tex');

% Set axis ranges
xlim([0 12])
ylim([-0.5 0.5])

% Configure axis properties
set(gca, 'FontName', 'Arial', 'FontSize', 18, 'GridLineStyle', '--', 'GridAlpha', 0.3);
grid off

% Configure legend
if ~isempty(legend_entries)
    legend(legend_entries, 'FontSize', 16, 'Location', 'northwest', 'FontName', 'Arial');
end

% Configure plot appearance
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
c.Label.String = 'Pressure (kPa)';
c.Label.FontSize = 20;
c.Label.FontName = 'Arial';
set(c, 'FontSize', 18, 'FontName', 'Arial');

% Set colorbar range
caxis([min(data_list)/1000, max(data_list)/1000]); % Convert to kPa
% Set colorbar ticks to display 10 intervals
c.Ticks = linspace(min(data_list)/1000, max(data_list)/1000, 10);

% Mark positions corresponding to data points on colorbar
hold on
for i = 1:num_curves
    % Calculate position of current pressure value on colorbar
    pos = (data_list(i)/1000 - min(data_list)/1000) / (max(data_list)/1000 - min(data_list)/1000);
end
hold off
