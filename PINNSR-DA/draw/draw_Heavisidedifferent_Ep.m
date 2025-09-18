clear all;
clc;

%% Define data list
% data_list = [10, 12, 15, 18, 20, 25, 30];
% data_list = {'5e7', '2e8', '3e8', '5e8', '7e8', '9e8', '1e9'};
data_list = {'1e7', '2e7', '3e7', '4e7', '5e7', '6e7', '7e7', '8e7', '9e7', '1e8'};
% data_list = {'1e7', '1e8'};

%% Automatically load data
data = cell(size(data_list));
base_path = 'H:\Batch_OscillatoryShear_6_2\Ep\p10000_ds0.005_rho2500_nH15_T10_';
% file_tail = '_mu0.50_I5.00e-04\post\Tau_Gammat.mat';
% file_tail = '_mu0.50_I7.50e-04\post\Tau_Gammat.mat';
file_tail = '_mu0.50_Gamma0.50\post\Tau_Gammat.mat';

for i = 1:length(data_list)
    % Construct full file path
    ep_value = data_list{i};  % Access cell content using curly braces
    file_path = [base_path, 'Ep', ep_value, file_tail];  % Direct string concatenation
    
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

light_color = [0.85, 0.80, 0.95];   % Light purple (RGB)
dark_color  = [0.25, 0.00, 0.50];   % Dark purple (RGB)

% light_color = [239, 229, 237] / 255;  % Start color: #EFE5ED (light purple)
% dark_color = [137, 78, 123] / 255;    % End color: #894E7B (dark magenta)

% light_color = [0.80, 0.95, 0.80];   % Light green (RGB)
% dark_color  = [0.00, 0.40, 0.00];   % Dark green (RGB)

% light_color = [248, 214, 229] / 255;  % Start color: #F8D6E5 (light pink)
% dark_color = [98, 110, 152] / 255;  % End color: #626E98 (blue-purple)

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
set(gcf, 'Position', [100, 100, 900, 500]); 

% Plot all curves
legend_entries = cell(1, num_curves); % Pre-allocate legend entries
for i = 1:num_curves
    if ~isempty(data{i})
        current_data = data{i};
        % Get color corresponding to current curve
        current_color = colors(i, :);
        
        % Extract Ep value for legend
        ep_str = data_list{i};
        
        % Convert scientific notation string to numeric value and format
        ep_num = str2double(ep_str);
        ep_formatted = sprintf('%.2e', ep_num);
        
        % Plot curve and store handle for legend
        h = plot(current_data.t(1:1201,1)-current_data.T, current_data.tau_xx(1:1201,1)./current_data.P0, ...
            'Color', current_color, ...
            'LineWidth', 2.5);
        
        % Store legend entry
        legend_entries{i} = sprintf('Ep = %s', ep_formatted);
        
        hold on
    end
end

hold off

% Set axis labels
xlabel('t (s)', 'FontName', 'Arial', 'FontSize', 20);
ylabel('\fontname{Arial} Normalized shear stress \it\tau\rm /\it\sigma', ...
    'FontSize', 20, ...
    'Interpreter', 'tex');

% Set axis ranges and ticks
xlim([0 12])
ylim([-0.5 0.5])

% Set axis tick font and grid properties
set(gca, 'FontName', 'Arial', 'FontSize', 18, 'GridLineStyle', '--', 'GridAlpha', 0.3);
grid off

% Configure legend
if ~isempty(legend_entries)
    legend(legend_entries, 'FontSize', 16, 'Location', 'northwest', 'FontName', 'Arial');
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
c.Label.String = 'Ep (Pa)';
c.Label.FontSize = 20;
c.Label.FontName = 'Arial';
set(c, 'FontSize', 18, 'FontName', 'Arial');

% Convert data_list to numeric array for calculations
data_list_num = str2double(data_list);

% Set colorbar range
caxis([min(data_list_num), max(data_list_num)]);

% Set colorbar ticks, displaying 5 ticks
c.Ticks = linspace(min(data_list_num), max(data_list_num), 5);

% Format tick labels in scientific notation
c.TickLabels = arrayfun(@(x) sprintf('%.1e', x), c.Ticks, 'UniformOutput', false);

% Mark positions corresponding to data points on colorbar
hold on
for i = 1:num_curves
    % Calculate normalized position of current Ep value on colorbar
    pos = (data_list_num(i) - min(data_list_num)) / (max(data_list_num) - min(data_list_num));
    
    % Calculate actual position on colorbar
    y_pos = min(c.Limits) + pos * (max(c.Limits) - min(c.Limits));
    
    % Create marker line (horizontal tick)
    % line([c.Position(1)-0.02, c.Position(1)], [y_pos, y_pos], 'Color', 'k', 'LineWidth', 1.5);
end
hold off
    