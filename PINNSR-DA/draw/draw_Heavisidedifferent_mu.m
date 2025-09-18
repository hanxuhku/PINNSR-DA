clear all;
clc;

%% Define data list
% data_list = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90];
% data_list = [0.50, 0.60, 0.70, 0.80, 0.90];
% data_list = [0.20, 0.30, 0.40, 0.50, 0.90]; % Values ≥0.6 show significant fluctuations?
% data_list = [0.20, 0.30, 0.40, 0.50]; 
% data_list = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50]; 
% data_list = [0.30, 0.40, 0.50, 0.6]; 
data_list = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.6]; 
% data_list = [0.40, 0.5];

%% Automatically load data
data = cell(size(data_list));
base_path = 'H:\Batch_OscillatoryShear_6_2\mu\p10000_ds0.005_rho2500_nH15_T10_Ep5e7_';
file_tail = '_Gamma0.50\post\Tau_Gammat.mat';
% file_tail = '_rho2500_nH15_T10_Ep5e7_mu0.50_I5.00e-04\post\Tau_Gammat.mat';

for i = 1:length(data_list)
    % Construct full file path
    mu_value = data_list(i);
    file_path = [base_path, 'mu', num2str(mu_value, '%0.2f'), file_tail];
    
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
        
        % Extract μ_p value for legend
        mu_value = data_list(i);
        
        % Plot curve and store handle for legend
        % h = plot(current_data.t(1:1201,1)-current_data.T, current_data.tau_xx(1:1201,1)./current_data.P0, ...
        %     'Color', current_color, ...
        %     'LineWidth', 2.5);
        h = plot((current_data.t(1:1201,1)-current_data.T)*current_data.gamma0/2.5, current_data.tau_xx(1:1201,1)./current_data.P0, ...
            'Color', current_color, ...
            'LineWidth', 2.5);
        
        % Store legend entry
        % legend_entries{i} = sprintf('μ_p = %.2f', mu_value);
        legend_entries{i} = sprintf('$\\mu_p = %.2f$', mu_value);
                
        hold on
    end
end

hold off

% Specify LaTeX interpreter for legend
legend(legend_entries, 'Interpreter', 'latex', 'FontSize', 12);

% Set axis labels
% xlabel('t (s)', 'FontName', 'Arial', 'FontSize', 20);
xlabel('$\dot{\gamma}_{0}t$', 'FontName', 'Arial', 'FontSize', 20, 'Interpreter', 'latex');
ylabel('\fontname{Arial} \it\mu', ...
    'FontSize', 22, ...
    'Interpreter', 'tex');
% ylabel('\fontname{Arial} Normalized shear stress \it\tau\rm /\it\sigma', ...
%     'FontSize', 20, ...
%     'Interpreter', 'tex');

% Set axis ranges and ticks
% xlim([0 12])
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
% c.Label.String = 'Inter-particle friction coefficient';
% c.Label.String = '$\hat{\mu}$';
c.Label.String = '$\mu_p$';
c.Label.Interpreter = 'latex';  % Specify LaTeX interpreter
c.Label.FontSize = 20;
c.Label.FontName = 'Arial';
set(c, 'FontSize', 18, 'FontName', 'Arial');

% Set colorbar range
caxis([min(data_list), max(data_list)]);
% Set colorbar ticks, displaying 7 ticks
c.Ticks = linspace(min(data_list), max(data_list), 7);

% Mark positions corresponding to data points on colorbar
hold on
for i = 1:num_curves
    % Calculate normalized position of current μ_p value on colorbar
    pos = (data_list(i) - min(data_list)) / (max(data_list) - min(data_list));
end
hold off
