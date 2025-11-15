% Load the CSV file
file_path = 'E:\Collaboration Work\With Saif\Asif Work\change_of_refl_90deg.csv';
data = readmatrix(file_path);

% Normalize the data for better contrast (optional)
data = (data - min(data(:))) / (max(data(:)) - min(data(:))); % Scale between 0 and 1

% Create figure with high resolution
figure('Units', 'normalized', 'Position', [0 0 1 1]); % Full-screen figure

% Display the data as a heatmap
imagesc(data);
set(gca, 'FontSize', 14); % Set font size for all text elements

% Choose the best colormap for thermal representation
colormap("hot"); % Options: "hot", "parula", "inferno", "jet", "turbo"

% Add a high-quality colorbar with labels
c = colorbar;
c.Label.String = 'Reflectivity Intensity';
c.Label.FontSize = 16;
c.Label.FontWeight = 'bold';

% Add title with formatting
title('High-Quality Thermal Map of Reflectivity Data', 'FontSize', 18, 'FontWeight', 'bold');

% Add x and y axis labels with bold formatting
xlabel('Columns (Spatial X-Axis)', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('Rows (Spatial Y-Axis)', 'FontSize', 16, 'FontWeight', 'bold');

% Set the tick labels to be bold and clearly visible
set(gca, 'FontSize', 14, 'FontWeight', 'bold', 'LineWidth', 1.5);

% Save the high-resolution figure (300 DPI for publication quality)
exportgraphics(gcf, 'ThermalMap_HighRes.png', 'Resolution', 300);
