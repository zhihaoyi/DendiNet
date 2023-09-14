% Specify the folder containing the .mat files
folder = '/Users/apple/Desktop/data/0deg/';

% Get a list of .mat files in the folder
matFiles = dir(fullfile(folder, '*.mat'));

% Loop through each .mat file
for i = 1:numel(matFiles)
    % Load the .mat file
    matData = load(fullfile(folder, matFiles(i).name));
    
    % Extract the struct cell from the loaded data
    dataCell = matData.meas;  % Replace 'data' with the appropriate struct cell name
    
    % Create an Excel filename for the current .mat file
    excelFile = fullfile(folder, [matFiles(i).name, '.xlsx']);
    
    % Convert the struct cell to a table
    dataTable = struct2table(dataCell);
    
    % Write the table to an Excel file
    writetable(dataTable, excelFile);
end
