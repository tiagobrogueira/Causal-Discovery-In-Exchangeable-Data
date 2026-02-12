function output_path = run_sim(func, datasets, read_dir, write_dir, overwrite, varargin)
% RUN_SIM  Run SIM causal evaluation on synthetic datasets
%
% func      Function handle accepting Nx(M+N) matrix [X,Y]
% datasets  Cell array of dataset names (optional)
% read_dir  Directory containing SIM datasets
% write_dir Directory to save results
% overwrite Logical, whether to overwrite existing results
% varargin  Optional parameters passed to func

project_root = fileparts(mfilename('fullpath'));

if nargin < 2 || isempty(datasets)
    datasets = {};
end
if nargin < 3 || isempty(read_dir)
    read_dir = fullfile(project_root, "..",'benchmarks', 'synthetic', 'SIM-Mooij');
end
if nargin < 4 || isempty(write_dir)
    write_dir = fullfile(project_root, "..", 'results');
end
if nargin < 5 || isempty(overwrite)
    overwrite = false;
end

% -----------------------------
% Dataset selection
% -----------------------------
all_entries = dir(read_dir);
all_entries = all_entries([all_entries.isdir]);  % only directories
datasets = {all_entries(3:end).name}; % skip '.' and '..'

if isempty(datasets)
    error('No datasets found in %s', read_dir);
end
fprintf('Datasets detected: %s\n', strjoin(datasets, ', '));

if ~exist(write_dir, 'dir')
    mkdir(write_dir);
end

output_path = fullfile(write_dir, 'SIM_scores.csv');

% -----------------------------
% Load existing CSV if available
% -----------------------------
if isfile(output_path)
    df_existing = readtable(output_path);
else
    df_existing = table([], [], [], [], [], [], datetime([],[],[],'TimeZone','UTC'), ...
                        'VariableNames', {'method', 'parameters', 'dataset', 'Pair', 'score', 'weight', 'timestamp'});
end

method_name = func2str(func);
method_name_lower = lower(method_name);
parameters = ''; % MATLAB doesn't have serialize_params like R; can extend if needed

% -----------------------------
% Loop over datasets
% -----------------------------
for d = 1:length(datasets)
    dataset = datasets{d};
    fprintf('\n🚀 Running SIM dataset: %s\n', dataset);
    
    dataset_dir = fullfile(read_dir, dataset);
    meta_file = fullfile(dataset_dir, 'pairmeta.txt');
    if ~isfile(meta_file)
        error('Missing pairmeta.txt in %s', dataset_dir);
    end
    
    % Load pairmeta.txt (no header)
    meta = readmatrix(meta_file);
    % Columns: pair, c_start, c_end, e_start, e_end, weight
    pair_ids = meta(:,1);
    
    % -----------------------------
    % Loop through all pairs
    % -----------------------------
    for i = 1:size(meta,1)
        pair_id = pair_ids(i);
        c_start = meta(i,2);
        c_end   = meta(i,3);
        e_start = meta(i,4);
        e_end   = meta(i,5);
        weight  = meta(i,6);
        
        % Check if already computed
        exists = any(strcmp(lower(string(df_existing.method)), method_name_lower) & ...
                     strcmp(string(df_existing.dataset), dataset) & ...
                     df_existing.Pair == pair_id);
        if exists && ~overwrite
            fprintf('⏩ Skipping %s Pair %d (already computed)\n', dataset, pair_id);
            continue;
        end
        
        % Load pair file
        pair_file = fullfile(dataset_dir, sprintf('pair%04d.txt', pair_id));
        if ~isfile(pair_file)
            fprintf('⚠️ Missing pair file: %s, skipping.\n', pair_file);
            continue;
        end
        data = readmatrix(pair_file);
        
        % Extract cause and effect
        X = data(:, c_start:c_end);
        Y = data(:, e_start:e_end);
        
        % Run method
        try
            score = func([X, Y], varargin{:});
        catch ME
            fprintf('⚠️ Error on %s Pair %d: %s\n', dataset, pair_id, ME.message);
            score = NaN;
        end
        
        % Append row
        new_row = {method_name, "", dataset, pair_id, score, weight, string(datetime('now','TimeZone','UTC'))};
        %new_row_tbl = cell2table(new_row, 'VariableNames', df_existing.Properties.VariableNames);
        new_row_tbl=new_row;
        
        % Overwrite if needed
        if overwrite
            idx = strcmp(lower(string(df_existing.method)), method_name_lower) & ...
                  strcmp(string(df_existing.dataset), dataset) & ...
                  df_existing.Pair == pair_id;
            df_existing(idx,:) = [];
        end
        
        df_existing = [df_existing; new_row_tbl];
        writetable(df_existing, output_path);
        fprintf('✔ Saved %s Pair %d\n', dataset, pair_id);
    end
end

fprintf('\n✅ Saved SIM results to %s\n', output_path);
end


function json_str = serialize_params(kwargs)
    json_str="";
end