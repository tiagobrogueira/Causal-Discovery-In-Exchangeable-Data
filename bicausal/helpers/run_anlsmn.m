function output_path = run_anlsmn(func, datasets, read_dir, write_dir, overwrite, varargin)
% RUN_ANLSMN Run ANLSMN benchmark datasets with a given function
%
% output_path = run_anlsmn(func, datasets, read_dir, write_dir, overwrite, ...)
%
% func      - function handle that takes an Nx2 matrix [X, Y]
% datasets  - cell array of dataset names (optional)
% read_dir  - directory containing ANLSMN datasets
% write_dir - directory to save results
% overwrite - logical, whether to recompute existing results
% varargin  - additional parameters for func

project_root = fileparts(mfilename('fullpath'));
if nargin < 2 || isempty(datasets)
    read_dir = fullfile(project_root,"..", 'benchmarks', 'synthetic', 'ANLSMN-Tagasovska');
end
if nargin < 3 || isempty(write_dir)
    write_dir = fullfile(project_root, "..", 'results');
end
if nargin < 4 || isempty(overwrite)
    overwrite = false;
end

% -----------------------------
% List datasets if not provided
% -----------------------------
if nargin < 2 || isempty(datasets)
    dirs = dir(read_dir);
    dirs = dirs([dirs.isdir] & ~startsWith({dirs.name}, '.'));
    datasets = {dirs.name};
end

if isempty(datasets)
    error('No datasets found in %s', read_dir);
end

if ~exist(write_dir, 'dir')
    mkdir(write_dir);
end

output_path = fullfile(write_dir, 'ANLSMN_scores.csv');

% -----------------------------
% Load or initialize CSV
% -----------------------------
if isfile(output_path)
    df_existing = readtable(output_path);
else
    df_existing = table('Size', [0 7], ...
                        'VariableTypes', {'string', 'string', 'string', 'double', 'double', 'double', 'datetime'}, ...
                        'VariableNames', {'method', 'parameters', 'dataset', 'Pair', 'score', 'weight', 'timestamp'});
end

method_name = func2str(func);
method_name_lower = lower(method_name);
parameters = serialize_params(varargin);

% -----------------------------
% Main loop over datasets
% -----------------------------
for i = 1:length(datasets)
    dataset = datasets{i};
    fprintf('\n🚀 Running ANLSMN dataset: %s\n', dataset);
    
    dir_ext = fullfile(read_dir, dataset);
    gt_file = fullfile(dir_ext, 'pairs_gt.txt');
    
    if ~isfile(gt_file)
        error('Missing ground truth file: %s', gt_file);
    end
    
    pairs_gt = readmatrix(gt_file);
    n_pairs = length(pairs_gt);
    weight = 1 / n_pairs;
    
    % -----------------------------
    % Loop through each pair
    % -----------------------------
    for pair_idx = 1:n_pairs
        % Check if already exists
        exists = any(lower(string(df_existing.method)) == method_name_lower & ...
                     string(df_existing.dataset) == dataset & ...
                     df_existing.Pair == pair_idx);
        
        if exists && ~overwrite
            fprintf('⏩ Skipping %s Pair %d (already computed)\n', dataset, pair_idx);
            continue;
        end
        
        % Load pair file
        pair_file = fullfile(dir_ext, sprintf('pair_%d.txt', pair_idx));
        if ~isfile(pair_file)
            fprintf('⚠️ Missing pair file: %s, skipping.\n', pair_file);
            continue;
        end
        
        df_pair = readmatrix(pair_file);
        x = df_pair(:, 2);
        y = df_pair(:, 3);
        
        % Correct direction using GT
        if pairs_gt(pair_idx) == 0
            disp("Flipped")
            tmp = x; x = y; y = tmp;
        end
        
        % Run method
        try
            score = func([x, y], varargin{:});
        catch ME
            fprintf('⚠️ Error on %s Pair %d: %s\n', dataset, pair_idx, ME.message);
            score = NaN;
        end
        
        if isnan(score)
            score = NaN;
        end
        
        % Remove previous row if overwrite
        if overwrite
            idx_remove = string(df_existing.method) == method_name_lower & ...
                         string(df_existing.dataset) == dataset & ...
                         df_existing.Pair == pair_idx;
            df_existing(idx_remove, :) = [];
        end
        
        % Append new row
        new_row = {method_name, parameters, dataset, pair_idx, score, weight, string(datetime('now','TimeZone','UTC','InputFormat', 'yyyy-MM-dd HH:mm:ss'))};
        df_existing = [df_existing; new_row];
        
        % Write CSV
        writetable(df_existing, output_path);
        fprintf('✔ Saved %s Pair %d\n', dataset, pair_idx);
    end
end

fprintf('\n✅ Saved ANLSMN results to %s\n', output_path);
end


function json_str = serialize_params(kwargs)
    json_str="";
end