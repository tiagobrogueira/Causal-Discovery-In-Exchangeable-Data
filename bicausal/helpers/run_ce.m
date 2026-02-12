function output_path = run_ce(func, datasets, read_dir, write_dir, overwrite, varargin)
% RUN_CE  Run causal evaluation on CE synthetic datasets
%
% output_path = run_ce(func, datasets, read_dir, write_dir, overwrite, ...)
%
% INPUT
%   func        Function handle accepting xy_data (Nx2 matrix)
%   datasets    Cell array of dataset names
%   read_dir    Directory containing CE CSV files
%   write_dir   Directory to store results
%   overwrite   Logical flag
%   varargin    Additional parameters passed to func
%
% OUTPUT
%   output_path Path to results CSV
    
    project_root = fileparts(mfilename('fullpath'));
    if nargin < 2 || isempty(datasets)
        datasets = {};
    end
    if nargin < 3 || isempty(read_dir)
        read_dir = fullfile(project_root,"..", "benchmarks", "synthetic", "CE-Guyon");
    end
    if nargin < 4 || isempty(write_dir)
        write_dir = fullfile(project_root, "..", "results");
    end
    if nargin < 5 || isempty(overwrite)
        overwrite = false;
    end

    % -----------------------------
    % CE dataset list
    % -----------------------------
    ALL_CE = {"CE-Cha","CE-Gauss","CE-Multi","CE-Net"};

    if isempty(datasets)
        datasets = ALL_CE;
    else
        invalid = setdiff(datasets, ALL_CE);
        if ~isempty(invalid)
            error("Invalid CE dataset(s): %s. Must be one of: %s", ...
                strjoin(invalid, ", "), strjoin(ALL_CE, ", "));
        end
    end

    % -----------------------------
    % Prepare output file
    % -----------------------------
    if ~exist(write_dir, 'dir')
        mkdir(write_dir);
    end

    output_path = fullfile(write_dir, "CE_scores.csv");

    method_name = func2str(func);
    method_name_lower = lower(method_name);

    % You can customize this if you already have serialize_params
    parameters = serialize_params(varargin);

    % -----------------------------
    % Load or initialize CSV
    % -----------------------------
    if isfile(output_path)
        df_existing = readtable(output_path, 'TextType','string');
        df_existing.method = lower(df_existing.method);
    else
        df_existing = table( ...
            strings(0,1), strings(0,1), strings(0,1), ...
            zeros(0,1), zeros(0,1), zeros(0,1), strings(0,1), ...
            'VariableNames', {'method','parameters','dataset','Pair','score','weight','timestamp'});
    end

    % =====================================================
    % Main loop
    % =====================================================
    for d = 1:length(datasets)

        dataset = datasets{d};
        fprintf('\n🚀 Running CE dataset: %s\n', dataset);

        pairs_file   = fullfile(read_dir, dataset + "_pairs.csv");
        targets_file = fullfile(read_dir, dataset + "_targets.csv");

        abs_path = fullfile(pwd, pairs_file);
        %fprintf("Trying to load (absolute path):\n%s\n\n", abs_path);

        if ~isfile(pairs_file)
            error("Pairs file not found: %s", pairs_file);
        end
        if ~isfile(targets_file)
            error("Targets file not found: %s", targets_file);
        end

        df_pairs = readtable(pairs_file, ...
                     'TextType','string',...
                     'Delimiter', ',');

        df_targets = readtable(targets_file, ...
                     'TextType','string',...
                     'Delimiter', ',');

        if height(df_pairs) ~= height(df_targets)
            error("Row count mismatch: %d pairs vs %d targets", ...
                height(df_pairs), height(df_targets));
        end

        weight = 1 / height(df_pairs);

        % -----------------------------
        % Loop over pairs
        % -----------------------------
        for idx = 1:height(df_pairs)

            x_str = string(df_pairs{idx,2});
            y_str = string(df_pairs{idx,3});

            x = str2double(regexp(x_str, '\s+', 'split'));
            y = str2double(regexp(y_str, '\s+', 'split'));
            x = x(:);
            y = y(:);

            if df_targets{idx,2} == -1
                tmp = x;
                x = y;
                y = tmp;
            end

            pair_idx = idx;

            class(df_existing.method)
            class(method_name_lower)

            % -----------------------------------------
            % Check existing
            % -----------------------------------------
            exists = any( ...
                lower(df_existing.method) == method_name_lower & ...
                df_existing.dataset == dataset & ...
                df_existing.Pair == pair_idx);

            if exists && ~overwrite
                fprintf('⏩ Skipping %s Pair %d (already computed)\n', dataset, pair_idx);
                continue;
            end

            % -----------------------------------------
            % Compute score
            % -----------------------------------------
            try
                xy_data = [x y];
                score = func(xy_data, varargin{:});
            catch ME
                fprintf('⚠️ Error in %s on %s Pair %d: %s\n', ...
                    method_name, dataset, pair_idx, ME.message);
                continue;
            end

            if isnan(score)
                score = NaN;
            end

            % -----------------------------------------
            % Overwrite if needed
            % -----------------------------------------
            if overwrite
                mask = lower(df_existing.method) == method_name_lower & ...
                       df_existing.dataset == dataset & ...
                       df_existing.Pair == pair_idx;
                df_existing(mask,:) = [];
            end

            % -----------------------------------------
            % Append row
            % -----------------------------------------
            new_row = { ...
                method_name, ...
                parameters, ...
                dataset, ...
                pair_idx, ...
                score, ...
                weight, ...
                string(datetime('now','TimeZone','UTC'))};

            df_existing = [df_existing; new_row];

            writetable(df_existing, output_path);

            fprintf('✔ Saved %s Pair %d\n', dataset, pair_idx);
        end
    end

    fprintf('\n✅ Saved CE results to %s\n', output_path);
end

function json_str = serialize_params(kwargs)
    json_str="";
end 

