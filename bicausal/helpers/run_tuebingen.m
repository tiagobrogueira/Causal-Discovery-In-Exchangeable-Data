function path = run_tuebingen(func, read_dir, write_dir, overwrite, varargin)
    % run_tuebingen_streaming(func, read_dir, write_dir, overwrite, 'Param1', val1, ...)

    if nargin < 2 || isempty(read_dir),  read_dir = 'bicausal/benchmarks/Tuebingen'; end
    if nargin < 3 || isempty(write_dir), write_dir = 'bicausal/results'; end
    if nargin < 4, overwrite = false; end

    % 1. Load Data
    [data, weights] = getTuebingen(read_dir);
    if isempty(data)
        fprintf('❌ No data found.\n');
        return;
    end

    % 2. Setup Paths and Metadata
    if ~exist(write_dir, 'dir'), mkdir(write_dir); end
    path = fullfile(write_dir, 'tuebingen_scores.csv');

    method_name = func2str(func);
    method_name_lower = lower(method_name);
    parameters = serialize_params(varargin);

    % 3. Manage Existing Results
    if isfile(path)
        df_existing = readtable(path, 'TextType', 'string');
        df_existing.parameters(ismissing(df_existing.parameters)) = "";
        df_existing.method = lower(df_existing.method);
    else
        df_existing = table([], [], [], [], [], [], ...
            'VariableNames', {'method', 'parameters', 'Pair', 'score', 'weight', 'timestamp'});
    end

    % 4. Main Processing Loop
    for i = 1:length(data)
        x = data{i}{1};
        y = data{i}{2};
        w = weights(i);

        % Check for existing entry
        exists = any(df_existing.method == method_name_lower & df_existing.Pair == i);

        if exists && ~overwrite
            fprintf('⏩ Skipping Pair %d for %s (already computed)\n', i, method_name);
            continue;
        end

        try
            xy_data = [x, y];
            score = func(xy_data, varargin{:});

            if ~isempty(score)
                res_row = table(string(method_name), string(parameters), i, score, w, ...
                    string(datetime('now', 'TimeZone', 'UTC')), ...
                    'VariableNames', {'method', 'parameters', 'Pair', 'score', 'weight', 'timestamp'});

                % If file exists, append; otherwise, write with headers
                if isfile(path) && overwrite == false
                    writetable(res_row, path, 'WriteMode', 'append');
                else
                    writetable(res_row, path); % overwrite or create new
                end

                % Update df_existing to prevent duplicates in the same run
                df_existing = [df_existing; res_row];
            end
        catch e
            fprintf('⚠️ Skipping Pair %d due to error: %s\n', i, e.message);
        end
    end

    fprintf('✅ Saved Tuebingen results to %s\n', path);
end

% --- Helper Functions ---

function json_str = serialize_params(kwargs)
    json_str="";
end

function [data_list, weights] = getTuebingen(read_dir)
    pairmeta_file = fullfile(read_dir, 'pairmeta.txt');
    if ~exist(pairmeta_file, 'file')
        error('❌ pairmeta.txt not found in %s', read_dir);
    end

    % Read meta file
    opts = detectImportOptions(pairmeta_file, 'FileType', 'text');
    meta = readmatrix(pairmeta_file); % Assumes space-separated numeric data
    
    data_list = {};
    weights = [];

    for i = 1:size(meta, 1)
        pair_idx = meta(i, 1);
        x_start = meta(i, 2); % MATLAB is 1-based, R logic handled -1+1
        x_end   = meta(i, 3);
        y_start = meta(i, 4);
        y_end   = meta(i, 5);
        w       = meta(i, 6);

        pair_filename = fullfile(read_dir, sprintf('pair%04d.txt', pair_idx));

        if ~exist(pair_filename, 'file')
            fprintf('⚠️ Missing %s, skipping.\n', pair_filename);
            continue;
        end

        try
            arr = readmatrix(pair_filename);
            % Subsetting columns
            x = arr(:, x_start:x_end);
            y = arr(:, y_start:y_end);
            
            data_list{end+1} = {x, y}; %#ok<AGROW>
            weights(end+1) = w; %#ok<AGROW>
        catch e
            fprintf('⚠️ Error reading %s: %s\n', pair_filename, e.message);
        end
    end
end