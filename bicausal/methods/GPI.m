function score = GPI(xy_data, varargin)
    % GPI: Wrapper for the Gaussian Process Inference (GPI) method
    % 
    % INPUT:
    %   xy_data:  Nx2 matrix where column 1 is X and column 2 is Y
    %   varargin: Optional 'Name', Value pairs for CFG_XY and CFG_X
    
    current_dir = fileparts(mfilename('fullpath'));
    gpi_dir = fullfile(current_dir, 'source_implementations','GPI', 'gpi');
    addpath(gpi_dir);
    fprintf("New pair!\n");

    max_points = 400;
    N = size(xy_data, 1);
    if N > max_points
        idx = randperm(N, max_points); % randomly select max_points indices
        xy_data = xy_data(idx, :);      % subsample both X and Y simultaneously
        fprintf("Subsampled to %d points.\n", max_points);
    end

    % 1. Extract X and Y
    X = xy_data(:, 1);
    Y = xy_data(:, 2);

    if size(X,2) ~= 1 || size(Y,2) ~= 1
        warning('Either X or Y is not uni-dimensional. Returning NaN.');
        score = NaN;
        return;
    end
    % 2. Parse Parameters
    % We allow the user to pass custom CFG structs via varargin
    % e.g., run_tuebingen(@GPI, ..., 'CFG_XY', my_struct)
    params = struct(varargin{:});
    
    if isfield(params, 'CFG_XY')
        CFG_XY = params.CFG_XY;
    else
        CFG_XY = struct(); % Use defaults in gpi_train
    end
    
    if isfield(params, 'CFG_X')
        CFG_X = params.CFG_X;
    else
        CFG_X = struct(); % Use defaults in mmlgmm
    end

    try
        % 3. Calculate DL for X -> Y (Direction 1)
        % dl1 = p(X) + p(Y|X)
        [dl1, ~, ~] = gpi_mml(X, Y, CFG_XY, CFG_X);

        % 4. Calculate DL for Y -> X (Direction 2)
        % dl2 = p(Y) + p(X|Y)
        [dl2, ~, ~] = gpi_mml(Y, X, CFG_XY, CFG_X);

        % 5. Causal Score
        % A positive score suggests X -> Y (dl2 > dl1)
        % A negative score suggests Y -> X (dl1 > dl2)
        score = dl2 - dl1;

    catch e
        fprintf('Error in GPI calculation: %s\n', e.message);
        score = NaN;
    end

    fprintf("GPI SCORE: %f\n", score);
end