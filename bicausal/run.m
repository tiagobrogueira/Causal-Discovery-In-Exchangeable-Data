% bicausal/run.m
% -------------------------------------------------------------------------
% Entry point for running methods on the Tuebingen benchmark
% Calls GPI with default settings   
% -------------------------------------------------------------------------

clc;
clear;

% -------------------------------------------------------------------------
% Ensure all subfolders (methods, helpers, benchmarks, etc.) are on path
% -------------------------------------------------------------------------
project_root = fileparts(mfilename('fullpath'));
addpath(genpath(project_root));

% -------------------------------------------------------------------------
% Default paths
% -------------------------------------------------------------------------
read_dir  = fullfile(project_root, 'benchmarks', 'Tuebingen');
write_dir = fullfile(project_root, 'results');
overwrite = false;   % set to true if you want to recompute existing scores

% -------------------------------------------------------------------------
% Run GPI with default internal settings
% (No CFG_XY or CFG_X passed -> GPI uses its own defaults)
% -------------------------------------------------------------------------
fprintf('🚀 Running GPI on Tuebingen benchmark...\n');
%run_tuebingen(@GPI_lx, read_dir, write_dir, overwrite);
%run_anlsmn(@GPI);
%run_sim(@GPI);
run_ce(@GPI);

fprintf('✅ Done.\n');
