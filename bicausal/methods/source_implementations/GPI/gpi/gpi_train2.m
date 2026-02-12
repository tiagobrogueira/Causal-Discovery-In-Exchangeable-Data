function [DL,INFO] = gpi_train2(X,Y,CFG)
% GPI_TRAIN: Gaussian Process Inference training using MATLAB optimizer
% Calculates GPI cost function in the direction X -> Y
% Uses fminunc from MATLAB Optimization Toolbox

  if nargin < 3
    CFG = struct;
  end

  % --- set default CFG fields ---
  if ~isfield(CFG,'Ncg'), CFG.Ncg = 10000; end
  if ~isfield(CFG,'gradcheck'), CFG.gradcheck = false; end
  if ~isfield(CFG,'uniform'), CFG.uniform = false; end
  if ~isfield(CFG,'epslabs'), CFG.epslabs = 1e-3; end
  if ~isfield(CFG,'profile'), CFG.profile = false; end
  if ~isfield(CFG,'cov'), CFG.cov = log([10;10;100]); end
  if ~isfield(CFG,'cov_f'), CFG.cov_f = {@gpi_kernel_seard}; end
  if ~isfield(CFG,'initE'), CFG.initE = true; end

  INFO = struct();

  % --- normalize data ---
  if CFG.uniform
    X = normalize(X,1);
    Y = normalize(Y,1);
  else
    X = normalize(X,2);
    Y = normalize(Y,2);
  end
  if isfield(CFG,'scaleX'), X = X * CFG.scaleX; end
  if isfield(CFG,'scaleY'), Y = Y * CFG.scaleY; end

  N = size(X,1);

  % --- fit standard GP X->Y for initialization of E ---
  hyp0GP.mean = [];
  hyp0GP.cov = log([0.4;1]);
  hyp0GP.lik = log(0.2);

  % optimize standard GP using gp function (existing)
  hypGP = gp_minimize(hyp0GP, X, Y, CFG.Ncg);

  [ymu, ~] = gp(hypGP,'infExact','meanZero','covSEiso','likGauss',X,Y,X);
  E0 = (Y - ymu) / exp(hypGP.lik);

  INFO.GP.lml = gp(hypGP,'infExact','meanZero','covSEiso','likGauss',X,Y);
  INFO.GP.hyp = hypGP;
  INFO.GP.E   = E0;

  % --- initialize hyperparameters for GPI ---
  hyp0 = struct();
  hyp0.cov = CFG.cov;
  if CFG.initE
    hyp0.e = E0 / std(E0);
  else
    hyp0.e = randn(size(E0));
  end

  % --- gradient check if requested ---
  if CFG.gradcheck
    success = gradcheck(hyp0,'gpi_objfun',CFG,CFG.cov_f,X,Y);
    if ~success
      error('gradcheck failed');
    end
  end

  if CFG.profile
    profile on
  end

  % --- compute initial cost and gradient ---
  [INFO.a0, INFO.b0] = gpi_objfun(hyp0, CFG, CFG.cov_f, X, Y);

  % --- flatten hyperparameters ---
  hyp_vec0 = [hyp0.cov; hyp0.e];

  % --- fminunc options ---
  options = optimoptions('fminunc', ...
      'Algorithm','quasi-newton', ...
      'MaxIterations', CFG.Ncg, ...
      'SpecifyObjectiveGradient', true, ...
      'Display','off');

  % --- wrapper function for fminunc ---
  fun = @(hyp_vec) gpi_objfun_wrapper(hyp_vec, CFG, CFG.cov_f, X, Y);

  % --- run optimizer ---
  [hyp_vec_opt, fval] = fminunc(fun, hyp_vec0, options);

  % --- unpack optimized hyperparameters ---
  hyp = struct();
  ncov = length(hyp0.cov);
  hyp.cov = hyp_vec_opt(1:ncov);
  hyp.e   = hyp_vec_opt(ncov+1:end);

  % --- final cost and gradient ---
  [INFO.a1, INFO.b1] = gpi_objfun(hyp, CFG, CFG.cov_f, X, Y);

  if CFG.profile
    profile off
  end

  % --- store results ---
  INFO.X = X;
  INFO.Y = Y;

  cost = gpi_objfun(hyp, CFG, CFG.cov_f, X, Y, true);
  INFO.cost = cost;
  INFO.hyp  = hyp;

  %Edited out - they serve as explanation!
  %INFO.dfe = gpi_predict(hyp, CFG, CFG.cov_f, X, Y, X, hyp.e, 1);
  %INFO.CFG = CFG;

  INFO.DL = cost.GP + cost.IT + cost.E + sum(cost.prior);
  DL = INFO.DL;

end

%% --- fminunc wrapper ---
function [f, g] = gpi_objfun_wrapper(hyp_vec, CFG, cov_f, X, Y)
    ncov = length(CFG.cov);
    hyp = struct();
    hyp.cov = hyp_vec(1:ncov);
    hyp.e   = hyp_vec(ncov+1:end);

    [f, dlml] = gpi_objfun(hyp, CFG, cov_f, X, Y);
    g = [dlml.cov; dlml.e];
end

%% --- simple GP minimizer ---
function hypGP = gp_minimize(hyp0GP, X, Y, Ncg)
    % wrapper around gp function to optimize standard GP hyperparameters
    hypGP = hyp0GP;
    % NOTE: you can implement a simple loop or use fminunc as above
    hyp_vec0 = [hyp0GP.cov; hyp0GP.lik];
    options = optimoptions('fminunc', ...
        'Algorithm','quasi-newton', ...
        'MaxIterations', Ncg, ...
        'SpecifyObjectiveGradient', true, ...
        'Display','off');
    fun = @(hv) gp_objfun_wrapper(hv, X, Y);
    [hv_opt, ~] = fminunc(fun, hyp_vec0, options);
    hypGP.cov = hv_opt(1:2);
    hypGP.lik = hv_opt(3);
end

function [f, g] = gp_objfun_wrapper(hyp_vec, X, Y)
    hyp = struct();
    hyp.cov = hyp_vec(1:2);
    hyp.lik = hyp_vec(3);
    [f, dlZ] = gp(hyp,'infExact','meanZero','covSEiso','likGauss',X,Y);
    g = [dlZ.cov; dlZ.lik];
end
