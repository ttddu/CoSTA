function main_CoSTA(override)

% Default benchmark: CEC2022 (F1-F12, D = 10 or 20, search range [-100, 100]^D).
% Main-experiment protocol: population N = 100, MaxFES = 1000*D, 30 independent
% runs per function. The solver is CoSTA.m in this directory.
%
% fobj accepts a 1-by-D row vector and returns a scalar. ConvergenceCurve(k) is
% the best-so-far value after the k-th function evaluation. CEC values include
% the official bias, as in the paper tables. The script also reports f - f*.

% ============================== User configuration ==============================
cfg.suite      = 'CEC2022';   % 'CEC2022' | 'CEC2017' | 'CEC2019'
cfg.func_id    = 1:12;        % function indices; a scalar runs one function
cfg.dim        = 10;          % CEC2022: 10 or 20; CEC2017 commonly uses 30/50/100
cfg.N          = 100;         % population size
cfg.MaxFES     = [];          % empty uses 1000*D; the CEC2019 comparison script used 20000
cfg.nRuns      = 30;          % number of independent runs
cfg.baseSeed   = 2026000;     % run r uses rng(baseSeed + r); [] leaves the seed unfixed
cfg.plotCurve  = true;        % plot the mean error curve across runs
% =====================================================================

if nargin >= 1 && ~isempty(override)
    cfg = apply_override(cfg, override);
end

rootDir = fileparts(mfilename('fullpath'));
if isempty(rootDir)
    rootDir = pwd;
end
prevDir = pwd;
restoreDir = onCleanup(@() cd(prevDir)); 

cd(rootDir);
costaFcn = @CoSTA;
solverInfo = functions(costaFcn);
expectedSolver = fullfile(rootDir, 'CoSTA.m');
if ~strcmpi(solverInfo.file, expectedSolver)
    error('main_CoSTA:WrongSolver', ...
        'Expected to call %s, but resolved to %s.', expectedSolver, solverInfo.file);
end

func_id = cfg.func_id(:)';
nFunc = numel(func_id);
scores = nan(nFunc, cfg.nRuns);
curves = cell(nFunc, 1);
names = cell(nFunc, 1);
fstars = nan(nFunc, 1);
dims = nan(nFunc, 1);

fprintf('CoSTA: %s\n', solverInfo.file);
fprintf('Suite: %s   N = %d   runs = %d\n', cfg.suite, cfg.N, cfg.nRuns);

for k = 1:nFunc
    prob = make_problem(cfg, func_id(k), rootDir);
    prepare_benchmark(prob);
    if isempty(cfg.MaxFES)
        MaxFES = 1000 * prob.dim;
    else
        MaxFES = cfg.MaxFES;
    end
    if MaxFES < cfg.N
        error('main_CoSTA:Budget', 'MaxFES (%d) cannot be smaller than the population size N (%d).', MaxFES, cfg.N);
    end

    runScore = nan(cfg.nRuns, 1);
    runCurve = nan(cfg.nRuns, MaxFES);
    for r = 1:cfg.nRuns
        if ~isempty(cfg.baseSeed)
            rng(cfg.baseSeed + r, 'twister');
        end
        fprintf('%s  D=%d  MaxFES=%d  run %d/%d\n', ...
            prob.name, prob.dim, MaxFES, r, cfg.nRuns);
        [best, ~, curve] = costaFcn(cfg.N, MaxFES, prob.lb, prob.ub, prob.dim, prob.fobj);
        curve = curve(:)';
        if numel(curve) ~= MaxFES
            error('main_CoSTA:CurveLength', ...
                'The convergence curve length should be %d, but it is %d.', MaxFES, numel(curve));
        end
        runCurve(r, :) = curve;
        runScore(r) = curve(end);
        if abs(runScore(r) - best) > 1e-8 * max(1, abs(best))
            runScore(r) = best;
            runCurve(r, end) = best;
        end
    end

    scores(k, :) = runScore;
    curves{k} = runCurve;
    names{k} = prob.name;
    fstars(k) = prob.fstar;
    dims(k) = prob.dim;
    print_summary(prob.name, runScore, prob.fstar);
end

outDir = fullfile(rootDir, 'CoSTA_single_results');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end
tag = result_tag(cfg, dims);
save(fullfile(outDir, ['CoSTA_' tag '.mat']), ...
    'cfg', 'func_id', 'names', 'dims', 'fstars', 'scores', 'curves', 'solverInfo');
write_summary_csv(fullfile(outDir, ['CoSTA_' tag '.csv']), ...
    func_id, names, scores, fstars);
if cfg.plotCurve
    plot_mean_curves(fullfile(outDir, ['CoSTA_' tag '_curve.png']), ...
        names, curves, fstars, tag);
end
fprintf('Results saved to %s\n', outDir);
end

function cfg = apply_override(cfg, override)
names = fieldnames(override);
for i = 1:numel(names)
    cfg.(names{i}) = override.(names{i});
end
end

function prob = make_problem(cfg, func_id, rootDir)
switch lower(strtrim(cfg.suite))
    case 'cec2022'
        if ~ismember(cfg.dim, [10, 20])
            error('main_CoSTA:Dim', 'CEC2022 only provides D = 10 and D = 20.');
        end
        if ~ismember(func_id, 1:12)
            error('main_CoSTA:Func', 'CEC2022 uses function indices 1 to 12.');
        end
        stars = [300, 400, 600, 800, 900, 1800, 2000, 2200, 2300, 2400, 2600, 2700];
        prob.workDir = rootDir;
        prob.dataDir = fullfile(rootDir, 'input_data22');
        prob.mexName = 'cec22_func';
        prob.dim = cfg.dim;
        prob.lb = -100 * ones(1, cfg.dim);
        prob.ub = 100 * ones(1, cfg.dim);
        prob.fstar = stars(func_id);
        prob.name = sprintf('CEC2022-F%d', func_id);
        prob.fobj = @(x) cec22_func(x', func_id);

    case 'cec2017'
        if func_id == 2 || ~ismember(func_id, [1, 3:30])
            error('main_CoSTA:Func', 'CEC2017 uses F1 and F3-F30 (F2 was removed).');
        end
        prob.workDir = fullfile(rootDir, 'CEC2017');
        prob.dataDir = fullfile(prob.workDir, 'input_data');
        prob.mexName = 'cec17_func';
        prob.dim = cfg.dim;
        prob.lb = -100 * ones(1, cfg.dim);
        prob.ub = 100 * ones(1, cfg.dim);
        prob.fstar = 100 * func_id;
        prob.name = sprintf('CEC2017-F%d', func_id);
        prob.fobj = @(x) cec17_func(x', func_id);

    case 'cec2019'
        dims = [9, 16, 18, repmat(10, 1, 7)];
        lbs = [-8192, -16384, -4, repmat(-100, 1, 7)];
        ubs = [8192, 16384, 4, repmat(100, 1, 7)];
        if ~ismember(func_id, 1:10)
            error('main_CoSTA:Func', 'CEC2019 uses function indices 1 to 10.');
        end
        prob.workDir = fullfile(rootDir, 'CEC2019');
        prob.dataDir = fullfile(prob.workDir, 'input_data');
        prob.mexName = 'cec19_func';
        prob.dim = dims(func_id);
        prob.lb = lbs(func_id) * ones(1, prob.dim);
        prob.ub = ubs(func_id) * ones(1, prob.dim);
        prob.fstar = 1;
        prob.name = sprintf('CEC2019-F%d', func_id);
        prob.fobj = @(x) cec19_func(x', func_id);

    otherwise
        error('main_CoSTA:Suite', ...
            'Unknown benchmark suite "%s". Use CEC2022, CEC2017, or CEC2019.', cfg.suite);
end
end

function prepare_benchmark(prob)
if isempty(prob.mexName)
    cd(prob.workDir);
    return;
end
if exist(prob.dataDir, 'dir') ~= 7
    error('main_CoSTA:Data', 'Test data directory not found: %s.', prob.dataDir);
end
% These MEX files read data from the current folder, and MATLAB caches a loaded MEX.
clear(prob.mexName); 
cd(prob.workDir);
if exist(prob.mexName, 'file') ~= 3
    error('main_CoSTA:Mex', ...
        '%s was not found in %s. Compile the corresponding CEC MEX first.', prob.mexName, prob.workDir);
end
end

function print_summary(name, score, fstar)
fprintf('-------------------------------------\n');
fprintf('%s  Best=%.6g  Worst=%.6g  Mean=%.6g  Std=%.6g\n', ...
    name, min(score), max(score), mean(score), std(score));
if isfinite(fstar)
    err = score - fstar;
    fprintf('    f* = %.6g   MeanError=%.6g  BestError=%.6g\n', ...
        fstar, mean(err), min(err));
end
fprintf('-------------------------------------\n');
end

function tag = result_tag(cfg, dims)
dimLabel = sprintf('%g', dims(1));
if any(dims ~= dims(1))
    dimLabel = 'official';
end
tag = sprintf('%s_D%s_N%d_R%d', upper(cfg.suite), dimLabel, cfg.N, cfg.nRuns);
end

function write_summary_csv(path, func_id, names, scores, fstars)
fid = fopen(path, 'w');
if fid < 0
    error('main_CoSTA:File', 'Unable to write %s.', path);
end
cleaner = onCleanup(@() fclose(fid)); 
fprintf(fid, 'Function,Name,Best,Worst,Mean,Std,Fstar,BestError,MeanError\n');
for k = 1:numel(func_id)
    s = scores(k, :);
    errBest = s - fstars(k);
    fprintf(fid, '%d,%s,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g\n', ...
        func_id(k), names{k}, min(s), max(s), mean(s), std(s), fstars(k), ...
        min(errBest), mean(errBest));
end
end

function plot_mean_curves(path, names, curves, fstars, tag)
nFunc = numel(curves);
nCol = min(4, nFunc);
nRow = ceil(nFunc / nCol);
fig = figure('Name', ['CoSTA ' tag], 'Color', 'w', ...
    'Position', [80, 80, 320 * nCol, 240 * nRow]);
for k = 1:nFunc
    subplot(nRow, nCol, k);
    err = mean(curves{k}, 1) - fstars(k);
    err = max(err, 1e-12);
    semilogy(err, 'Color', [0.75, 0.10, 0.10], 'LineWidth', 1.2);
    grid on;
    xlabel('Function evaluations');
    ylabel('Mean error');
    title(names{k}, 'Interpreter', 'none');
end
exportgraphics(fig, path, 'Resolution', 200);
end
