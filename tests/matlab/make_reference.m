% make_reference.m  (step 2 of 2 of fixture generation, see make_reference_inputs.py)
% Runs the ORIGINAL, unmodified MATLAB toolkit (src/external/...) on fixed
% inputs and stores the outputs as reference values for tests/test_prox_crit.py.
% Run with:  matlab -batch "run('tests/matlab/make_reference.m')"

root = fileparts(fileparts(fileparts(mfilename('fullpath'))));  % repository root
addpath(fullfile(root, 'src', 'external', 'proxy_crit_toolkit_sam_sooter_2026_10_09', 'src'));
in = load(fullfile(root, 'tests', 'fixtures', 'prox_crit_inputs.mat'));
fits = {'YuleWalker', 'MaxLikelihood'};

% ---- calc_db on all cases --------------------------------------------------
nc = size(in.cases, 1);
db = nan(nc, 1); sddb = nan(nc, 1); sigma = nan(nc, 1); exit_status = nan(nc, 1);
kern = cell(nc, 1); kernc = cell(nc, 1); H = cell(nc, 1); errmsg = cell(nc, 1);
for i = 1:nc
    c = in.cases(i, :);
    dbopt = struct('fit_method', fits{c(4)+1}, 'with_err_bars', logical(c(5)), ...
                   'with_QC', false, 'with_parallel', false);
    errmsg{i} = '';
    try
        [db(i), sddb(i), kern{i}, sigma(i), H{i}, kernc{i}, exit_status(i)] = ...
            calc_db(in.dsets{c(1)}, c(2), 1, c(3), dbopt);
    catch e
        errmsg{i} = e.message;   % expected for the order < b/2 case
    end
end

% ---- calc_dbv1 ---------------------------------------------------------------
nk = size(in.dbv1_kerns, 1); dbv1 = nan(nk, 1); dbv1_kernc = cell(nk, 1);
for i = 1:nk
    k = in.dbv1_kerns(i, 3:2+in.dbv1_kerns(i, 1));
    [dbv1(i), dbv1_kernc{i}] = calc_dbv1(k, in.dbv1_kerns(i, 2));
end

% ---- spike binning: the demo's get_spike_counts = histc(st, t1:deltaT:t2) ----
S = load(fullfile(root, 'data', 'external', 'senzai2019_YMV15.mat'), 'st', 'WAKE');
st = S.st; deltaT = 0.04;
bins_full = min(st):deltaT:max(st);
counts_full = histc(st, bins_full);
w = double(S.WAKE(2, :));                     % one behavioural-state window
bins_wake = w(1):deltaT:w(2);
counts_wake = histc(st, bins_wake);
t0 = min(st) + 30*7;                          % one sliding window (T = 60 s)
bins_win = t0:deltaT:(t0+60);
counts_win = histc(st, bins_win);

save(fullfile(root, 'tests', 'fixtures', 'prox_crit_reference.mat'), ...
     'db', 'sddb', 'sigma', 'exit_status', 'kern', 'kernc', 'H', 'errmsg', ...
     'dbv1', 'dbv1_kernc', 'bins_full', 'counts_full', 'bins_wake', ...
     'counts_wake', 'bins_win', 'counts_win', '-v7');
disp('reference written');
disp([db sddb exit_status]); disp(errmsg);
