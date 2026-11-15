% Title: robustness_tests_pack4_UHRS.m
% Author: N. D. Vilanakis
%
% A progressive multi-output 1-D convolutional network for lower-frequency
% trace reconstruction of UHRS seismic data
% N. D. Vilanakis, N. Economou, A. Vafidis
% Computers and Geosciences
%
% User guide
%   Purpose : Robustness comparison across the models of this sweep (Section 3.3, loss weighting sweep).
%             Varied quantity: per-stage loss weighting scheme.
%   Inputs  : the trained networks of this pack and
%             dataset_train_val_test_UHRS.mat. Paths are set at the top.
%   Outputs : a results .mat file and a readable summary .txt for the pack.

%% robustness_tests_pack4_UHRS.m
%
% Ablation Pack 4 (UHRS dataset) — Robustness: Loss Weighting Scheme Sweep
% Variants: (1) geoAdaptive — original progressive model (reference)
%           (2) uniform     — equal weights across all stages
%           (3) geometric   — geometrically increasing weights
%           (4) adaptive    — adaptive (loss-ratio) weights
%
% UHRS wavelet: Gaussian-windowed cosine.
%   fHF=6400 Hz, wavle=78 samp, alphaval=20, nw_buf=1023, dt=1e-5 s
%   sizew(f) = floor(6400/f * 78),  alphaw(f) = 20 * sizew(f) / 78
%
% Outputs:
%   robustness_pack4_UHRS_results.mat
%   robustness_pack4_UHRS_summary.txt
%
% Run from: Ablation\Pack4_Loss_Weighting\

clear; clc; close all;

%% -------------------- Variants --------------------
% All .mat files are in the same folder as this script.
dataPath = 'dataset_train_val_test_UHRS.mat';

variants = { ...
    'CNN_progressive_multioutput_UHRS.mat',    'geoAdaptive (ref)'; ...
    'CNN_progressive_lw_uniform_UHRS.mat',     'uniform'; ...
    'CNN_progressive_lw_geometric_UHRS.mat',   'geometric'; ...
    'CNN_progressive_lw_adaptive_UHRS.mat',    'adaptive' ...
};
nVariants = size(variants, 1);

%% -------------------- Constants --------------------
fHF           = 6400;
miniBatchSize = 512;
maxN          = 4500;
snr_list      = [40 30 20 10 5 0];

%% -------------------- Pre-load dataset once --------------------
fprintf('Loading metadata from: %s\n', variants{1,1});
M0 = load(variants{1,1}, 'nStages','targetVarIdx','t','dt','targetHz');
nStages0      = M0.nStages;
targetVarIdx0 = M0.targetVarIdx;
targetHz0     = M0.targetHz;
t0            = M0.t(:);
dt0           = mean(diff(t0));
nt0           = numel(t0);

fprintf('Loading test data from: %s\n', dataPath);
vars = {'X_test'};
for s = 1:nStages0
    vars{end+1} = sprintf('Y_test%d', targetVarIdx0(s));   %#ok<SAGROW>
end
D      = load(dataPath, vars{:});
X_full = D.X_test;
Ntest  = size(X_full, 2);
Y_full = zeros(nt0, Ntest, nStages0);
for s = 1:nStages0
    fname = sprintf('Y_test%d', targetVarIdx0(s));
    Y_full(:,:,s) = D.(fname);
end
clear D;

N      = min(Ntest, maxN);
X_base = X_full(:, 1:N);
Y_base = Y_full(:, 1:N, :);
clear X_full Y_full;
rms_Xbase = sqrt(mean(X_base(:).^2));
fprintf('Using %d test traces (of %d available).\n', N, Ntest);

%% -------------------- Pre-compute UHRS wavelets --------------------
wUHRS_LF = cell(nStages0, 1);
for s = 1:nStages0
    wUHRS_LF{s} = uhrs_wavelet(targetHz0(s), dt0, 1.0);
end
wHF_UHRS = uhrs_wavelet(fHF, dt0, 1.0);

%% -------------------- Pre-compute perturbations (shared across variants) ----
% A1: LF freq shift +15%
Y_A1 = Y_base;
for s = 1:nStages0
    wNew = uhrs_wavelet(targetHz0(s) * 1.15, dt0, 1.0);
    Y_A1(:,:,s) = wavelet_reshape(Y_base(:,:,s), wUHRS_LF{s}, wNew, nt0);
end

% A2: LF phase rotation +45 deg
Y_A2 = Y_base;
for s = 1:nStages0
    Y_A2(:,:,s) = phase_rotate_matrix(Y_base(:,:,s), 45);
end

% A3 (UHRS adaptation): looser Gaussian window (alpha * 0.5 = wider envelope)
Y_A3 = Y_base;
for s = 1:nStages0
    wWide = uhrs_wavelet(targetHz0(s), dt0, 0.5);
    Y_A3(:,:,s) = wavelet_reshape(Y_base(:,:,s), wUHRS_LF{s}, wWide, nt0);
end

% A4: LF Hilbert phase (+90 deg)
Y_A4 = Y_base;
for s = 1:nStages0
    Y_A4(:,:,s) = imag(hilbert(Y_base(:,:,s)));
end

% A5: LF minimum-phase equivalent
Y_A5 = Y_base;
for s = 1:nStages0
    wMin = minphase_wavelet(wUHRS_LF{s});
    Y_A5(:,:,s) = wavelet_reshape(Y_base(:,:,s), wUHRS_LF{s}, wMin, nt0);
end

% A6: LF wavelet truncated to central 50% of active samples
Y_A6 = Y_base;
for s = 1:nStages0
    w0      = wUHRS_LF{s};
    nActive = find(w0 ~= 0, 1, 'last');   % UHRS wavelet is zero-padded
    L       = round(0.25 * nActive);
    mid     = round(nActive / 2);
    wTr     = zeros(numel(w0), 1);
    wTr(mid - L : mid + L) = w0(mid - L : mid + L);
    Y_A6(:,:,s) = wavelet_reshape(Y_base(:,:,s), wUHRS_LF{s}, wTr, nt0);
end

% A7: LF Klauder wavelet (vibroseis-style)
Y_A7 = Y_base;
for s = 1:nStages0
    wKl = klauder_wavelet(targetHz0(s), targetHz0(s) * 0.5, dt0, ...
                          uhrs_active_length(targetHz0(s)) * dt0);
    Y_A7(:,:,s) = wavelet_reshape(Y_base(:,:,s), wUHRS_LF{s}, wKl, nt0);
end

% A8: HF dominant frequency shift 6400 -> 5500 Hz
wHF_5k = uhrs_wavelet(5500, dt0, 1.0);
X_A8   = wavelet_reshape(X_base, wHF_UHRS, wHF_5k, nt0);

% A9: HF phase rotation +45 deg
X_A9 = phase_rotate_matrix(X_base, 45);

% C1: OOD spiky reflectivity (k=22,24,26 — above k_max=20), amplitude-matched
cfgC1 = default_cfg_UHRS(nt0, dt0, fHF, targetHz0, nStages0);
cfgC1.k_values  = [22 24 26];   % genuinely OOD: above k_max=20
cfgC1.g_values  = 6001:6600;
cfgC1.maxTraces = maxN;
[XHF_C1, YLF_C1] = make_dataset_UHRS(cfgC1);
sc1    = rms_Xbase / (sqrt(mean(XHF_C1(:).^2)) + 1e-12);
XHF_C1 = XHF_C1 * sc1;   YLF_C1 = YLF_C1 * sc1;

% C3: OOD colored reflectivity, amplitude-matched
cfgC3 = default_cfg_UHRS(nt0, dt0, fHF, targetHz0, nStages0);
cfgC3.k_values            = 10:-2:2;
cfgC3.g_values            = 8001:8600;
cfgC3.maxTraces           = maxN;
cfgC3.refl_mode           = "colored";
cfgC3.color_sigma_samples = 5;
[XHF_C3, YLF_C3] = make_dataset_UHRS(cfgC3);
sc3    = rms_Xbase / (sqrt(mean(XHF_C3(:).^2)) + 1e-12);
XHF_C3 = XHF_C3 * sc3;   YLF_C3 = YLF_C3 * sc3;

%% -------------------- GPU --------------------
useGPU = canUseGPU();
if useGPU
    try
        g = gpuDevice();
        fprintf('GPU: %s (%s)\n', g.Name, g.ComputeCapability);
    catch ME
        fprintf('GPU reported available but failed: %s\n', ME.message);
        useGPU = false;
    end
end

%% -------------------- Open summary file --------------------
txtFile = 'robustness_pack4_UHRS_summary.txt';
fid = fopen(txtFile, 'w');
assert(fid > 0, 'Could not open %s for writing.', txtFile);
fprintf(fid, 'ROBUSTNESS SUMMARY — Ablation Pack 4 (UHRS): Loss Weighting Scheme Sweep\n');
fprintf(fid, 'Date: %s\n', datestr(now));   %#ok<TNOW1,DATST>
fprintf(fid, 'Variants: geoAdaptive (ref) | uniform | geometric | adaptive\n');
fprintf(fid, 'Wavelet: UHRS (Gaussian-windowed cosine, fHF=%d, wavle=78, alpha=20)\n', fHF);
fprintf(fid, 'Dataset: %s  (N=%d traces)\n', dataPath, N);
fprintf(fid, 'Target Hz: %s\n\n', sprintf('%d ', targetHz0));

%% -------------------- Main variant loop --------------------
allVariantResults = cell(nVariants, 1);

for vi = 1:nVariants
    label   = variants{vi, 2};
    matFile = variants{vi, 1};
    fprintf('\n========== Variant: %s ==========\n', label);

    M = load(matFile, ...
        'encNet','headNets','quotaNets', ...
        'muX','sigX','muY','sigY', ...
        'targetHz','targetVarIdx','nStages','t','dt');
    encNet    = M.encNet;
    headNets  = M.headNets;
    quotaNets = M.quotaNets;
    muX = M.muX;  sigX = M.sigX;
    muY = M.muY;  sigY = M.sigY;
    targetHz  = M.targetHz;
    nStages   = M.nStages;

    % Baseline (first element — initialises varResults)
    m = eval_net(encNet, headNets, quotaNets, X_base, Y_base, ...
        muX, sigX, muY, sigY, targetHz, nStages, miniBatchSize, useGPU);
    m.TestName = "Baseline";  m.Section = "Baseline";  m.Variant = label;
    varResults = m;
    print_m(m);

    % A tests
    Atests = { ...
        X_base, Y_A1, "A1: LF UHRS freq shift +15%"; ...
        X_base, Y_A2, "A2: LF phase rotation +45 deg"; ...
        X_base, Y_A3, "A3: LF wider Gaussian window (alpha*0.5)"; ...
        X_base, Y_A4, "A4: LF Hilbert phase (+90 deg)"; ...
        X_base, Y_A5, "A5: LF minimum-phase equivalent"; ...
        X_base, Y_A6, "A6: LF truncated to 50% active length"; ...
        X_base, Y_A7, "A7: LF Klauder (bw=50%*f0)"; ...
        X_A8,   Y_base, "A8: HF UHRS freq shift 6400->5500 Hz"; ...
        X_A9,   Y_base, "A9: HF phase rotation +45 deg" ...
    };
    for ti = 1:size(Atests, 1)
        m = eval_net(encNet, headNets, quotaNets, Atests{ti,1}, Atests{ti,2}, ...
            muX, sigX, muY, sigY, targetHz, nStages, miniBatchSize, useGPU);
        m.TestName = Atests{ti,3};  m.Section = "A";  m.Variant = label;
        varResults(end+1) = m;  %#ok<AGROW>
        print_m(m);
    end

    % B: SNR sweep
    for ii = 1:numel(snr_list)
        X_noisy = add_noise_snr(X_base, snr_list(ii));
        m = eval_net(encNet, headNets, quotaNets, X_noisy, Y_base, ...
            muX, sigX, muY, sigY, targetHz, nStages, miniBatchSize, useGPU);
        m.TestName = sprintf('B: HF noise SNR=%g dB', snr_list(ii));
        m.Section  = "B";  m.Variant = label;
        varResults(end+1) = m;  %#ok<AGROW>
        print_m(m);
    end

    % C: OOD
    Ctests = { ...
        XHF_C1, YLF_C1, "C1: OOD spikiness k={22,24,26} (above k_max=20)"; ...
        0.3*X_base, 0.3*Y_base, "C2: amplitude amp=0.3"; ...
        XHF_C3, YLF_C3, "C3: colored reflectivity (sigma=5 samp)"; ...
        2.0*X_base, 2.0*Y_base, "C4: amplitude amp=2.0" ...
    };
    for ti = 1:size(Ctests, 1)
        m = eval_net(encNet, headNets, quotaNets, Ctests{ti,1}, Ctests{ti,2}, ...
            muX, sigX, muY, sigY, targetHz, nStages, miniBatchSize, useGPU);
        m.TestName = Ctests{ti,3};  m.Section = "C";  m.Variant = label;
        varResults(end+1) = m;  %#ok<AGROW>
        print_m(m);
    end

    % D: Combined
    Dtests = { ...
        add_noise_snr(X_base,10), Y_A2, "D1: SNR=10 dB + LF phase +45 deg"; ...
        add_noise_snr(X_base,20), Y_A1, "D2: SNR=20 dB + LF UHRS freq +15%"; ...
        add_noise_snr(XHF_C1,20), YLF_C1, "D3: OOD k={22,24,26} + SNR=20 dB" ...
    };
    for ti = 1:size(Dtests, 1)
        m = eval_net(encNet, headNets, quotaNets, Dtests{ti,1}, Dtests{ti,2}, ...
            muX, sigX, muY, sigY, targetHz, nStages, miniBatchSize, useGPU);
        m.TestName = Dtests{ti,3};  m.Section = "D";  m.Variant = label;
        varResults(end+1) = m;  %#ok<AGROW>
        print_m(m);
    end

    allVariantResults{vi} = varResults;

    % Write detail block
    fprintf(fid, '\n%s\n', repmat('=', 1, 80));
    fprintf(fid, 'VARIANT: %s\n', label);
    fprintf(fid, '%s\n', repmat('=', 1, 80));
    for ii = 1:numel(varResults)
        write_block(fid, varResults(ii), targetHz);
    end
end

%% -------------------- Save --------------------
save('robustness_pack4_UHRS_results.mat', ...
    'allVariantResults', 'variants', 'snr_list', ...
    'fHF', 'targetHz0', 'nStages0', 'dt0', 'nt0', '-v7.3');
fprintf('\nSaved: robustness_pack4_UHRS_results.mat\n');

%% -------------------- Aggregate table --------------------
fprintf(fid, '\n\n%s\n', repmat('=', 1, 180));
fprintf(fid, 'AGGREGATE TABLE — MeanRMSE per test per variant\n');
fprintf(fid, '  DELTA columns: variant - geoAdaptive(ref)\n');
fprintf(fid, '%s\n', repmat('=', 1, 180));

refResults = allVariantResults{1};
testNames  = {refResults.TestName};
nTests     = numel(testNames);

fprintf(fid, '%-54s | %-16s', 'Test', variants{1,2});
for vi = 2:nVariants
    fprintf(fid, ' | %-12s | %-10s', variants{vi,2}, 'DELTA');
end
fprintf(fid, '\n%s\n', repmat('-', 1, 54 + 18 + (nVariants-1)*26));

for ti = 1:nTests
    fprintf(fid, '%-54s', testNames{ti});
    ref_rmse = NaN;
    vr_ref   = allVariantResults{1};
    idx_ref  = find(strcmp({vr_ref.TestName}, testNames{ti}), 1);
    if ~isempty(idx_ref)
        ref_rmse = vr_ref(idx_ref).MeanRMSE;
        fprintf(fid, ' | %16.4g', ref_rmse);
    else
        fprintf(fid, ' | %16s', 'N/A');
    end
    for vi = 2:nVariants
        vr  = allVariantResults{vi};
        idx = find(strcmp({vr.TestName}, testNames{ti}), 1);
        if isempty(idx)
            fprintf(fid, ' | %12s | %10s', 'N/A', 'N/A');
        else
            val   = vr(idx).MeanRMSE;
            delta = val - ref_rmse;
            fprintf(fid, ' | %12.4g | %+10.4g', val, delta);
        end
    end
    fprintf(fid, '\n');
end
fprintf(fid, '%s\n', repmat('=', 1, 180));
fclose(fid);
fprintf('Saved: %s\n', txtFile);
disp('Done.');

%% ========================== LOCAL FUNCTIONS ==========================

% ----- UHRS wavelet -----
function w = uhrs_wavelet(f, dt, alpha_scale)
% Symmetric UHRS (Gaussian-windowed cosine) wavelet matching
% reflectivity_generator_UHRS_symmetric.m exactly:
%   custom gausswin_local, pedestal subtraction, even-length window.
%   fHF=6400, wavle=78, alphaval=20, nw_buf=1023
% alpha_scale: multiplier on Gaussian width hwidth (default 1.0), used by A3.
if nargin < 3, alpha_scale = 1.0; end
fHF_ref  = 6400;
wavle    = 78;
alphaval = 20;
nw_buf   = 1023;
sizew_HF = floor(fHF_ref / fHF_ref * wavle);   % = 78
sizew    = floor(fHF_ref / f  * wavle);
alphaw   = alpha_scale * alphaval * (sizew / sizew_HF);   % alpha from floored length
sizew    = sizew + mod(sizew, 2) + 2;                     % force even length (newUHRS symmetric)
t1       = (0:sizew-1)' * dt;
cosu     = -cos(2 * pi * f * t1);
gwin     = gausswin_local(sizew, alphaw);
gwin     = gwin - min(gwin);                              % pedestal subtraction: zero at edges
wavtemp  = gwin .* cosu;
w        = zeros(nw_buf, 1);
w(1:sizew) = wavtemp;
end

% ----- Custom Gaussian window (matches newUHRS generator; NOT MATLAB built-in) -----
function gw = gausswin_local(n, hwidth)
% gw = exp(-((x - n/2)/hwidth)^2), x = 1..n
no = n/2;
x  = 1:n;
gw = exp( -((x - no)/hwidth).^2 )';
end

% ----- Active length for Klauder T parameter -----
function L = uhrs_active_length(f)
fHF_ref = 6400;  wavle = 78;
L = floor(fHF_ref / f * wavle);
end

% ----- Wavelet reshape -----
function Y_new = wavelet_reshape(Y, w_from, w_to, nt)
Yf      = fft(Y, nt, 1);
Wf      = fft(w_from(:), nt);
Wt      = fft(w_to(:),   nt);
eps_reg = 0.01 * max(abs(Wf).^2);
ratio   = (Wt .* conj(Wf)) ./ (abs(Wf).^2 + eps_reg);
Y_new   = real(ifft(Yf .* ratio, nt, 1));
end

function Y_rot = phase_rotate_matrix(Y, phi_deg)
phi   = deg2rad(phi_deg);
Y_rot = real(ifft(fft(Y, [], 1) * exp(1i * phi), [], 1));
end

function X_noisy = add_noise_snr(X, snr_db)
Px      = mean(X.^2, 1);
Pn      = Px / 10^(snr_db / 10);
X_noisy = X + randn(size(X)) .* sqrt(Pn);
end

function wmin = minphase_wavelet(w)
W    = fft(w);
A    = abs(W) + eps;
phi  = imag(hilbert(log(A)));
wmin = real(ifft(exp(log(A) + 1i * phi)));
end

function w = klauder_wavelet(f0, bw, dt, T)
t = (-T/2 : dt : T/2)';
k = bw / T;
w = cos(2*pi*(f0*t + 0.5*k*t.^2)) .* hann(length(t));
w = w / max(abs(w));
end

% ----- OOD dataset (UHRS wavelets) -----
function cfg = default_cfg_UHRS(nt, dt, fHF, targetHz, nStages)
cfg.nt                  = nt;
cfg.dt                  = dt;
cfg.fHF                 = fHF;
cfg.targetHz            = targetHz;
cfg.nStages             = nStages;
cfg.k_values            = 10:-2:2;
cfg.g_values            = 2:400;
cfg.amp                 = 1;
cfg.maxTraces           = 2000;
cfg.refl_mode           = "white";
cfg.color_sigma_samples = 5;
end

function [XHF, YLF_all] = make_dataset_UHRS(cfg)
nt       = cfg.nt;
dt       = cfg.dt;
tmax     = (nt - 1) * dt;
nStages  = cfg.nStages;
targetHz = cfg.targetHz;
k_values = cfg.k_values;
g_values = cfg.g_values;
Nuse     = min(numel(k_values) * numel(g_values), cfg.maxTraces);

wHF = uhrs_wavelet(cfg.fHF, dt, 1.0);
wLF = cell(nStages, 1);
for s = 1:nStages
    wLF{s} = uhrs_wavelet(targetHz(s), dt, 1.0);
end

XHF     = zeros(nt, Nuse);
YLF_all = zeros(nt, Nuse, nStages);
idx     = 0;
for ik = 1:numel(k_values)
    k = k_values(ik);
    for ig = 1:numel(g_values)
        idx = idx + 1;
        if idx > Nuse, break; end
        g = g_values(ig);
        r = reflec(cfg.refl_mode, tmax, dt, cfg.amp, k, g, cfg.color_sigma_samples);
        r = flipud(r);
        r(1001:end) = 0;   % shallow reflectors only (matches generator)
        tmp = conv(r, wHF);  XHF(:, idx) = tmp(1:nt);
        for s = 1:nStages
            tmp = conv(r, wLF{s});  YLF_all(:, idx, s) = tmp(1:nt);
        end
    end
    if idx >= Nuse, break; end
end
end

function r = reflec(mode, tmax, dt, amp, m, seed, color_sigma)
t_vec = (0:dt:tmax)';
rng(abs(seed), 'twister');
tmp = randn(size(t_vec));
if string(mode) == "colored"
    L  = max(3, round(6 * color_sigma));
    x  = (-L:L)';
    gk = exp(-(x.^2) / (2 * color_sigma^2));
    gk = gk / sum(gk);
    tmp = conv(tmp, gk, 'same');
end
m = round(m);
if mod(m, 2) == 0
    r = (tmp.^m) .* sign(tmp);
else
    r = tmp.^m;
end
r = amp * r / (max(abs(r)) + 1e-12);
end

% ----- Network evaluation -----
function metrics = eval_net(encNet, headNets, quotaNets, XHF, YLF_all, ...
    muX, sigX, muY, sigY, targetHz, nStages, miniBatchSize, useGPU)
[nt, N] = size(XHF);
Xn      = single((XHF - muX) / sigX);
YPred_n = zeros(nt, N, nStages, 'single');

for b = 1:ceil(N / miniBatchSize)
    si    = (b-1)*miniBatchSize + 1;
    ei    = min(b*miniBatchSize, N);
    bSize = ei - si + 1;

    xb  = reshape(Xn(:,si:ei).', [1, bSize, nt]);
    dlX = dlarray(xb, 'CBT');
    if useGPU, dlX = gpuArray(dlX); end

    feat          = predict(encNet, dlX);
    predsForInput = cell(nStages, 1);

    for s = 1:nStages
        if s == 1 || isempty(quotaNets{s})
            headIn = feat;
        else
            prevStack = cat(1, predsForInput{1:s-1});
            quotaOut  = predict(quotaNets{s}, prevStack);
            headIn    = cat(1, feat, quotaOut);
        end
        pred             = predict(headNets{s}, headIn);
        predsForInput{s} = pred;

        pd = extractdata(pred);
        if useGPU, pd = gather(pd); end
        pd = squeeze(pd);
        if bSize == 1
            YPred_n(:,si:ei,s) = pd(:);
        else
            YPred_n(:,si:ei,s) = pd.';
        end
    end
end

half             = floor(nt/2) + 1;
RMSE_s           = zeros(1,nStages);
RRMSE_s          = zeros(1,nStages);
R2_s             = zeros(1,nStages);
StdRatio_s       = zeros(1,nStages);
OutputSNR_dB_s   = zeros(1,nStages);
MedianEnvCorr_s  = zeros(1,nStages);
P10EnvCorr_s     = zeros(1,nStages);
P90EnvCorr_s     = zeros(1,nStages);
MedianSpecCorr_s = zeros(1,nStages);
MedianCorr_s     = zeros(1,nStages);
P10Corr_s        = zeros(1,nStages);
P90Corr_s        = zeros(1,nStages);

for s = 1:nStages
    Yhat  = double(YPred_n(:,:,s))*sigY(s) + muY(s);
    Ytrue = YLF_all(:,:,s);
    err   = Yhat - Ytrue;

    rmse_s  = sqrt(mean(err(:).^2));
    rrmse_s = rmse_s / (max(Ytrue(:)) - min(Ytrue(:)) + 1e-12);
    ss_res  = sum(err(:).^2);
    ss_tot  = sum((Ytrue(:)-mean(Ytrue(:))).^2);
    r2_s    = 1 - ss_res/(ss_tot+1e-12);
    stdratio_s   = std(Yhat(:))/(std(Ytrue(:))+1e-12);
    snr_s   = 10*log10(sum(Ytrue(:).^2)/(sum(err(:).^2)+1e-30));

    Y0     = Ytrue - mean(Ytrue,1);
    H0     = Yhat  - mean(Yhat, 1);
    corr_s = sum(Y0.*H0,1) ./ sqrt(sum(Y0.^2,1).*sum(H0.^2,1)+1e-12);

    env_t = abs(hilbert(Ytrue)); env_p = abs(hilbert(Yhat));
    Ev    = env_t - mean(env_t,1); Ep = env_p - mean(env_p,1);
    env_s = sum(Ev.*Ep,1) ./ sqrt(sum(Ev.^2,1).*sum(Ep.^2,1)+1e-12);

    Sf = abs(fft(Yhat, nt,1)); Sf=Sf(1:half,:);
    St = abs(fft(Ytrue,nt,1)); St=St(1:half,:);
    Sf0=Sf-mean(Sf,1); St0=St-mean(St,1);
    spec_s = sum(Sf0.*St0,1)./(sqrt(sum(Sf0.^2,1).*sum(St0.^2,1))+1e-12);

    RMSE_s(s)           = rmse_s;
    RRMSE_s(s)          = rrmse_s;
    R2_s(s)             = r2_s;
    StdRatio_s(s)       = stdratio_s;
    OutputSNR_dB_s(s)   = snr_s;
    MedianEnvCorr_s(s)  = median(env_s);
    P10EnvCorr_s(s)     = prctile(env_s,10);
    P90EnvCorr_s(s)     = prctile(env_s,90);
    MedianSpecCorr_s(s) = median(spec_s);
    MedianCorr_s(s)     = median(corr_s);
    P10Corr_s(s)        = prctile(corr_s,10);
    P90Corr_s(s)        = prctile(corr_s,90);
end

metrics                = struct();
metrics.RMSE           = RMSE_s;
metrics.RRMSE          = RRMSE_s;
metrics.R2             = R2_s;
metrics.StdRatio       = StdRatio_s;
metrics.OutputSNR_dB   = OutputSNR_dB_s;
metrics.MedianEnvCorr  = MedianEnvCorr_s;
metrics.P10EnvCorr     = P10EnvCorr_s;
metrics.P90EnvCorr     = P90EnvCorr_s;
metrics.MedianSpecCorr = MedianSpecCorr_s;
metrics.MedianCorr     = MedianCorr_s;
metrics.P10Corr        = P10Corr_s;
metrics.P90Corr        = P90Corr_s;
metrics.MeanRMSE       = mean(RMSE_s);
metrics.N              = N;
metrics.targetHz       = targetHz;
metrics.TestName       = '';
metrics.Section        = '';
metrics.Variant        = '';
end

function print_m(m)
fprintf('  %-54s | MeanRMSE=%9.4g | MeanCorr=%6.3f\n', ...
    m.TestName, m.MeanRMSE, mean(m.MedianCorr));
end

function write_block(fid, m, targetHz)
fprintf(fid, '\n--- %s | Section=%s | Variant=%s | N=%d | MeanRMSE=%.4g ---\n', ...
    m.TestName, m.Section, m.Variant, m.N, m.MeanRMSE);
fprintf(fid, '%-8s | %8s | %7s | %8s | %8s | %9s | %8s | %7s | %7s | %9s | %8s | %7s | %7s\n', ...
    'Freq(Hz)','RMSE','RRMSE','R2','StdRatio','SNRout_dB', ...
    'EnvCorr','E_P10','E_P90','SpecCorr','medCorr','C_P10','C_P90');
fprintf(fid, '%s\n', repmat('-',1,132));
for s = 1:numel(targetHz)
    fprintf(fid, '%-8d | %8.5g | %7.4f | %8.5f | %8.5f | %9.2f | %8.3f | %7.3f | %7.3f | %9.4f | %8.3f | %7.3f | %7.3f\n', ...
        targetHz(s), m.RMSE(s), m.RRMSE(s), m.R2(s), m.StdRatio(s), ...
        m.OutputSNR_dB(s), m.MedianEnvCorr(s), m.P10EnvCorr(s), m.P90EnvCorr(s), ...
        m.MedianSpecCorr(s), m.MedianCorr(s), m.P10Corr(s), m.P90Corr(s));
end
end

function tf = canUseGPU()
try, tf = (gpuDeviceCount("available") > 0); catch, tf = false; end
end
