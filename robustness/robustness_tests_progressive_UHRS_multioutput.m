% Title: robustness_tests_progressive_UHRS_multioutput.m
% Author: N. D. Vilanakis
%
% A progressive multi-output 1-D convolutional network for low frequency
% seismic trace reconstruction
% N. D. Vilanakis, N. Economou, A. Vafidis
% Computers and Geosciences
%
% User guide
%   Purpose : Runs the robustness suite of Section 3.4. Deployment conditions
%             are perturbed one at a time: wavelet frequency shift, phase
%             rotation, Hilbert phase, minimum phase, truncation, Klauder
%             wavelet, additive noise from 40 dB down to 0 dB, and
%             out-of-distribution reflectivity.
%   Inputs  : CNN_progressive_matchedRF_UHRS.mat, the trained network, and dataset_train_val_test_UHRS.mat,
%             the UHRS test split. The Signal Processing Toolbox is needed here,
%             for the hilbert function.
%   Outputs : robustness_progressive_results.mat with the full per-test metrics, and
%             robustness_progressive_summary.txt, a readable summary that ends with the
%             composite robustness index.

%% robustness_tests_progressive_multioutput.m  (v4  UHRS: MX150 / full N=4500)
%
% Changes from v2:
%   1. UHRS wavelet: replaced UHRS_fixed()  which was a Ricker  with
%      uhrs_wavelet(), a genuine Gaussian-windowed cosine matching
%      reflectivity_generator_UHRS_symmetric.m exactly:
%        sizew(f) = floor(6400/f * 78),  alphaw(f) = 20 * sizew/78
%        w = (gausswin_local(sizew,alphaw) - min) .* (-cos(2*pi*f*t)), even sizew, zero-padded to 1023.
%   2. A3 test: UHRS wavelet length is frequency-dependent so a fixed nw
%      buffer change is meaningless. A3 now tests Gaussian alpha � 0.5
%      (half-width envelope � broader effective wavelet shape mismatch).
%   3. A8: HF freq shift now uses UHRS(5500 Hz)  the nearest frequency in
%      this model's list [500 600 800 1100 1500 2000 2600 3300 4500 5500 6400].
%   4. OOD synthetic traces (C1, C3): make_dataset_progressive now uses
%      causal truncation [conv(r,w); result(1:nt)] and r(1001:end)=0,
%      matching the actual reflectivity_generator_UHRS_symmetric.m exactly.
%   5. C1 k-values: changed from {12,14,16} (IN training distribution,
%      k_max=20) to {22,24,26}  genuinely OOD spikiness.
%   6. eval_net_progressive: quota guard added (isempty check on quotaNets{s}).
%
% Changes from v3:
%   7. miniBatchSize: 64 -> 512  (NVIDIA MX150, Pascal, 2 GB GDDR5).
%   8. maxN: 2000 -> 4500  (full test split: 15% of 30000 grouped traces).
%
% DESIGN PHILOSOPHY (unchanged from v2):
%   Baseline          : X_test,       Y_test_all          (real data, no change)
%   LF wavelet (A1A7): X_test,       wavelet_reshape(Y)  (perturb LF reference)
%   HF wavelet (A8A9): reshape/phase(X_test), Y_test_all (perturb HF input)
%   Noise (B1B6)     : noisy(X_test), Y_test_all         (perturb HF input)
%   Amplitude (C2,C4) : ��X_test,     ��Y_test_all        (joint scaling)
%   OOD reflectivity  : synthetic BUT amplitude-matched to rms(X_test)
%   Combined (D1D3)  : combine operators above
%
% Input:  6400 Hz traces (2001 samples, dt=1e-5)
% Output: 10 LF traces [500 600 800 1100 1500 2000 2600 3300 4500 5500] Hz
%
% Outputs:
%   robustness_progressive_results.mat
%   robustness_progressive_summary.txt

clear; clc; close all;

%% -------------------- Load trained progressive model --------------------
fprintf('Loading progressive model...\n');
M = load('CNN_progressive_matchedRF_UHRS.mat', ...
    'encNet','headNets','quotaNets', ...
    'muX','sigX','muY','sigY', ...
    'targetHz','targetVarIdx','nStages','C_quota','detachUpTo','t','dt');

encNet       = M.encNet;
headNets     = M.headNets;
quotaNets    = M.quotaNets;
muX          = M.muX;    sigX = M.sigX;
muY          = M.muY;    sigY = M.sigY;
targetHz     = M.targetHz;
targetVarIdx = M.targetVarIdx;
nStages      = M.nStages;
t            = M.t(:);
dt           = mean(diff(t));
nt           = numel(t);

fprintf('Loaded. nStages=%d, nt=%d, dt=%g\n', nStages, nt, dt);
fprintf('Target Hz: %s\n', sprintf('%d ', targetHz));

%% -------------------- GPU detection --------------------
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

%% -------------------- Constants --------------------
fHF           = 6400;
nw_klauder    = 511;   % sweep duration for Klauder wavelet ONLY (511 * dt = 5.11 ms)
miniBatchSize = 256;    % RTX 5060 Laptop (Blackwell, 8 GB) - inference only, no effect on results; raise to 1024 if VRAM allows
maxN          = 4500;  % full test split: 15% of 30000 grouped traces

%% -------------------- Load actual test data --------------------
fprintf('\nLoading actual test data...\n');
vars = {'X_test'};
for s = 1:nStages
    vars{end+1} = sprintf('Y_test%d', targetVarIdx(s));   %#ok<SAGROW>
end
D      = load('dataset_train_val_test_UHRS.mat', vars{:});
X_full = D.X_test;   % [nt x Ntest]
Ntest  = size(X_full, 2);

Y_full = zeros(nt, Ntest, nStages);
for s = 1:nStages
    fname = sprintf('Y_test%d', targetVarIdx(s));
    Y_full(:,:,s) = D.(fname);
end
clear D;

% Cap at maxN traces; use the first maxN to keep experiments comparable
N      = min(Ntest, maxN);
X_base = X_full(:, 1:N);       % [nt x N]   real HF input
Y_base = Y_full(:, 1:N, :);    % [nt x N x nStages]  real LF targets
clear X_full Y_full;

% RMS amplitude of real HF data  used to amplitude-match OOD synthetic traces
rms_Xbase = sqrt(mean(X_base(:).^2));

fprintf('Using %d test traces (of %d available).\n', N, Ntest);

%% -------------------- Pre-compute training UHRS wavelets --------------------
% uhrs_wavelet() replicates reflectivity_generator_UHRS_symmetric.m exactly.
% No nw argument: wavelet length is floor(6400/f * 78), buffer = 1023.
wUHRS_LF = cell(nStages, 1);
for s = 1:nStages
    wUHRS_LF{s} = uhrs_wavelet(targetHz(s), dt);
end
wHF_UHRS = uhrs_wavelet(fHF, dt);

%% =================== Baseline ===================
fprintf('\n=== Baseline (real test data, matched wavelets, clean HF) ===\n');
metrics_baseline = eval_net_progressive(encNet, headNets, quotaNets, ...
    X_base, Y_base, muX, sigX, muY, sigY, targetHz, nStages, miniBatchSize, useGPU);
metrics_baseline.TestName = "Baseline";
metrics_baseline.Section  = "Baseline";
allResults = metrics_baseline;
print_metrics_progressive(metrics_baseline, targetHz);

%% =================== (A) Wavelet mismatch ===================
fprintf('\n=== (A) LF target wavelet mismatch ===\n');

% A1) LF relative frequency shift +15% on all stages
%     Converts each Y_base stage from UHRS(fS) to UHRS(fS*1.15) via
%     spectral division + reconvolution.
Y_A1 = Y_base;
for s = 1:nStages
    wNew = uhrs_wavelet(targetHz(s) * 1.15, dt);
    Y_A1(:,:,s) = wavelet_reshape(Y_base(:,:,s), wUHRS_LF{s}, wNew, nt);
end
mA1 = eval_net_progressive(encNet, headNets, quotaNets, ...
    X_base, Y_A1, muX, sigX, muY, sigY, targetHz, nStages, miniBatchSize, useGPU);
mA1.TestName = "LF wavelet: rel freq shift +15% (all stages)";
mA1.Section  = "A";
allResults(end+1) = mA1;  print_metrics_progressive(mA1, targetHz);
corr_A1    = mean(mA1.MedianCorr);       % R6: freq-shift tolerance

% A2) LF phase rotation +45� on all stages
Y_A2 = Y_base;
for s = 1:nStages
    Y_A2(:,:,s) = phase_rotate_matrix(Y_base(:,:,s), 45);
end
mA2 = eval_net_progressive(encNet, headNets, quotaNets, ...
    X_base, Y_A2, muX, sigX, muY, sigY, targetHz, nStages, miniBatchSize, useGPU);
mA2.TestName = "LF wavelet: phase rotation +45 deg";
mA2.Section  = "A";
allResults(end+1) = mA2;  print_metrics_progressive(mA2, targetHz);

% A3) UHRS Gaussian alpha � 0.5 on all stages (broader envelope mismatch)
%     UHRS wavelet length is frequency-dependent, so a fixed nw buffer change
%     (used in the Ricker version) is not applicable here.
%     Instead we halve alphaw: a narrower/broader Gaussian window changes the
%     effective wavelet shape in a way directly analogous to length mismatch.
Y_A3 = Y_base;
for s = 1:nStages
    wPerturbed = uhrs_wavelet(targetHz(s), dt, 0.5);   % alpha_scale = 0.5
    Y_A3(:,:,s) = wavelet_reshape(Y_base(:,:,s), wUHRS_LF{s}, wPerturbed, nt);
end
mA3 = eval_net_progressive(encNet, headNets, quotaNets, ...
    X_base, Y_A3, muX, sigX, muY, sigY, targetHz, nStages, miniBatchSize, useGPU);
mA3.TestName = "LF wavelet: Gaussian alpha x0.5 (broader envelope)";
mA3.Section  = "A";
allResults(end+1) = mA3;  print_metrics_progressive(mA3, targetHz);

% A4) LF Hilbert phase (+90�)
Y_A4 = Y_base;
for s = 1:nStages
    Y_A4(:,:,s) = imag(hilbert(Y_base(:,:,s)));
end
mA4 = eval_net_progressive(encNet, headNets, quotaNets, ...
    X_base, Y_A4, muX, sigX, muY, sigY, targetHz, nStages, miniBatchSize, useGPU);
mA4.TestName = "LF wavelet: Hilbert phase (+90 deg)";
mA4.Section  = "A";
allResults(end+1) = mA4;  print_metrics_progressive(mA4, targetHz);

% A5) LF minimum-phase equivalent
Y_A5 = Y_base;
for s = 1:nStages
    wMin = minphase_wavelet(wUHRS_LF{s});
    Y_A5(:,:,s) = wavelet_reshape(Y_base(:,:,s), wUHRS_LF{s}, wMin, nt);
end
mA5 = eval_net_progressive(encNet, headNets, quotaNets, ...
    X_base, Y_A5, muX, sigX, muY, sigY, targetHz, nStages, miniBatchSize, useGPU);
mA5.TestName = "LF wavelet: minimum-phase equivalent";
mA5.Section  = "A";
allResults(end+1) = mA5;  print_metrics_progressive(mA5, targetHz);

% A6) LF wavelet truncated to central 50% of its active length
%     The UHRS wavelet has sizew active samples then zeros; we truncate the
%     active portion to 50% and reshape the target.
Y_A6 = Y_base;
for s = 1:nStages
    w0  = wUHRS_LF{s};
    % Find active (non-zero) length
    actLen = find(w0 ~= 0, 1, 'last');
    if isempty(actLen), actLen = numel(w0); end
    L   = round(0.25 * actLen);
    mid = round(actLen / 2);
    L   = min(L, mid - 1);   % guard against out-of-bounds
    wTr = w0(max(1, mid - L) : min(actLen, mid + L));
    Y_A6(:,:,s) = wavelet_reshape(Y_base(:,:,s), wUHRS_LF{s}, wTr, nt);
end
mA6 = eval_net_progressive(encNet, headNets, quotaNets, ...
    X_base, Y_A6, muX, sigX, muY, sigY, targetHz, nStages, miniBatchSize, useGPU);
mA6.TestName = "LF wavelet: truncated to 50% active length";
mA6.Section  = "A";
allResults(end+1) = mA6;  print_metrics_progressive(mA6, targetHz);

% A7) LF Klauder wavelet (vibroseis-style linear sweep)
%     Per stage: centre freq = targetHz(s), bandwidth = 50% of centre freq,
%     sweep duration = nw_klauder * dt = 5.11 ms.
Y_A7 = Y_base;
for s = 1:nStages
    wKl = klauder_wavelet(targetHz(s), targetHz(s) * 0.5, dt, nw_klauder * dt);
    Y_A7(:,:,s) = wavelet_reshape(Y_base(:,:,s), wUHRS_LF{s}, wKl, nt);
end
mA7 = eval_net_progressive(encNet, headNets, quotaNets, ...
    X_base, Y_A7, muX, sigX, muY, sigY, targetHz, nStages, miniBatchSize, useGPU);
mA7.TestName = "LF wavelet: Klauder (bw=50%*f0, T=5.11ms per stage)";
mA7.Section  = "A";
allResults(end+1) = mA7;  print_metrics_progressive(mA7, targetHz);

% ------ HF input wavelet mismatch (A8A9) ------
fprintf('\n=== (A cont.) HF input wavelet mismatch ===\n');

% A8) HF dominant frequency shift: UHRS(6400) � UHRS(5500)
%     5500 Hz is the nearest lower frequency in this model's target list.
wHF_5500 = uhrs_wavelet(5500, dt);
X_A8 = wavelet_reshape(X_base, wHF_UHRS, wHF_5500, nt);
mA8 = eval_net_progressive(encNet, headNets, quotaNets, ...
    X_A8, Y_base, muX, sigX, muY, sigY, targetHz, nStages, miniBatchSize, useGPU);
mA8.TestName = "HF wavelet: UHRS freq shift 6400->5500 Hz";
mA8.Section  = "A";
allResults(end+1) = mA8;  print_metrics_progressive(mA8, targetHz);

% A9) HF phase rotation +45� on input
X_A9 = phase_rotate_matrix(X_base, 45);
mA9 = eval_net_progressive(encNet, headNets, quotaNets, ...
    X_A9, Y_base, muX, sigX, muY, sigY, targetHz, nStages, miniBatchSize, useGPU);
mA9.TestName = "HF wavelet: phase rotation +45 deg";
mA9.Section  = "A";
allResults(end+1) = mA9;  print_metrics_progressive(mA9, targetHz);

%% =================== (B) Noise on HF input (SNR sweep) ===================
fprintf('\n=== (B) HF input noise tests ===\n');

snr_list = [40 30 20 10 5 0];
mB = cell(numel(snr_list), 1);
snr_corr_B = zeros(1, numel(snr_list)); % R6: per-SNR MeanCorr tracking

for ii = 1:numel(snr_list)
    X_noisy = add_noise_snr_matrix(X_base, snr_list(ii));
    mB{ii}  = eval_net_progressive(encNet, headNets, quotaNets, ...
        X_noisy, Y_base, muX, sigX, muY, sigY, targetHz, nStages, miniBatchSize, useGPU);
    mB{ii}.TestName = sprintf('HF noise: SNR=%g dB', snr_list(ii));
    mB{ii}.Section  = "B";
    allResults(end+1) = mB{ii};
    print_metrics_progressive(mB{ii}, targetHz);
    snr_corr_B(ii) = mean(mB{ii}.MedianCorr);
end

%% =================== (C) OOD reflectivity statistics ===================
fprintf('\n=== (C) OOD reflectivity statistics ===\n');

% C1) OOD spikiness k={22,24,26}  above training k_max=20.
%     Training used k_values = 20:-2:2 (i.e., k in {2,4,...,20}).
%     k={22,24,26} is genuinely outside the training reflectivity distribution.
%     Seeds 6001:6600 are outside the training seed range (2:3001).
%     Amplitude is matched to rms(X_base) to stay within the normalisation range.
cfgC1.nt        = nt;
cfgC1.dt        = dt;
cfgC1.fHF       = fHF;
cfgC1.targetHz  = targetHz;
cfgC1.nStages   = nStages;
cfgC1.k_values  = [22 24 26];
cfgC1.g_values  = 6001:6600;
cfgC1.amp       = 1;
cfgC1.maxTraces = maxN;
cfgC1.refl_mode           = "white";
cfgC1.color_sigma_samples = 5;
[XHF_C1, YLF_C1] = make_dataset_progressive(cfgC1);
scale_C1 = rms_Xbase / (sqrt(mean(XHF_C1(:).^2)) + 1e-12);
XHF_C1   = XHF_C1 * scale_C1;
YLF_C1   = YLF_C1 * scale_C1;
mC1 = eval_net_progressive(encNet, headNets, quotaNets, ...
    XHF_C1, YLF_C1, muX, sigX, muY, sigY, targetHz, nStages, miniBatchSize, useGPU);
mC1.TestName = "OOD: spikiness k={22,24,26} (above k_max=20)";
mC1.Section  = "C";
allResults(end+1) = mC1;  print_metrics_progressive(mC1, targetHz);

% C2) Amplitude scaling 0.3�  joint HF + LF scaling (quieter data)
X_C2 = 0.3 * X_base;
Y_C2 = 0.3 * Y_base;
mC2  = eval_net_progressive(encNet, headNets, quotaNets, ...
    X_C2, Y_C2, muX, sigX, muY, sigY, targetHz, nStages, miniBatchSize, useGPU);
mC2.TestName = "OOD: amplitude amp=0.3";
mC2.Section  = "C";
allResults(end+1) = mC2;  print_metrics_progressive(mC2, targetHz);

% C3) Colored (band-limited) reflectivity  non-white reflectivity statistics.
%     Seeds 8001:8600 are outside the training range; k_values within training
%     range is intentional (isolating the coloring as the OOD factor).
cfgC3.nt        = nt;
cfgC3.dt        = dt;
cfgC3.fHF       = fHF;
cfgC3.targetHz  = targetHz;
cfgC3.nStages   = nStages;
cfgC3.k_values  = 10:-2:2;
cfgC3.g_values  = 8001:8600;
cfgC3.amp       = 1;
cfgC3.maxTraces = maxN;
cfgC3.refl_mode           = "colored";
cfgC3.color_sigma_samples = 5;
[XHF_C3, YLF_C3] = make_dataset_progressive(cfgC3);
scale_C3 = rms_Xbase / (sqrt(mean(XHF_C3(:).^2)) + 1e-12);
XHF_C3   = XHF_C3 * scale_C3;
YLF_C3   = YLF_C3 * scale_C3;
mC3 = eval_net_progressive(encNet, headNets, quotaNets, ...
    XHF_C3, YLF_C3, muX, sigX, muY, sigY, targetHz, nStages, miniBatchSize, useGPU);
mC3.TestName = "OOD: colored reflectivity (sigma=5 samp)";
mC3.Section  = "C";
allResults(end+1) = mC3;  print_metrics_progressive(mC3, targetHz);
corr_C3    = mean(mC3.MedianCorr);       % R6: coloured-refl. tolerance

% C4) Amplitude scaling 2.0�  joint HF + LF scaling (louder data)
X_C4 = 2.0 * X_base;
Y_C4 = 2.0 * Y_base;
mC4  = eval_net_progressive(encNet, headNets, quotaNets, ...
    X_C4, Y_C4, muX, sigX, muY, sigY, targetHz, nStages, miniBatchSize, useGPU);
mC4.TestName = "OOD: amplitude amp=2.0 (input gain shift)";
mC4.Section  = "C";
allResults(end+1) = mC4;  print_metrics_progressive(mC4, targetHz);

%% =================== (D) Combined stress tests ===================
fprintf('\n=== (D) Combined stress tests ===\n');

% D1) Moderate noise (SNR=10 dB) + LF phase rotation +45�
X_D1 = add_noise_snr_matrix(X_base, 10);
Y_D1 = Y_A2;
mD1  = eval_net_progressive(encNet, headNets, quotaNets, ...
    X_D1, Y_D1, muX, sigX, muY, sigY, targetHz, nStages, miniBatchSize, useGPU);
mD1.TestName = "Combined: SNR=10 dB + LF phase +45 deg";
mD1.Section  = "D";
allResults(end+1) = mD1;  print_metrics_progressive(mD1, targetHz);

% D2) Moderate noise (SNR=20 dB) + LF relative frequency shift +15%
X_D2 = add_noise_snr_matrix(X_base, 20);
Y_D2 = Y_A1;
mD2  = eval_net_progressive(encNet, headNets, quotaNets, ...
    X_D2, Y_D2, muX, sigX, muY, sigY, targetHz, nStages, miniBatchSize, useGPU);
mD2.TestName = "Combined: SNR=20 dB + LF rel freq shift +15%";
mD2.Section  = "D";
allResults(end+1) = mD2;  print_metrics_progressive(mD2, targetHz);

% D3) OOD spikiness k={22,24,26} + moderate noise (SNR=20 dB)
%     Reuses amplitude-matched OOD data from C1.
XHF_D3 = add_noise_snr_matrix(XHF_C1, 20);
mD3 = eval_net_progressive(encNet, headNets, quotaNets, ...
    XHF_D3, YLF_C1, muX, sigX, muY, sigY, targetHz, nStages, miniBatchSize, useGPU);
mD3.TestName = "Combined: OOD k={22,24,26} + SNR=20 dB";
mD3.Section  = "D";
allResults(end+1) = mD3;  print_metrics_progressive(mD3, targetHz);


%% =================== (R6) Composite Robustness Index ===================
fprintf('\n=== (R6) Composite Robustness Index (CRI) ===\n');
fprintf('  Three independent robustness axes:\n');
fprintf('  (a) SNR knee  : Section-B SNR at which MeanCorr first drops below 0.90\n');
fprintf('  (b) A1 corr   : MeanCorr under +15%% LF freq shift   (spectral tolerance)\n');
fprintf('  (c) C3 corr   : MeanCorr under coloured reflectivity (statistical OOD)\n');
fprintf('  CRI = ( knee_norm + clamp(A1,0,1) + clamp(C3,0,1) ) / 3   in [0,1]\n\n');

% (a) SNR knee
snr_knee_idx = find(snr_corr_B < 0.90, 1, 'first');
if isempty(snr_knee_idx)
    snr_knee_norm = 1.0;
    snr_knee_str  = 'immune (>= 0.90 at all B-tests incl. SNR=0 dB)';
else
    snr_knee_norm = (snr_knee_idx - 1) / numel(snr_list);
    snr_knee_str  = sprintf('first failure at %g dB (B%d), knee_norm=%.3f', ...
        snr_list(snr_knee_idx), snr_knee_idx, snr_knee_norm);
end
% (b) A1, (c) C3
A1_norm = max(0, corr_A1);
C3_norm = max(0, corr_C3);
CRI     = (snr_knee_norm + A1_norm + C3_norm) / 3;

fprintf('  (a) SNR knee  : %s\n',            snr_knee_str);
fprintf('  (b) A1 corr   : %+.4f  (norm: %.4f)\n', corr_A1, A1_norm);
fprintf('  (c) C3 corr   : %+.4f  (norm: %.4f)\n', corr_C3, C3_norm);
fprintf('  CRI           : (%.4f + %.4f + %.4f) / 3 = %.4f\n', ...
    snr_knee_norm, A1_norm, C3_norm, CRI);
fprintf('  1.0 = perfect robustness on all axes  |  0.0 = complete failure.\n');
fprintf('  CRI is comparable across UHRS / Ricker and across ablation configurations.\n\n');

%% =================== Save results ===================
save('robustness_progressive_results.mat', ...
    'allResults', 'snr_list', 'fHF', 'targetHz', 'nStages', 'dt', 'nt', ...
    'CRI', 'snr_knee_str', 'corr_A1', 'corr_C3', 'snr_corr_B', '-v7.3');
fprintf('\nSaved: robustness_progressive_results.mat\n');

%% =================== Write text summary ===================
txtFile = 'robustness_progressive_summary.txt';
fid = fopen(txtFile, 'w');
assert(fid > 0, 'Could not open %s for writing.', txtFile);

fprintf(fid, 'ROBUSTNESS TEST SUMMARY  Progressive Multi-Output CNN  (v3 UHRS corrected)\n');
fprintf(fid, 'Date: %s\n', datestr(now));   %#ok<TNOW1,DATST>
fprintf(fid, 'Net file: CNN_progressive_matchedRF_UHRS.mat  (dilated heads, d=[4 4])\n');
fprintf(fid, 'Test data: dataset_train_val_test_UHRS.mat  (N=%d traces)\n', N);
fprintf(fid, 'Target Hz: %s\n', sprintf('%d ', targetHz));
fprintf(fid, 'fHF=%d | nStages=%d | nt=%d | dt=%.2g s\n\n', fHF, nStages, nt, dt);

for ii = 1:numel(allResults)
    write_metrics_block_progressive(fid, allResults(ii), targetHz);
end

fprintf(fid, '\n%s\n', repmat('=', 1, 160));
fprintf(fid, 'COMPACT SUMMARY\n');
fprintf(fid, '%s\n', repmat('=', 1, 160));
fprintf(fid, '%-55s | %5s | %9s | %s | %8s\n', ...
    'Test', 'N', 'MeanRMSE', sprintf('%7dHz', targetHz), 'MeanCorr');
fprintf(fid, '%s\n', repmat('-', 1, 160));
for ii = 1:numel(allResults)
    m = allResults(ii);
    fprintf(fid, '%-55s | %5d | %9.4g |', m.TestName, m.N, m.MeanRMSE);
    fprintf(fid, ' %7.4f', m.RMSE);
    fprintf(fid, ' | %8.4f\n', mean(m.MedianCorr));
end
fprintf(fid, '%s\n', repmat('=', 1, 160));


% R6: Composite Robustness Index in text summary
fprintf(fid, '\n%s\n', repmat('=', 1, 80));
fprintf(fid, 'COMPOSITE ROBUSTNESS INDEX (R6)\n');
fprintf(fid, '%s\n', repmat('-', 1, 80));
fprintf(fid, '  (a) SNR knee  : %s\n',  snr_knee_str);
fprintf(fid, '  (b) A1 corr   : %+.4f\n', corr_A1);
fprintf(fid, '  (c) C3 corr   : %+.4f\n', corr_C3);
fprintf(fid, '  CRI           : %.4f\n',   CRI);
fprintf(fid, '%s\n', repmat('=', 1, 80));

fclose(fid);
fprintf('Saved: %s\n', txtFile);
disp('Done.');

%% ========================== LOCAL FUNCTIONS ==========================

% -----------------------------------------------------------------------
function w = uhrs_wavelet(f, dt, alpha_scale)
% Generate symmetric UHRS (Gaussian-windowed cosine) wavelet, matching
% reflectivity_generator_UHRS_symmetric.m exactly.
%
% Parameters (from generator):
%   fHF_ref  = 6400 Hz   (reference / highest frequency)
%   wavle    = 78        (base length scale in samples)
%   alphaval = 20        (Gaussian width parameter at fHF_ref)
%   nw_buf   = 1023      (zero-padded output buffer length)
%
%   sizew(f) = floor(6400/f * 78)
%   alphaw(f) = alpha_scale * 20 * sizew(f) / sizew(6400)
%
% alpha_scale (optional, default 1.0): multiply alphaw  used by A3 to
%   test Gaussian width mismatch (alpha_scale = 0.5 � broader envelope).
if nargin < 3, alpha_scale = 1.0; end

fHF_ref  = 6400;
wavle    = 78;
alphaval = 20;
nw_buf   = 1023;

sizew_HF = floor(fHF_ref / fHF_ref * wavle);   % = 78 (at f = fHF_ref)
sizew    = floor(fHF_ref / f * wavle);
alphaw   = alpha_scale * alphaval * (sizew / sizew_HF);   % alpha from floored length
sizew    = sizew + mod(sizew, 2) + 2;                     % force even length (newUHRS symmetric)

t1   = (0 : sizew - 1)' * dt;
cosu = -cos(2 * pi * f * t1);
gwin = gausswin_local(sizew, alphaw);
gwin = gwin - min(gwin);   % pedestal subtraction: zero at edges (newUHRS)

wavtemp = gwin .* cosu;
w       = zeros(nw_buf, 1);
w(1:sizew) = wavtemp;
end

% -----------------------------------------------------------------------
function gw = gausswin_local(n, hwidth)
% Custom Gaussian window matching reflectivity_generator_UHRS_symmetric.m:
%   gw = exp(-((x - n/2) / hwidth)^2),  x = 1..n
% NOTE: MATLAB's built-in gausswin() uses a different parameterisation;
%       this implementation must not be replaced with it.
no = n / 2;
x  = (1:n)';
gw = exp(-((x - no) / hwidth).^2);
end

% -----------------------------------------------------------------------
function Y_new = wavelet_reshape(Y, w_from, w_to, nt)
% Convert traces from one wavelet to another via Wiener-stabilised spectral ratio.
% Y      : [nt x N]
% w_from : source wavelet (any length; zero-padded to nt internally)
% w_to   : target wavelet (any length; zero-padded to nt internally)
Yf      = fft(Y, nt, 1);
Wf      = fft(w_from(:), nt);
Wt      = fft(w_to(:),   nt);
eps_reg = 0.01 * max(abs(Wf).^2);   % 1% water-level Wiener regularisation
ratio   = (Wt .* conj(Wf)) ./ (abs(Wf).^2 + eps_reg);
Y_new   = real(ifft(Yf .* ratio, nt, 1));
end

% -----------------------------------------------------------------------
function Y_rot = phase_rotate_matrix(Y, phi_deg)
phi   = deg2rad(phi_deg);
Y_rot = real(ifft(fft(Y, [], 1) * exp(1i * phi), [], 1));
end

% -----------------------------------------------------------------------
function X_noisy = add_noise_snr_matrix(X, snr_db)
Px      = mean(X.^2, 1);
Pn      = Px / 10^(snr_db / 10);
noise   = randn(size(X)) .* sqrt(Pn);
X_noisy = X + noise;
end

% -----------------------------------------------------------------------
function metrics = eval_net_progressive( ...
    encNet, headNets, quotaNets, XHF, YLF_all, ...
    muX, sigX, muY, sigY, targetHz, nStages, miniBatchSize, useGPU)
% Evaluates the progressive network on [nt x N] inputs.

[nt, N] = size(XHF);
Xn      = single((XHF - muX) / sigX);
YPred_n = zeros(nt, N, nStages, 'single');

numBatches = ceil(N / miniBatchSize);
for b = 1:numBatches
    si    = (b - 1) * miniBatchSize + 1;
    ei    = min(b * miniBatchSize, N);
    bSize = ei - si + 1;

    xb  = reshape(Xn(:, si:ei).', [1, bSize, nt]);
    dlX = dlarray(xb, 'CBT');
    if useGPU, dlX = gpuArray(dlX); end

    feat          = predict(encNet, dlX);
    predsForInput = cell(nStages, 1);

    for s = 1:nStages
        % Quota guard: skip quota gate if stage 1 or if quotaNets{s} is empty
        if s == 1 || isempty(quotaNets{s})
            headIn = feat;
        else
            prevStack = cat(1, predsForInput{1:s-1});
            quotaOut  = predict(quotaNets{s}, prevStack);
            headIn    = cat(1, feat, quotaOut);
        end

        pred = predict(headNets{s}, headIn);
        predsForInput{s} = pred;

        predData = extractdata(pred);
        if useGPU, predData = gather(predData); end
        predData = squeeze(predData);
        if bSize == 1
            YPred_n(:, si:ei, s) = predData(:);
        else
            YPred_n(:, si:ei, s) = predData.';
        end
    end
end

half = floor(nt / 2) + 1;

RMSE_s           = zeros(1, nStages);
RRMSE_s          = zeros(1, nStages);
R2_s             = zeros(1, nStages);
StdRatio_s       = zeros(1, nStages);
OutputSNR_dB_s   = zeros(1, nStages);
MedianEnvCorr_s  = zeros(1, nStages);
P10EnvCorr_s     = zeros(1, nStages);
P90EnvCorr_s     = zeros(1, nStages);
MedianSpecCorr_s = zeros(1, nStages);
MedianCorr_s     = zeros(1, nStages);
P10Corr_s        = zeros(1, nStages);
P90Corr_s        = zeros(1, nStages);

for s = 1:nStages
    Yhat  = double(YPred_n(:,:,s)) * sigY(s) + muY(s);
    Ytrue = YLF_all(:,:,s);
    err   = Yhat - Ytrue;

    rmse_s  = sqrt(mean(err(:).^2));
    rrmse_s = rmse_s / (max(Ytrue(:)) - min(Ytrue(:)) + 1e-12);
    ss_res  = sum(err(:).^2);
    ss_tot  = sum((Ytrue(:) - mean(Ytrue(:))).^2);
    r2_s    = 1 - ss_res / (ss_tot + 1e-12);
    stdratio_s   = std(Yhat(:)) / (std(Ytrue(:)) + 1e-12);
    snr_s   = 10 * log10(sum(Ytrue(:).^2) / (sum(err(:).^2) + 1e-30));

    Y0     = Ytrue - mean(Ytrue, 1);
    H0     = Yhat  - mean(Yhat,  1);
    corr_s = sum(Y0.*H0,1) ./ sqrt(sum(Y0.^2,1).*sum(H0.^2,1) + 1e-12);

    env_t  = abs(hilbert(Ytrue));
    env_p  = abs(hilbert(Yhat));
    Ev     = env_t - mean(env_t, 1);
    Ep     = env_p - mean(env_p, 1);
    env_s  = sum(Ev.*Ep,1) ./ sqrt(sum(Ev.^2,1).*sum(Ep.^2,1) + 1e-12);

    Sf     = abs(fft(Yhat,  nt, 1));  Sf = Sf(1:half,:);
    St     = abs(fft(Ytrue, nt, 1));  St = St(1:half,:);
    Sf0    = Sf - mean(Sf,1);
    St0    = St - mean(St,1);
    spec_s = sum(Sf0.*St0,1) ./ sqrt(sum(Sf0.^2,1).*sum(St0.^2,1) + 1e-12);

    RMSE_s(s)           = rmse_s;
    RRMSE_s(s)          = rrmse_s;
    R2_s(s)             = r2_s;
    StdRatio_s(s)       = stdratio_s;
    OutputSNR_dB_s(s)   = snr_s;
    MedianEnvCorr_s(s)  = median(env_s);
    P10EnvCorr_s(s)     = prctile(env_s, 10);
    P90EnvCorr_s(s)     = prctile(env_s, 90);
    MedianSpecCorr_s(s) = median(spec_s);
    MedianCorr_s(s)     = median(corr_s);
    P10Corr_s(s)        = prctile(corr_s, 10);
    P90Corr_s(s)        = prctile(corr_s, 90);
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
end

% -----------------------------------------------------------------------
function print_metrics_progressive(m, targetHz)
fprintf('%-55s | N=%5d | MeanRMSE=%9.4g | RMSE:', ...
    m.TestName, m.N, m.MeanRMSE);
for s = 1:numel(m.RMSE)
    fprintf(' %7.4f', m.RMSE(s));
end
fprintf(' | MeanCorr=%6.3f\n', mean(m.MedianCorr));
end

% -----------------------------------------------------------------------
function write_metrics_block_progressive(fid, m, targetHz)
fprintf(fid, '\n--- %s | Section=%s | N=%d | MeanRMSE=%.4g ---\n', ...
    m.TestName, m.Section, m.N, m.MeanRMSE);
fprintf(fid, '%-8s | %8s | %7s | %8s | %8s | %9s | %8s | %7s | %7s | %9s | %8s | %7s | %7s\n', ...
    'Freq(Hz)','RMSE','RRMSE','R2','StdRatio','SNRout_dB', ...
    'EnvCorr','E_P10','E_P90','SpecCorr','medCorr','C_P10','C_P90');
fprintf(fid, '%s\n', repmat('-', 1, 132));
for s = 1:numel(targetHz)
    fprintf(fid, '%-8d | %8.5g | %7.4f | %8.5f | %8.5f | %9.2f | %8.3f | %7.3f | %7.3f | %9.4f | %8.3f | %7.3f | %7.3f\n', ...
        targetHz(s), ...
        m.RMSE(s),           m.RRMSE(s), ...
        m.R2(s),             m.StdRatio(s), ...
        m.OutputSNR_dB(s), ...
        m.MedianEnvCorr(s),  m.P10EnvCorr(s),  m.P90EnvCorr(s), ...
        m.MedianSpecCorr(s), m.MedianCorr(s),  m.P10Corr(s),    m.P90Corr(s));
end
end

% -----------------------------------------------------------------------
function [XHF, YLF_all] = make_dataset_progressive(cfg)
% Generate paired synthetic HF/LF traces for OOD tests (C1, C3 only).
% Matches reflectivity_generator_UHRS_symmetric.m exactly:
%   - causal convolution: conv(r, w), truncated to first nt samples
%   - r(1001:end) = 0  (only first half of time axis has reflectors)
% Caller is responsible for amplitude-matching to real data RMS.

nt       = cfg.nt;
dt       = cfg.dt;
tmax     = (nt - 1) * dt;
nStages  = cfg.nStages;
targetHz = cfg.targetHz;
k_values = cfg.k_values;
g_values = cfg.g_values;
nK       = numel(k_values);
nG       = numel(g_values);
Nuse     = min(nK * nG, cfg.maxTraces);

% Build UHRS wavelets (matching generator)
wHF = uhrs_wavelet(cfg.fHF, dt);
wLF = cell(nStages, 1);
for s = 1:nStages
    wLF{s} = uhrs_wavelet(targetHz(s), dt);
end

XHF     = zeros(nt, Nuse);
YLF_all = zeros(nt, Nuse, nStages);
idx     = 0;

for ik = 1:nK
    k = k_values(ik);
    for ig = 1:nG
        idx = idx + 1;
        if idx > Nuse, break; end
        g = g_values(ig);

        % Reflectivity  same logic as generator's reflec()
        r = reflec_ood(cfg.refl_mode, tmax, dt, cfg.amp, k, g, ...
                       cfg.color_sigma_samples);
        r = flipud(r);
        r(1001:end) = 0;   % match generator: only first half has reflectors

        % Causal convolution, truncated to nt  matches generator
        tmp = conv(r, wHF);
        XHF(:, idx) = tmp(1:nt);
        for s = 1:nStages
            tmp = conv(r, wLF{s});
            YLF_all(:, idx, s) = tmp(1:nt);
        end
    end
    if idx >= Nuse, break; end
end
end

% -----------------------------------------------------------------------
function r = reflec_ood(mode, tmax, dt, amp, m, seed, color_sigma)
% Reflectivity generator for OOD tests.
% Uses rng('twister') for reproducibility (modern equivalent of randn('seed',...)).
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

% -----------------------------------------------------------------------
function wmin = minphase_wavelet(w)
W    = fft(w);
A    = abs(W) + eps;
phi  = imag(hilbert(log(A)));
wmin = real(ifft(exp(log(A) + 1i * phi)));
end

% -----------------------------------------------------------------------
function w = klauder_wavelet(f0, bw, dt, T)
t = (-T/2 : dt : T/2)';
k = bw / T;
w = cos(2*pi*(f0*t + 0.5*k*t.^2)) .* hann(length(t));
w = w / max(abs(w));
end

% -----------------------------------------------------------------------
function tf = canUseGPU()
try
    tf = (gpuDeviceCount("available") > 0);
catch
    tf = false;
end
end
