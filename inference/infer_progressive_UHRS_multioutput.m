% Title: infer_progressive_UHRS_multioutput.m
% Author: N. D. Vilanakis
%
% A progressive multi-output 1-D convolutional network for lower-frequency
% trace reconstruction of UHRS seismic data
% N. D. Vilanakis, N. Economou, A. Vafidis
% Computers and Geosciences
%
% User guide
%   Purpose : Applies the trained network to the held-out test set and reports
%             the per-band RMSE and the per-trace correlations used in
%             Tables 1 and 2 of the paper.
%   Inputs  : CNN_progressive_matchedRF_UHRS.mat, the trained network,
%             and dataset_train_val_test_UHRS.mat, the UHRS test split.
%   Outputs : predictions_progressive_test.mat, holding the predicted bands YPred_test,
%             the true bands Y_test, the input X_test, rmsePerFreq, t and
%             targetHz, plus a bar chart of the per-band RMSE.

%% infer_progressive_multioutput.m
% Inference using the progressive multi-output 1-D CNN.
% Loads CNN_progressive_multioutput_UHRS.mat and applies it to 6400 Hz
% test traces, predicting 10 LF traces simultaneously.
%
% Based on: infer_cnn_6400_to_singleLF_prelu_v7.m (single-output version)

clear; clc;

%% ========================= LOAD TRAINED MODEL ============================
fprintf('Loading trained progressive model...\n');
model = load('CNN_progressive_matchedRF_UHRS.mat', ...
    'encNet', 'headNets', 'quotaNets', ...
    'muX', 'sigX', 'muY', 'sigY', ...
    'targetHz', 'targetVarIdx', 'nStages', 'C_quota', 'detachUpTo', ...
    't', 'dt');

encNet    = model.encNet;
headNets  = model.headNets;
quotaNets = model.quotaNets;
muX       = model.muX;
sigX      = model.sigX;
muY       = model.muY;       % [1 x nStages]
sigY      = model.sigY;      % [1 x nStages]
targetHz  = model.targetHz;  % [5500 4500 3300 2600 2000 1500 1100 800 600 500]
targetVarIdx = model.targetVarIdx;  % [10 9 8 7 6 5 4 3 2 1]
nStages   = model.nStages;
t         = model.t;
dt        = model.dt;

fprintf('Model loaded: %d stages, frequencies: %s Hz\n', ...
    nStages, sprintf('%d ', targetHz));

%% ========================= LOAD TEST DATA ================================
fprintf('Loading test data...\n');

% Build variable list for test set
vars = {'X_test'};
for s = 1:nStages
    vars{end+1} = sprintf('Y_test%d', targetVarIdx(s));  %#ok<SAGROW>
end

data = load('dataset_train_val_test_UHRS.mat', vars{:});
X_test = data.X_test;   % [nt x Ntest], 6400 Hz

[nt, Ntest] = size(X_test);
fprintf('Inference on %d test traces, each with %d samples.\n', Ntest, nt);

%% ========================= SELECT EXAMPLE TRACE =========================
rng('shuffle');
idxExample = 1451;%randi(Ntest);
fprintf('\nPlotting example trace #%d\n', idxExample);

%
% Collect ground truth for all stages: [nt x Ntest x nStages]
Y_test = zeros(nt, Ntest, nStages);
hasGroundTruth = true;
for s = 1:nStages
    fname = sprintf('Y_test%d', targetVarIdx(s));
    if isfield(data, fname)
        Y_test(:,:,s) = data.(fname);
    else
        hasGroundTruth = false;
        fprintf('Warning: ground truth %s not found.\n', fname);
    end
end
clear data;

%% ========================= GPU DETECTION =================================
useGPU = false;
try
    if gpuDeviceCount("available") > 0
        useGPU = true;
        g = gpuDevice();
        fprintf('GPU: %s (%s)\n', g.Name, g.ComputeCapability);
    end
catch
    % no GPU available
end

%% ========================= NORMALIZE INPUT ===============================
fprintf('Normalizing input traces...\n');
X_test_n = single((X_test - muX) / sigX);

%% ========================= RUN INFERENCE =================================
fprintf('Running progressive inference...\n');
miniBatchSize = 64;
numBatches = ceil(Ntest / miniBatchSize);

% Pre-allocate output: [nt x Ntest x nStages] in normalised space
YPred_test_n = zeros(nt, Ntest, nStages, 'single');

tInferStart = tic;

for b = 1:numBatches
    si = (b-1)*miniBatchSize + 1;
    ei = min(b*miniBatchSize, Ntest);
    bSize = ei - si + 1;

    % Format as dlarray [C x B x T] ('CBT')
    xb  = reshape(X_test_n(:, si:ei).', [1, bSize, nt]);
    dlX = dlarray(xb, 'CBT');
    if useGPU
        dlX = gpuArray(dlX);
    end

    % Progressive predict chain (inference mode: stored BN stats, no dropout)
    feat = predict(encNet, dlX);     % [128 x B x T]
    predsForInput = cell(nStages, 1);

    for s = 1:nStages
        if s == 1
            headIn = feat;
        else
            prevStack = cat(1, predsForInput{1:s-1});       % [(s-1) x B x T]
            quotaOut  = predict(quotaNets{s}, prevStack);   % [C_quota x B x T]
            headIn    = cat(1, feat, quotaOut);              % [(128+C_quota) x B x T]
        end

        pred = predict(headNets{s}, headIn);   % [1 x B x T]
        predsForInput{s} = pred;

        % Extract and store: pred is [1 x B x T] CBT -> need [nt x B]
        predData = extractdata(pred);       % [1 x B x T]
        if useGPU
            predData = gather(predData);
        end
        predData = squeeze(predData);       % [B x T] or [T] if B==1
        if bSize == 1
            YPred_test_n(:, si:ei, s) = predData(:);   % [T x 1]
        else
            YPred_test_n(:, si:ei, s) = predData.';     % [T x B]
        end
    end

    if mod(b, 50) == 0 || b == numBatches
        fprintf('  Batch %d/%d\n', b, numBatches);
    end
end

inferTimeSec = toc(tInferStart);
fprintf('Inference finished in %.2f sec (%.4f sec/trace)\n', ...
    inferTimeSec, inferTimeSec/Ntest);

%% ========================= DENORMALIZE PREDICTIONS =======================
fprintf('Denormalizing predictions...\n');
YPred_test = zeros(nt, Ntest, nStages);
for s = 1:nStages
    YPred_test(:,:,s) = double(YPred_test_n(:,:,s)) * sigY(s) + muY(s);
end

%% ========================= COMPUTE RMSE PER FREQUENCY ====================
if hasGroundTruth
    fprintf('\nPer-frequency RMSE on test set:\n');
    fprintf('%-10s  %-12s  %-12s\n', 'Frequency', 'RMSE', 'RRMSE(%)');
    fprintf('%s\n', repmat('-', 1, 40));

    rmsePerFreq = zeros(1, nStages);
    for s = 1:nStages
        err = YPred_test(:,:,s) - Y_test(:,:,s);
        rmsePerFreq(s) = sqrt(mean(err(:).^2));
        sigTarget = std(Y_test(:,:,s), 0, 'all');
        relRMSE = 100 * rmsePerFreq(s) / sigTarget;
        fprintf('%5d Hz    %.6g     %.2f%%\n', targetHz(s), rmsePerFreq(s), relRMSE);
    end
    fprintf('%s\n', repmat('-', 1, 40));
    fprintf('Mean RMSE:  %.6g\n', mean(rmsePerFreq));
end



%% ========================= FIGURE 1: All frequencies panel ===============
fig1 = figure('Position', [50 50 900 1400]);

% Subplot 1: Input trace (6400 Hz)
subplot(nStages + 1, 1, 1);
plot(t, X_test(:, idxExample), 'b-', 'LineWidth', 0.8);
xlabel('Time (s)'); ylabel('Amp');
title(sprintf('Input – 6400 Hz (test #%d)', idxExample));
set(gca, 'FontSize', 7); grid on;

% Subplots 2-11: each predicted frequency vs ground truth
for s = 1:nStages
    subplot(nStages + 1, 1, 1 + s);
    if hasGroundTruth
        plot(t, Y_test(:, idxExample, s), 'k-', 'LineWidth', 0.8); hold on;
    end
    plot(t, YPred_test(:, idxExample, s), 'r--', 'LineWidth', 0.8);
    xlabel('Time (s)'); ylabel('Amp');

    if hasGroundTruth
        traceRMSE = sqrt(mean((YPred_test(:,idxExample,s) - Y_test(:,idxExample,s)).^2));
        title(sprintf('%d Hz  (RMSE=%.4g)', targetHz(s), traceRMSE));
    else
        title(sprintf('%d Hz', targetHz(s)));
    end

    if s == 1
        if hasGroundTruth
            legend('True', 'Predicted', 'Location', 'northeast', 'FontSize', 6);
        else
            legend('Predicted', 'Location', 'northeast', 'FontSize', 6);
        end
    end
    set(gca, 'FontSize', 7); grid on;
    hold off;
end

sgtitle(sprintf('Progressive CNN – Test trace #%d', idxExample), 'FontSize', 10);

fileName1 = sprintf('%d_progressive_all_freqs.png', idxExample);
exportgraphics(fig1, fileName1, 'Resolution', 300);
fprintf('Figure saved as: %s\n', fileName1);

%% ========================= FIGURE 2: Selected frequency overlay ==========
fig2 = figure('Position', [100 100 1000 500]);

selectedStages = [1, 3, 6, 10];   % 5000, 3000, 1100, 500 Hz
colors = lines(numel(selectedStages));

hold on;
for k = 1:numel(selectedStages)
    s = selectedStages(k);
    plot(t, YPred_test(:, idxExample, s), '-', 'LineWidth', 1.2, ...
        'Color', colors(k,:), ...
        'DisplayName', sprintf('%d Hz (pred)', targetHz(s)));
end
if hasGroundTruth
    for k = 1:numel(selectedStages)
        s = selectedStages(k);
        plot(t, Y_test(:, idxExample, s), '--', 'LineWidth', 0.8, ...
            'Color', colors(k,:), ...
            'DisplayName', sprintf('%d Hz (true)', targetHz(s)));
    end
end
hold off;

xlabel('Time (s)'); ylabel('Amplitude');
title(sprintf('Selected frequency overlay (test #%d)', idxExample));
legend('Location', 'best'); grid on;

fileName2 = sprintf('%d_progressive_overlay.png', idxExample);
exportgraphics(fig2, fileName2, 'Resolution', 300);
fprintf('Figure saved as: %s\n', fileName2);

%% ========================= FIGURE 3: Per-frequency RMSE bar chart ========
if hasGroundTruth
    fig3 = figure('Position', [150 150 800 400]);
    bar(rmsePerFreq, 'FaceColor', [0.2 0.5 0.8]);
    set(gca, 'XTickLabel', arrayfun(@(f) sprintf('%d', f), targetHz, ...
        'UniformOutput', false));
    xlabel('Target Frequency (Hz)');
    ylabel('RMSE');
    title('Test Set RMSE per Frequency');
    grid on;

    fileName3 = 'progressive_rmse_bar.png';
    exportgraphics(fig3, fileName3, 'Resolution', 300);
    fprintf('Figure saved as: %s\n', fileName3);
end

%% ========================= SAVE PREDICTIONS ==============================
saveFile = 'predictions_progressive_test.mat';
if hasGroundTruth
    save(saveFile, 'YPred_test', 'Y_test', 'X_test', 'rmsePerFreq', ...
        'targetHz', 't', 'dt', 'idxExample', 'Ntest', 'nt', '-v7.3');
else
    save(saveFile, 'YPred_test', 'X_test', ...
        'targetHz', 't', 'dt', 'idxExample', 'Ntest', 'nt', '-v7.3');
end
fprintf('Predictions saved to %s\n', saveFile);

fprintf('Done.\n');

%% ========================= LOCAL HELPERS =================================

function tf = canUseGPU()
    try
        tf = (gpuDeviceCount("available") > 0);
    catch
        tf = false;
    end
end
