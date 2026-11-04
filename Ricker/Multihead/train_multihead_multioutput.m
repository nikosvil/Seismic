% Title: train_multihead_multioutput.m
% Author: N. D. Vilanakis
%
% A progressive multi-output 1-D convolutional network for low frequency
% seismic trace reconstruction
% N. D. Vilanakis, N. Economou, A. Vafidis
% Computers and Geosciences
%
% User guide
%   Purpose : Trains the network described in Section 2.5 of the paper, with a
%             custom training loop so that the per-stage losses and the gradient
%             detachment are handled explicitly.
%             Conditioning  : none. C_quota = 0 and no gradient detachment
%             Head receptive field : 7.96 ms (no dilation)
%   Inputs  : dataset_train_val_test_Ricker.mat, written by the split script in common/.
%             The training settings are at the top of the file: miniBatchSize,
%             maxEpochs, initialLearnRate, l2Reg, C_quota, detachUpTo and the
%             random seed rng(42,'twister').
%   Outputs : CNN_multihead_multioutput_Ricker.mat, holding the trained encoder, heads and quota
%             gates together with the normalization constants, and a text file
%             with the training history, which lists the per-stage validation
%             loss every fifth epoch.

%% train_multihead_multioutput.m  (RICKER)
% ABLATION PACK 1 — Multi-head multi-output 1-D CNN (NO quota conditioning)
% Baseline to compare against the Ricker progressive model.
%
% PUBLICATION RUN (ARIS). Pairs with the Ricker progressive run in
% Progressive/Ricker/Freqs_3/20_3001_128_flat/build_files/.
%
% Matched conditions for the Pack 1 comparison — BOTH runs must use:
%     dataset        30000 traces (22500 / 3000 / 4500), g_final_value = 3001
%     miniBatchSize  256
%     maxEpochs      300, validationFreq 5, validationPatience 25
% Batch size and dataset size both change the trained model, so neither may
% differ between the two models being compared. The earlier Ricker progressive
% model (miniBatch 128) and the New-UHRS multi-head (miniBatch 512) are NOT
% valid partners for this run.
%
% The driver Multi_Trace_CNN_Ricker.m generates the dataset into the working
% directory before training, so the dataset is loaded by bare filename.
%
% Input:  6400 Hz traces (2001 samples, dt=1e-5)
% Output: 10 LF traces [5500,4500,3300,2600,2000,1500,1100,800,600,500] Hz
%
% Architecture:
%   Shared encoder (3 Conv blocks) -> 128-channel features
%   10 INDEPENDENT prediction heads (2 Conv blocks + 1x1 output each):
%     ALL stages: encoder features only (no quota gate, no cross-stage cond.)
%
% Key difference from train_progressive_multioutput.m:
%   - quotaNets removed entirely; every head sees only encOut (128 ch)
%   - detachUpTo = 0 (no detachment needed, no quota path to protect)
%   - All other hyperparameters, loss weighting, and training identical
%
% Training:
%   Custom loop: dlfeval + dlgradient + adamupdate
%   Piecewise LR decay, early stopping on validation
%   Adaptive stage weights: rebalanced based on per-stage validation MSE

clear; close all;

%% ========================================================================
% HARD HEADLESS MODE (SERVER / MATLAB Runtime SAFE)
% =========================================================================
DO_PLOTS = false;   % set true only on your local machine if you want figures
try, set(0,'DefaultFigureVisible','off'); end
try, set(groot,'defaultFigureVisible','off'); end

%% ========================= CONFIGURATION ================================
nStages       = 10;
targetHz      = [5500 4500 3300 2600 2000 1500 1100 800 600 500];
targetVarIdx  = [10   9    8    7    6    5    4   3   2   1  ];
%   targetVarIdx maps progressive stage s to dataset variable Y_train<idx>:
%     Stage 1 -> Y_train10 (5500 Hz), ..., Stage 10 -> Y_train1 (500 Hz)

% --- Network architecture ---
C_quota        = 0;         % no quota gate (kept for save compatibility)
detachUpTo     = 0;         % no detachment needed (no quota path)
filterSizeEnc  = 200;       % encoder conv filter size (large receptive field)
filterSizeHead = 100;       % head conv filter size
encFilters     = [128 64 128];   % encoder block filter counts
headFilters    = [64 32];        % head block filter counts
dropoutRate    = 0.15;

% --- Training hyperparameters ---
miniBatchSize      = 256;   % MUST match the paired Pack 1 run
maxEpochs          = 300;
initialLearnRate   = 1e-4;
l2Reg              = 1e-4;
lrDropFactor       = 0.5;
lrDropPeriod       = 75;     % epochs between LR drops
validationFreq     = 5;      % validate every N epochs
validationPatience = 25;     % early-stop after N val checks with no improvement

% --- Stage loss weights ---
% Geometric base: lambda^(s-1), emphasising harder (later) stages slightly.
% After each validation, weights are further rebalanced by relative val MSE.
lambdaBase  = 1.2;
baseWeights = lambdaBase .^ (0:nStages-1);
baseWeights = baseWeights / mean(baseWeights);   % normalise to mean=1

%% ========================= LOAD DATA ====================================
fprintf('Loading data...\n');
vars = {'X_train','X_val','t','dt'};
for s = 1:nStages
    vars{end+1} = sprintf('Y_train%d', targetVarIdx(s));  %#ok<SAGROW>
    vars{end+1} = sprintf('Y_val%d',   targetVarIdx(s));  %#ok<SAGROW>
end
% Generated into the working directory by the driver (see header).
data = load('dataset_train_val_test_Ricker.mat', vars{:});

X_train = data.X_train;   % [nt x Ntrain], 6400 Hz
X_val   = data.X_val;     % [nt x Nval],   6400 Hz
t       = data.t;
dt      = data.dt;
assert(isfield(data,'dt'), 'dt missing - dataset is invalid');

[nt, Ntrain] = size(X_train);
[~,  Nval]   = size(X_val);
fprintf('Ntrain=%d, Nval=%d, nt=%d, dt=%.6g\n', Ntrain, Nval, nt, dt);

% GPU detection
useGPU = canUseGPU();
if useGPU
    try
        g = gpuDevice();
        fprintf('GPU: %s (%s)\n', g.Name, g.ComputeCapability);
    catch ME
        fprintf('GPU reported available but failed: %s\n', ME.message);
    end
end

%% ========================= NORMALISATION ================================
muX  = mean(X_train(:));
sigX = std(X_train(:));
if sigX == 0, sigX = 1; end

X_train_n = single((X_train - muX) / sigX);
X_val_n   = single((X_val   - muX) / sigX);

muY  = zeros(1, nStages);
sigY = zeros(1, nStages);
Y_train_n = zeros(nt, Ntrain, nStages, 'single');
Y_val_n   = zeros(nt, Nval,   nStages, 'single');

for s = 1:nStages
    idx = targetVarIdx(s);
    Yt = data.(sprintf('Y_train%d', idx));
    Yv = data.(sprintf('Y_val%d',   idx));
    muY(s)  = mean(Yt(:));
    sigY(s) = std(Yt(:));
    if sigY(s) == 0, sigY(s) = 1; end
    Y_train_n(:,:,s) = single((Yt - muY(s)) / sigY(s));
    Y_val_n(:,:,s)   = single((Yv - muY(s)) / sigY(s));
end
clear data Yt Yv;

%% ========================= BUILD NETWORKS ===============================
fprintf('Building multi-head network (%d stages, no quota conditioning)...\n', nStages);

rng(42, 'twister');   % reproducible weight initialisation

encNet = buildEncoder(filterSizeEnc, encFilters, dropoutRate);

% All heads receive only encoder output (128 ch) — no quota
headNets = cell(nStages, 1);
for s = 1:nStages
    headNets{s} = buildHead(encFilters(end), filterSizeHead, headFilters, dropoutRate, s);
end

quotaNets = cell(nStages, 1);   % empty — kept for API/save compatibility

% Parameter count
nParams = countParams(encNet);
for s = 1:nStages,  nParams = nParams + countParams(headNets{s});  end
fprintf('Total learnable parameters: %d (%.2f M)\n', nParams, nParams/1e6);

%% ========================= INIT TRAINING STATE ==========================
currentLR      = initialLearnRate;
iteration      = 0;
bestValLoss    = Inf;
bestNets       = struct();
patienceCounter = 0;
numBatches     = ceil(Ntrain / miniBatchSize);

avgGradEnc = [];  avgSqGradEnc = [];
avgGradHead  = cell(nStages, 1);  avgSqGradHead  = cell(nStages, 1);

stageWeights = baseWeights;

epochLog       = [];
trainLossLog   = [];
valLossLog     = [];
valStageLossLog = [];
lrLog          = [];

%% ========================= TRAINING LOOP ================================
fprintf('Starting training (%d epochs, batch=%d, LR=%.1e)...\n', ...
    maxEpochs, miniBatchSize, currentLR);
tTrainStart   = tic;
earlyStopEpoch = maxEpochs;

for epoch = 1:maxEpochs
    shuffleIdx = randperm(Ntrain);
    epochLoss       = 0;
    epochStageLosses = zeros(1, nStages);

    for b = 1:numBatches
        iteration = iteration + 1;

        si = (b-1)*miniBatchSize + 1;
        ei = min(b*miniBatchSize, Ntrain);
        bIdx  = shuffleIdx(si:ei);
        bSize = numel(bIdx);

        xb = reshape(X_train_n(:,bIdx).', [1, bSize, nt]);
        yb = permute(Y_train_n(:,bIdx,:), [3 2 1]);
        dlX = dlarray(xb, 'CBT');
        dlY = dlarray(yb, 'CBT');
        if useGPU
            dlX = gpuArray(dlX);
            dlY = gpuArray(dlY);
        end

        [loss, grads, encState, headStates, stageLossVals] = dlfeval( ...
            @modelLoss, encNet, headNets, ...
            dlX, dlY, stageWeights);

        encNet.State = encState;
        for s = 1:nStages
            headNets{s}.State = headStates{s};
        end

        % L2 weight decay
        allLearnables = cell(1, 1 + nStages);
        allLearnables{1} = encNet.Learnables;
        for s = 1:nStages
            allLearnables{1+s} = headNets{s}.Learnables;
        end
        for gi = 1:numel(grads)
            grads{gi} = applyL2(grads{gi}, allLearnables{gi}, l2Reg);
        end

        % Adam update: encoder
        gIdx = 1;
        [encNet.Learnables, avgGradEnc, avgSqGradEnc] = adamupdate( ...
            encNet.Learnables, grads{gIdx}, avgGradEnc, avgSqGradEnc, ...
            iteration, currentLR);
        gIdx = gIdx + 1;

        % Adam update: heads
        for s = 1:nStages
            [headNets{s}.Learnables, avgGradHead{s}, avgSqGradHead{s}] = ...
                adamupdate(headNets{s}.Learnables, grads{gIdx}, ...
                avgGradHead{s}, avgSqGradHead{s}, ...
                iteration, currentLR);
            gIdx = gIdx + 1;
        end

        bLoss = double(extractdata(loss));
        epochLoss        = epochLoss + bLoss * bSize;
        epochStageLosses = epochStageLosses + stageLossVals(:).' * bSize;
    end

    epochLoss        = epochLoss / Ntrain;
    epochStageLosses = epochStageLosses / Ntrain;

    if mod(epoch, 5) == 0 || epoch == 1
        fprintf('Ep %3d/%d | LR %.2e | Loss %.6f | Stages: %s\n', ...
            epoch, maxEpochs, currentLR, epochLoss, ...
            sprintf('%.4f ', epochStageLosses));
    end

    if mod(epoch, validationFreq) == 0
        [valLoss, valStageLosses] = evaluateValidation( ...
            encNet, headNets, X_val_n, Y_val_n, ...
            stageWeights, nt, Nval, miniBatchSize, useGPU);

        fprintf('  Val  Loss=%.6f | Stages: %s\n', valLoss, ...
            sprintf('%.4f ', valStageLosses));

        epochLog(end+1)        = epoch;             %#ok<SAGROW>
        trainLossLog(end+1)    = epochLoss;          %#ok<SAGROW>
        valLossLog(end+1)      = valLoss;            %#ok<SAGROW>
        valStageLossLog(end+1,:) = valStageLosses;   %#ok<SAGROW>
        lrLog(end+1)           = currentLR;          %#ok<SAGROW>

        % Adaptive weight rebalancing
        relLosses    = valStageLosses / mean(valStageLosses);
        stageWeights = baseWeights .* relLosses;
        stageWeights = stageWeights / mean(stageWeights) * mean(baseWeights);

        if valLoss < bestValLoss
            bestValLoss     = valLoss;
            patienceCounter = 0;
            bestNets.encNet    = encNet;
            bestNets.headNets  = headNets;
            bestNets.quotaNets = quotaNets;
            bestNets.epoch     = epoch;
        else
            patienceCounter = patienceCounter + 1;
        end

        if patienceCounter >= validationPatience
            fprintf('Early stopping at epoch %d (patience=%d checks)\n', ...
                epoch, validationPatience);
            earlyStopEpoch = epoch;
            break;
        end
    end

    if mod(epoch, lrDropPeriod) == 0
        currentLR = currentLR * lrDropFactor;
        fprintf('  LR decayed to %.2e\n', currentLR);
    end
end

trainTimeSec = toc(tTrainStart);
fprintf('Training finished in %.1f sec (%.1f min)\n', trainTimeSec, trainTimeSec/60);

%% ========================= RESTORE BEST CHECKPOINT ======================
if isfield(bestNets, 'encNet')
    encNet    = bestNets.encNet;
    headNets  = bestNets.headNets;
    quotaNets = bestNets.quotaNets;
    fprintf('Restored best checkpoint from epoch %d (valLoss=%.6f)\n', ...
        bestNets.epoch, bestValLoss);
end

%% ========================= FINAL EVALUATION =============================
fprintf('\nPer-frequency RMSE on validation set (physical units):\n');
[~, valStageMSE] = evaluateValidation( ...
    encNet, headNets, X_val_n, Y_val_n, ...
    ones(1, nStages), nt, Nval, miniBatchSize, useGPU);

for s = 1:nStages
    rmse_phys = sqrt(valStageMSE(s)) * sigY(s);
    fprintf('  Stage %2d (%4d Hz): RMSE = %.6g\n', s, targetHz(s), rmse_phys);
end

%% ========================= OPTIONAL PLOTS ===============================
if DO_PLOTS && ~isempty(epochLog)
    figure('Name','Training curves');
    subplot(2,1,1);
    plot(epochLog, trainLossLog, 'b-', epochLog, valLossLog, 'r-', 'LineWidth',1);
    xlabel('Epoch'); ylabel('Weighted MSE'); legend('Train','Val');
    title('Total loss — Multi-head baseline');

    subplot(2,1,2);
    plot(epochLog, valStageLossLog, 'LineWidth', 1);
    xlabel('Epoch'); ylabel('MSE (normalised)');
    legendStrs = arrayfun(@(f) sprintf('%d Hz', f), targetHz, 'UniformOutput', false);
    legend(legendStrs, 'Location','best');
    title('Per-stage validation MSE');
end

%% ========================= SAVE =========================================
saveFile = 'CNN_multihead_multioutput_Ricker.mat';
save(saveFile, ...
    'encNet', 'headNets', 'quotaNets', ...
    'muX', 'sigX', 'muY', 'sigY', ...
    'targetHz', 'targetVarIdx', 'nStages', 'C_quota', 'detachUpTo', ...
    'baseWeights', 'stageWeights', ...
    'filterSizeEnc', 'filterSizeHead', 'encFilters', 'headFilters', ...
    'dropoutRate', 'miniBatchSize', 'maxEpochs', 'initialLearnRate', 'l2Reg', ...
    't', 'dt', ...
    'epochLog', 'trainLossLog', 'valLossLog', 'valStageLossLog', 'lrLog', ...
    'bestValLoss', 'trainTimeSec', '-v7.3');
fprintf('Saved trained model to %s\n', saveFile);

%% ========================= TRAINING HISTORY FILE ========================
histFile = fullfile(pwd, 'training_history_multihead_Ricker.txt');
fid = fopen(histFile, 'w');
if fid > 0
    fprintf(fid, 'TRAINING HISTORY - Multi-Head Multi-Output CNN (Pack1 Ablation, RICKER)\n');
    fprintf(fid, 'Ablation: Pack1 Multi-Head baseline (no quota conditioning)\n');
    fprintf(fid, 'Date: %s\n', datestr(now));  %#ok<TNOW1,DATST>
    fprintf(fid, 'Target frequencies (Hz): %s\n', sprintf('%d ', targetHz));
    fprintf(fid, 'Ntrain=%d, Nval=%d, nt=%d, dt=%.6g\n', Ntrain, Nval, nt, dt);
    fprintf(fid, 'Encoder: filterSize=%d, filters=[%s], dropout=%.2f\n', ...
        filterSizeEnc, sprintf('%d ', encFilters), dropoutRate);
    fprintf(fid, 'Heads: filterSize=%d, filters=[%s], dropout=%.2f\n', ...
        filterSizeHead, sprintf('%d ', headFilters), dropoutRate);
    fprintf(fid, 'C_quota=%d (no quota gates), detachUpTo=%d\n', C_quota, detachUpTo);
    fprintf(fid, 'miniBatch=%d, maxEpochs=%d, LR=%.1e, L2=%.1e\n', ...
        miniBatchSize, maxEpochs, initialLearnRate, l2Reg);
    fprintf(fid, 'LR schedule: piecewise %.2fx every %d epochs\n', ...
        lrDropFactor, lrDropPeriod);
    fprintf(fid, 'Early stopping: freq=%d epochs, patience=%d checks\n', ...
        validationFreq, validationPatience);
    fprintf(fid, 'Stage weight lambda=%.2f (geometric+adaptive)\n', lambdaBase);
    fprintf(fid, 'Total parameters: %d (%.2f M)\n', nParams, nParams/1e6);
    fprintf(fid, 'Training time: %.1f sec (%.1f min)\n', trainTimeSec, trainTimeSec/60);
    if isfield(bestNets, 'epoch')
        fprintf(fid, 'Best val loss: %.6f (epoch %d)\n', bestValLoss, bestNets.epoch);
    end
    if earlyStopEpoch < maxEpochs
        fprintf(fid, 'Early stopping triggered at epoch %d\n', earlyStopEpoch);
    else
        fprintf(fid, 'Training ran to completion (maxEpochs=%d)\n', maxEpochs);
    end
    fprintf(fid, '\nPer-frequency RMSE on validation (physical units):\n');
    for s = 1:nStages
        fprintf(fid, '  Stage %2d (%4d Hz): RMSE = %.6g\n', s, targetHz(s), sqrt(valStageMSE(s))*sigY(s));
    end
    fprintf(fid, '\n%-6s | %-10s | %-10s | %-10s | %s\n', ...
        'Epoch', 'TrainLoss', 'ValLoss', 'LR', 'Per-stage val MSE');
    fprintf(fid, '%s\n', repmat('-', 1, 80));
    for i = 1:numel(epochLog)
        fprintf(fid, '%5d  | %.6f | %.6f | %.2e | %s\n', ...
            epochLog(i), trainLossLog(i), valLossLog(i), lrLog(i), ...
            sprintf('%.5f ', valStageLossLog(i,:)));
    end
    fclose(fid);
    fprintf('Saved training history to %s\n', histFile);
end

fprintf('Done.\n');

%% ======================= LOCAL HELPER FUNCTIONS =========================

% -------------------------------------------------------------------------
% buildEncoder: shared feature extractor from 6400 Hz input
% -------------------------------------------------------------------------
function net = buildEncoder(filterSize, filters, dropRate)
    layers = [
        sequenceInputLayer(1, Name="enc_input")

        convolution1dLayer(filterSize, filters(1), Padding="same", Name="enc_conv1")
        batchNormalizationLayer(Name="enc_bn1")
        preluLayer(Name="enc_prelu1")
        dropoutLayer(dropRate, Name="enc_drop1")

        convolution1dLayer(filterSize, filters(2), Padding="same", Name="enc_conv2")
        batchNormalizationLayer(Name="enc_bn2")
        preluLayer(Name="enc_prelu2")
        dropoutLayer(dropRate, Name="enc_drop2")

        convolution1dLayer(filterSize, filters(3), Padding="same", Name="enc_conv3")
        batchNormalizationLayer(Name="enc_bn3")
        preluLayer(Name="enc_prelu3")
        dropoutLayer(dropRate, Name="enc_drop3")
    ];
    net = dlnetwork(layers);
end

% -------------------------------------------------------------------------
% buildHead: 2-block prediction head for one frequency stage
%   inputCh = encFilters(end) for all stages (128 ch, no quota)
% -------------------------------------------------------------------------
function net = buildHead(inputCh, filterSize, filters, dropRate, stageIdx)
    tag = sprintf("h%d", stageIdx);
    layers = [
        sequenceInputLayer(inputCh, Name=tag+"_input")

        convolution1dLayer(filterSize, filters(1), Padding="same", Name=tag+"_conv1")
        batchNormalizationLayer(Name=tag+"_bn1")
        preluLayer(Name=tag+"_prelu1")
        dropoutLayer(dropRate, Name=tag+"_drop1")

        convolution1dLayer(filterSize, filters(2), Padding="same", Name=tag+"_conv2")
        batchNormalizationLayer(Name=tag+"_bn2")
        preluLayer(Name=tag+"_prelu2")
        dropoutLayer(dropRate, Name=tag+"_drop2")

        convolution1dLayer(1, 1, Padding="same", Name=tag+"_out")
    ];
    net = dlnetwork(layers);
end

% -------------------------------------------------------------------------
% modelLoss: forward pass + loss + gradients (called inside dlfeval)
%   No quota gates: every head receives only encoder output.
% -------------------------------------------------------------------------
function [loss, grads, encState, headStates, stageLossVals] = ...
        modelLoss(encNet, headNets, dlX, dlY, weights)

    nStages = numel(headNets);

    [feat, encState] = forward(encNet, dlX);   % [encCh x B x T]

    headStates    = cell(nStages, 1);
    stageLossVals = zeros(1, nStages);
    loss = dlarray(single(0));

    for s = 1:nStages
        % All heads receive encoder output only — no cross-stage conditioning
        headIn = feat;

        [pred, headStates{s}] = forward(headNets{s}, headIn);

        diff  = pred - dlY(s, :, :);
        sLoss = mean(diff.^2, 'all');
        loss  = loss + weights(s) * sLoss;
        stageLossVals(s) = double(extractdata(sLoss));
    end

    % Collect learnables for dlgradient: enc + nStages heads
    nNets = 1 + nStages;
    learnablesList = cell(1, nNets);
    learnablesList{1} = encNet.Learnables;
    for s = 1:nStages
        learnablesList{1 + s} = headNets{s}.Learnables;
    end

    grads = cell(1, nNets);
    [grads{:}] = dlgradient(loss, learnablesList{:});
end

% -------------------------------------------------------------------------
% evaluateValidation: compute val loss using predict (inference mode)
% -------------------------------------------------------------------------
function [valLoss, valStageLosses] = evaluateValidation( ...
        encNet, headNets, X_val_n, Y_val_n, ...
        weights, nt, Nval, miniBatchSize, useGPU)

    nStages = numel(headNets);
    valStageLosses = zeros(1, nStages);
    numValBatches = ceil(Nval / miniBatchSize);

    for b = 1:numValBatches
        si = (b-1)*miniBatchSize + 1;
        ei = min(b*miniBatchSize, Nval);
        bSize = ei - si + 1;

        xb = reshape(X_val_n(:, si:ei).', [1, bSize, nt]);
        yb = permute(Y_val_n(:, si:ei, :), [3 2 1]);
        dlX = dlarray(xb, 'CBT');
        dlY = dlarray(yb, 'CBT');
        if useGPU
            dlX = gpuArray(dlX);
            dlY = gpuArray(dlY);
        end

        feat = predict(encNet, dlX);

        for s = 1:nStages
            pred = predict(headNets{s}, feat);

            diff = pred - dlY(s, :, :);
            mse  = double(extractdata(mean(diff.^2, 'all')));
            valStageLosses(s) = valStageLosses(s) + mse * bSize;
        end
    end

    valStageLosses = valStageLosses / Nval;
    valLoss = sum(weights .* valStageLosses);
end

% -------------------------------------------------------------------------
% countParams: count total learnable elements in a dlnetwork
% -------------------------------------------------------------------------
function n = countParams(net)
    n = 0;
    for i = 1:height(net.Learnables)
        n = n + numel(net.Learnables.Value{i});
    end
end

% -------------------------------------------------------------------------
% applyL2: add L2 weight decay to gradients
% -------------------------------------------------------------------------
function gradTbl = applyL2(gradTbl, paramTbl, l2Reg)
    for i = 1:height(gradTbl)
        gradTbl.Value{i} = gradTbl.Value{i} + l2Reg * paramTbl.Value{i};
    end
end

% -------------------------------------------------------------------------
% canUseGPU: safe GPU availability check
% -------------------------------------------------------------------------
function tf = canUseGPU()
    try
        tf = (gpuDeviceCount("available") > 0);
    catch
        tf = false;
    end
end
