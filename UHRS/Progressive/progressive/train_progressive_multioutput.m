% Title: train_progressive_multioutput.m
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
%             Conditioning  : quota gates, C_quota = 8, detachUpTo = 3
%             Head receptive field : 7.96 ms (no dilation)
%   Inputs  : dataset_train_val_test_UHRS.mat, written by the split script in common/.
%             The training settings are at the top of the file: miniBatchSize,
%             maxEpochs, initialLearnRate, l2Reg, C_quota, detachUpTo and the
%             random seed rng(42,'twister').
%   Outputs : CNN_progressive_multioutput_UHRS.mat, holding the trained encoder, heads and quota
%             gates together with the normalization constants, and a text file
%             with the training history, which lists the per-stage validation
%             loss every fifth epoch.
%   Note    : Running this script on a dataset built with the initial UHRS
%             generator gives the initial-wavelet model of Table 3.

%% train_progressive_multioutput.m
% Progressive multi-output 1-D CNN with custom training loop
% Input:  6400 Hz traces (2001 samples, dt=1e-5)
% Output: 10 LF traces [5000,3800,3000,1900,1500,1100,800,700,600,500] Hz
%
% Architecture:
%   Shared encoder (3 Conv blocks) -> 128-channel features
%   10 progressive prediction heads (2 Conv blocks + 1x1 output each):
%     Stage 1 (5000 Hz): encoder features only
%     Stage 2+ (3800..500 Hz): encoder + quota(prev predictions)
%   Quota gates: 1x1 Conv compressing previous predictions into C_quota ch
%
% Progressive gradient flow:
%   Stages 1-3: predictions detached when used as input to later stages
%               (their own loss still backprops normally)
%   Stages 4-10: full gradient flow through previous predictions
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
%     Stage 1 -> Y_train10 (5000 Hz), ..., Stage 10 -> Y_train1 (500 Hz)

% --- Network architecture ---
C_quota        = 8;         % quota gate output channels
detachUpTo     = 3;         % detach stages 1..3 from downstream gradient
filterSizeEnc  = 200;       % encoder conv filter size (large receptive field)
filterSizeHead = 100;       % head conv filter size
encFilters     = [128 64 128];   % encoder block filter counts
headFilters    = [64 32];        % head block filter counts
dropoutRate    = 0.15;

% --- Training hyperparameters ---
miniBatchSize      = 256;
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
data = load('dataset_train_val_test_UHRS.mat', vars{:});

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

%% ========================= NORMALIZATION ================================
% Input: global scalar normalization (same as single-output scripts)
muX  = mean(X_train(:));
sigX = std(X_train(:));
if sigX == 0, sigX = 1; end

X_train_n = single((X_train - muX) / sigX);
X_val_n   = single((X_val   - muX) / sigX);

% Targets: per-frequency normalisation
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
clear data Yt Yv;   % free memory

%% ========================= BUILD NETWORKS ===============================
fprintf('Building progressive network (%d stages, C_quota=%d)...\n', ...
    nStages, C_quota);
rng(42, 'twister');   % reproducible weight initialisation
encNet = buildEncoder(filterSizeEnc, encFilters, dropoutRate);

headNets = cell(nStages, 1);
headNets{1} = buildHead(encFilters(end), filterSizeHead, headFilters, dropoutRate, 1);
for s = 2:nStages
    headNets{s} = buildHead(encFilters(end) + C_quota, ...
        filterSizeHead, headFilters, dropoutRate, s);
end

quotaNets = cell(nStages, 1);
for s = 2:nStages
    quotaNets{s} = buildQuotaGate(s-1, C_quota, s);
end

% Parameter count
nParams = countParams(encNet);
for s = 1:nStages,  nParams = nParams + countParams(headNets{s});  end
for s = 2:nStages,  nParams = nParams + countParams(quotaNets{s}); end
fprintf('Total learnable parameters: %d (%.2f M)\n', nParams, nParams/1e6);

%% ========================= INIT TRAINING STATE ==========================
currentLR      = initialLearnRate;
iteration      = 0;
bestValLoss    = Inf;
bestNets       = struct();   % checkpoint of best model
patienceCounter = 0;
numBatches     = ceil(Ntrain / miniBatchSize);

% Adam optimiser states (one per sub-network)
avgGradEnc = [];  avgSqGradEnc = [];
avgGradHead  = cell(nStages, 1);  avgSqGradHead  = cell(nStages, 1);
avgGradQuota = cell(nStages, 1);  avgSqGradQuota = cell(nStages, 1);

% Active stage weights (start from base, adjusted on validation)
stageWeights = baseWeights;

% History logs
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
    % --- Shuffle training data ---
    shuffleIdx = randperm(Ntrain);
    epochLoss       = 0;
    epochStageLosses = zeros(1, nStages);

    for b = 1:numBatches
        iteration = iteration + 1;

        % --- Extract mini-batch ---
        si = (b-1)*miniBatchSize + 1;
        ei = min(b*miniBatchSize, Ntrain);
        bIdx  = shuffleIdx(si:ei);
        bSize = numel(bIdx);

        % Format: dlarray [C x B x T]  ('CBT')
        xb = reshape(X_train_n(:,bIdx).', [1, bSize, nt]);   % [1 x B x nt]
        yb = permute(Y_train_n(:,bIdx,:), [3 2 1]);          % [nStages x B x nt]
        dlX = dlarray(xb, 'CBT');
        dlY = dlarray(yb, 'CBT');
        if useGPU
            dlX = gpuArray(dlX);
            dlY = gpuArray(dlY);
        end

        % --- Forward + backward ---
        [loss, grads, encState, headStates, stageLossVals] = dlfeval( ...
            @modelLoss, encNet, headNets, quotaNets, ...
            dlX, dlY, stageWeights, detachUpTo);

        % --- Update batch-norm running statistics ---
        encNet.State = encState;
        for s = 1:nStages
            headNets{s}.State = headStates{s};
        end

        % --- Apply L2 weight decay to gradients ---
        % adamupdate does not accept L2 reg directly; we add l2Reg*w
        % to each gradient manually (equivalent to weight decay).
        allLearnables = cell(1, numel(grads));
        allLearnables{1} = encNet.Learnables;
        for s = 1:nStages
            allLearnables{1+s} = headNets{s}.Learnables;
        end
        for s = 2:nStages
            allLearnables{1+nStages+s-1} = quotaNets{s}.Learnables;
        end
        for gi = 1:numel(grads)
            grads{gi} = applyL2(grads{gi}, allLearnables{gi}, l2Reg);
        end

        % --- Adam update: encoder ---
        gIdx = 1;
        [encNet.Learnables, avgGradEnc, avgSqGradEnc] = adamupdate( ...
            encNet.Learnables, grads{gIdx}, avgGradEnc, avgSqGradEnc, ...
            iteration, currentLR);
        gIdx = gIdx + 1;

        % --- Adam update: heads ---
        for s = 1:nStages
            [headNets{s}.Learnables, avgGradHead{s}, avgSqGradHead{s}] = ...
                adamupdate(headNets{s}.Learnables, grads{gIdx}, ...
                avgGradHead{s}, avgSqGradHead{s}, ...
                iteration, currentLR);
            gIdx = gIdx + 1;
        end

        % --- Adam update: quota gates ---
        for s = 2:nStages
            [quotaNets{s}.Learnables, avgGradQuota{s}, avgSqGradQuota{s}] = ...
                adamupdate(quotaNets{s}.Learnables, grads{gIdx}, ...
                avgGradQuota{s}, avgSqGradQuota{s}, ...
                iteration, currentLR);
            gIdx = gIdx + 1;
        end

        % --- Accumulate epoch statistics ---
        bLoss = double(extractdata(loss));
        epochLoss        = epochLoss + bLoss * bSize;
        epochStageLosses = epochStageLosses + stageLossVals(:).' * bSize;
    end

    epochLoss        = epochLoss / Ntrain;
    epochStageLosses = epochStageLosses / Ntrain;

    % --- Progress printout ---
    if mod(epoch, 5) == 0 || epoch == 1
        fprintf('Ep %3d/%d | LR %.2e | Loss %.6f | Stages: %s\n', ...
            epoch, maxEpochs, currentLR, epochLoss, ...
            sprintf('%.4f ', epochStageLosses));
    end

    % --- Validation ---
    if mod(epoch, validationFreq) == 0
        [valLoss, valStageLosses] = evaluateValidation( ...
            encNet, headNets, quotaNets, X_val_n, Y_val_n, ...
            stageWeights, nt, Nval, miniBatchSize, useGPU);

        fprintf('  Val  Loss=%.6f | Stages: %s\n', valLoss, ...
            sprintf('%.4f ', valStageLosses));

        % Log
        epochLog(end+1)        = epoch;             %#ok<SAGROW>
        trainLossLog(end+1)    = epochLoss;          %#ok<SAGROW>
        valLossLog(end+1)      = valLoss;            %#ok<SAGROW>
        valStageLossLog(end+1,:) = valStageLosses;   %#ok<SAGROW>
        lrLog(end+1)           = currentLR;          %#ok<SAGROW>

        % Adaptive weight rebalancing:
        %   Stages with higher validation MSE receive proportionally more
        %   weight in the next training epoch.
        relLosses    = valStageLosses / mean(valStageLosses);
        stageWeights = baseWeights .* relLosses;
        stageWeights = stageWeights / mean(stageWeights) * mean(baseWeights);

        % Early stopping
        if valLoss < bestValLoss
            bestValLoss     = valLoss;
            patienceCounter = 0;
            % Save best checkpoint
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

    % --- LR decay ---
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
    encNet, headNets, quotaNets, X_val_n, Y_val_n, ...
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
    title('Total loss');

    subplot(2,1,2);
    plot(epochLog, valStageLossLog, 'LineWidth', 1);
    xlabel('Epoch'); ylabel('MSE (normalised)');
    legendStrs = arrayfun(@(f) sprintf('%d Hz', f), targetHz, 'UniformOutput', false);
    legend(legendStrs, 'Location','best');
    title('Per-stage validation MSE');
end

%% ========================= SAVE =========================================
saveFile = 'CNN_progressive_multioutput_UHRS.mat';
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
histFile = fullfile(pwd, 'training_history_progressive.txt');
fid = fopen(histFile, 'w');
if fid > 0
    fprintf(fid, 'TRAINING HISTORY - Progressive Multi-Output CNN\n');
    fprintf(fid, 'Date: %s\n', datestr(now));  %#ok<TNOW1,DATST>
    fprintf(fid, 'Target frequencies (Hz): %s\n', sprintf('%d ', targetHz));
    fprintf(fid, 'Ntrain=%d, Nval=%d, nt=%d, dt=%.6g\n', Ntrain, Nval, nt, dt);
    fprintf(fid, 'Encoder: filterSize=%d, filters=[%s], dropout=%.2f\n', ...
        filterSizeEnc, sprintf('%d ', encFilters), dropoutRate);
    fprintf(fid, 'Heads: filterSize=%d, filters=[%s], dropout=%.2f\n', ...
        filterSizeHead, sprintf('%d ', headFilters), dropoutRate);
    fprintf(fid, 'C_quota=%d, detachUpTo=%d\n', C_quota, detachUpTo);
    fprintf(fid, 'miniBatch=%d, maxEpochs=%d, LR=%.1e, L2=%.1e\n', ...
        miniBatchSize, maxEpochs, initialLearnRate, l2Reg);
    fprintf(fid, 'LR schedule: piecewise %.2fx every %d epochs\n', ...
        lrDropFactor, lrDropPeriod);
    fprintf(fid, 'Early stopping: freq=%d epochs, patience=%d checks\n', ...
        validationFreq, validationPatience);
    fprintf(fid, 'Stage weight lambda=%.2f\n', lambdaBase);
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
%   inputCh = encFilters(end) for stage 1, encFilters(end)+C_quota for 2+
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
% buildQuotaGate: 1x1 conv compressing (s-1) previous predictions -> C_quota
% -------------------------------------------------------------------------
function net = buildQuotaGate(numPrevPreds, cQuota, stageIdx)
    tag = sprintf("q%d", stageIdx);
    layers = [
        sequenceInputLayer(numPrevPreds, Name=tag+"_input")
        convolution1dLayer(1, cQuota, Padding="same", Name=tag+"_conv")
        preluLayer(Name=tag+"_prelu")
    ];
    net = dlnetwork(layers);
end

% -------------------------------------------------------------------------
% modelLoss: forward pass + loss + gradients (called inside dlfeval)
% -------------------------------------------------------------------------
function [loss, grads, encState, headStates, stageLossVals] = ...
        modelLoss(encNet, headNets, quotaNets, dlX, dlY, weights, detachUpTo)

    nStages = numel(headNets);

    % --- Encoder ---
    [feat, encState] = forward(encNet, dlX);   % [encCh x B x T]

    preds        = cell(nStages, 1);
    predsForInput = cell(nStages, 1);   % possibly detached
    headStates   = cell(nStages, 1);
    stageLossVals = zeros(1, nStages);

    loss = dlarray(single(0));

    for s = 1:nStages
        % Build head input
        if s == 1
            headIn = feat;
        else
            % Stack previous predictions along channel dim
            prevStack = cat(1, predsForInput{1:s-1});  % [(s-1) x B x T]
            % Quota gate: compress to C_quota channels
            quotaOut = forward(quotaNets{s}, prevStack); % [C_quota x B x T]
            headIn   = cat(1, feat, quotaOut);           % [(encCh+C_quota) x B x T]
        end

        % Head prediction
        [preds{s}, headStates{s}] = forward(headNets{s}, headIn);  % [1 x B x T]

        % Stage loss (MSE in normalised space)
        diff  = preds{s} - dlY(s, :, :);
        sLoss = mean(diff.^2, 'all');
        loss  = loss + weights(s) * sLoss;
        stageLossVals(s) = double(extractdata(sLoss));

        % Detach predictions for stages 1..detachUpTo so that downstream
        % losses do not backprop into these heads (own loss still does).
        if s <= detachUpTo
            predsForInput{s} = dlarray(extractdata(preds{s}), dims(preds{s}));
        else
            predsForInput{s} = preds{s};
        end
    end

    % --- Collect all learnables for dlgradient ---
    nNets = 1 + nStages + (nStages - 1);   % enc + heads + quotas(2..10)
    learnablesList = cell(1, nNets);
    learnablesList{1} = encNet.Learnables;
    for s = 1:nStages
        learnablesList{1 + s} = headNets{s}.Learnables;
    end
    for s = 2:nStages
        learnablesList{1 + nStages + s - 1} = quotaNets{s}.Learnables;
    end

    grads = cell(1, nNets);
    [grads{:}] = dlgradient(loss, learnablesList{:});
end

% -------------------------------------------------------------------------
% evaluateValidation: compute val loss using predict (inference mode)
% -------------------------------------------------------------------------
function [valLoss, valStageLosses] = evaluateValidation( ...
        encNet, headNets, quotaNets, X_val_n, Y_val_n, ...
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

        % predict = inference mode (stored BN stats, no dropout)
        feat = predict(encNet, dlX);
        predsForInput = cell(nStages, 1);

        for s = 1:nStages
            if s == 1
                headIn = feat;
            else
                prevStack = cat(1, predsForInput{1:s-1});
                quotaOut  = predict(quotaNets{s}, prevStack);
                headIn    = cat(1, feat, quotaOut);
            end

            pred = predict(headNets{s}, headIn);
            predsForInput{s} = pred;

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
% applyL2: add L2 weight decay (l2Reg * param) to each gradient entry
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
