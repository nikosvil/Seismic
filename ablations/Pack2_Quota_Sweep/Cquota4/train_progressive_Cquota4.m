% Title: train_progressive_Cquota4.m
% Author: N. D. Vilanakis
%
% A progressive multi-output 1-D convolutional network for lower-frequency
% trace reconstruction of UHRS seismic data
% N. D. Vilanakis, N. Economou, A. Vafidis
% Computers and Geosciences
%
% User guide
%   Purpose : One member of the Section 3.3, quota width sweep.
%             Varied quantity: quota gate width C_quota.
%             Everything else matches the published configuration.
%   Inputs  : dataset_train_val_test_UHRS.mat, written by
%             common/split_train_val_test_UHRS.m.
%   Outputs : the trained network as a .mat file and a text file with the
%             training history. The file names are set at the top.

%% train_progressive_Cquota0.m
% ABLATION PACK 2 — Quota gate size sweep: C_quota = 4
% C_quota=4: quota gate compresses (s-1) prev predictions to 4 channels.


%
% Input:  6400 Hz traces (2001 samples, dt=1e-5)
% Output: 10 LF traces [5500,4500,3300,2600,2000,1500,1100,800,600,500] Hz
%
% All other hyperparameters identical to train_progressive_multioutput.m.

clear; close all;

DO_PLOTS = false;
try, set(0,'DefaultFigureVisible','off'); end
try, set(groot,'defaultFigureVisible','off'); end

%% ========================= CONFIGURATION ================================
nStages       = 10;
targetHz      = [5500 4500 3300 2600 2000 1500 1100 800 600 500];
targetVarIdx  = [10   9    8    7    6    5    4   3   2   1  ];

% --- Network architecture ---
C_quota        = 4;        % ABLATION: quota gate output channels
detachUpTo     = 3;
filterSizeEnc  = 200;
filterSizeHead = 100;
encFilters     = [128 64 128];
headFilters    = [64 32];
dropoutRate    = 0.15;

% --- Training hyperparameters ---
miniBatchSize      = 64;
maxEpochs          = 300;
initialLearnRate   = 1e-4;
l2Reg              = 1e-4;
lrDropFactor       = 0.5;
lrDropPeriod       = 75;
validationFreq     = 5;
validationPatience = 25;

% --- Stage loss weights ---
lambdaBase  = 1.2;
baseWeights = lambdaBase .^ (0:nStages-1);
baseWeights = baseWeights / mean(baseWeights);

%% ========================= LOAD DATA ====================================
fprintf('Loading data...\n');
vars = {'X_train','X_val','t','dt'};
for s = 1:nStages
    vars{end+1} = sprintf('Y_train%d', targetVarIdx(s));  %#ok<SAGROW>
    vars{end+1} = sprintf('Y_val%d',   targetVarIdx(s));  %#ok<SAGROW>
end
data = load('dataset_train_val_test_UHRS.mat', vars{:});

X_train = data.X_train;
X_val   = data.X_val;
t       = data.t;
dt      = data.dt;
assert(isfield(data,'dt'), 'dt missing - dataset is invalid');

[nt, Ntrain] = size(X_train);
[~,  Nval]   = size(X_val);
fprintf('Ntrain=%d, Nval=%d, nt=%d, dt=%.6g\n', Ntrain, Nval, nt, dt);

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
fprintf('Building progressive network (%d stages, C_quota=%d)...\n', nStages, C_quota);

rng(42, 'twister');

encNet = buildEncoder(filterSizeEnc, encFilters, dropoutRate);

% C_quota=0: all heads receive encoder output only (no conditioning)
headNets = cell(nStages, 1);
if C_quota == 0
    for s = 1:nStages
        headNets{s} = buildHead(encFilters(end), filterSizeHead, headFilters, dropoutRate, s);
    end
    quotaNets = cell(nStages, 1);   % empty
else
    headNets{1} = buildHead(encFilters(end), filterSizeHead, headFilters, dropoutRate, 1);
    for s = 2:nStages
        headNets{s} = buildHead(encFilters(end) + C_quota, ...
            filterSizeHead, headFilters, dropoutRate, s);
    end
    quotaNets = cell(nStages, 1);
    for s = 2:nStages
        quotaNets{s} = buildQuotaGate(s-1, C_quota, s);
    end
end

nParams = countParams(encNet);
for s = 1:nStages,  nParams = nParams + countParams(headNets{s});  end
if C_quota > 0
    for s = 2:nStages,  nParams = nParams + countParams(quotaNets{s}); end
end
fprintf('Total learnable parameters: %d (%.2f M)\n', nParams, nParams/1e6);

%% ========================= INIT TRAINING STATE ==========================
currentLR      = initialLearnRate;
iteration      = 0;
bestValLoss    = Inf;
bestNets       = struct();
patienceCounter = 0;
numBatches     = ceil(Ntrain / miniBatchSize);

avgGradEnc   = [];  avgSqGradEnc   = [];
avgGradHead  = cell(nStages, 1);  avgSqGradHead  = cell(nStages, 1);
avgGradQuota = cell(nStages, 1);  avgSqGradQuota = cell(nStages, 1);

stageWeights = baseWeights;

epochLog        = [];
trainLossLog    = [];
valLossLog      = [];
valStageLossLog = [];
lrLog           = [];

%% ========================= TRAINING LOOP ================================
fprintf('Starting training (%d epochs, batch=%d, LR=%.1e)...\n', ...
    maxEpochs, miniBatchSize, currentLR);
tTrainStart   = tic;
earlyStopEpoch = maxEpochs;

for epoch = 1:maxEpochs
    shuffleIdx = randperm(Ntrain);
    epochLoss        = 0;
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
            @modelLoss, encNet, headNets, quotaNets, ...
            dlX, dlY, stageWeights, detachUpTo);

        encNet.State = encState;
        for s = 1:nStages
            headNets{s}.State = headStates{s};
        end

        % L2 weight decay
        quotaIdxList = find(~cellfun(@isempty, quotaNets));
        nNets_train  = 1 + nStages + numel(quotaIdxList);
        allLearnables = cell(1, nNets_train);
        allLearnables{1} = encNet.Learnables;
        for s = 1:nStages
            allLearnables{1+s} = headNets{s}.Learnables;
        end
        for k = 1:numel(quotaIdxList)
            allLearnables{1+nStages+k} = quotaNets{quotaIdxList(k)}.Learnables;
        end
        for gi = 1:numel(grads)
            grads{gi} = applyL2(grads{gi}, allLearnables{gi}, l2Reg);
        end

        gIdx = 1;
        [encNet.Learnables, avgGradEnc, avgSqGradEnc] = adamupdate( ...
            encNet.Learnables, grads{gIdx}, avgGradEnc, avgSqGradEnc, ...
            iteration, currentLR);
        gIdx = gIdx + 1;

        for s = 1:nStages
            [headNets{s}.Learnables, avgGradHead{s}, avgSqGradHead{s}] = ...
                adamupdate(headNets{s}.Learnables, grads{gIdx}, ...
                avgGradHead{s}, avgSqGradHead{s}, ...
                iteration, currentLR);
            gIdx = gIdx + 1;
        end

        for k = 1:numel(quotaIdxList)
            s = quotaIdxList(k);
            [quotaNets{s}.Learnables, avgGradQuota{s}, avgSqGradQuota{s}] = ...
                adamupdate(quotaNets{s}.Learnables, grads{gIdx}, ...
                avgGradQuota{s}, avgSqGradQuota{s}, ...
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
            encNet, headNets, quotaNets, X_val_n, Y_val_n, ...
            stageWeights, nt, Nval, miniBatchSize, useGPU);

        fprintf('  Val  Loss=%.6f | Stages: %s\n', valLoss, ...
            sprintf('%.4f ', valStageLosses));

        epochLog(end+1)          = epoch;           %#ok<SAGROW>
        trainLossLog(end+1)      = epochLoss;        %#ok<SAGROW>
        valLossLog(end+1)        = valLoss;          %#ok<SAGROW>
        valStageLossLog(end+1,:) = valStageLosses;   %#ok<SAGROW>
        lrLog(end+1)             = currentLR;        %#ok<SAGROW>

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
    encNet, headNets, quotaNets, X_val_n, Y_val_n, ...
    ones(1, nStages), nt, Nval, miniBatchSize, useGPU);

for s = 1:nStages
    rmse_phys = sqrt(valStageMSE(s)) * sigY(s);
    fprintf('  Stage %2d (%4d Hz): RMSE = %.6g\n', s, targetHz(s), rmse_phys);
end

%% ========================= OPTIONAL PLOTS ===============================
if DO_PLOTS && ~isempty(epochLog)
    figure('Name', sprintf('Training curves — C\\_quota=%d', C_quota));
    subplot(2,1,1);
    plot(epochLog, trainLossLog, 'b-', epochLog, valLossLog, 'r-', 'LineWidth',1);
    xlabel('Epoch'); ylabel('Weighted MSE'); legend('Train','Val');
    title(sprintf('Total loss — C\\_quota=%d', C_quota));
    subplot(2,1,2);
    plot(epochLog, valStageLossLog, 'LineWidth', 1);
    xlabel('Epoch'); ylabel('MSE (normalised)');
    legend(arrayfun(@(f) sprintf('%d Hz',f), targetHz, 'UniformOutput',false), 'Location','best');
    title('Per-stage validation MSE');
end

%% ========================= SAVE =========================================
saveFile = sprintf('CNN_progressive_Cquota%d_UHRS.mat', C_quota);
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
histFile = fullfile(pwd, sprintf('training_history_Cquota%d.txt', C_quota));
fid = fopen(histFile, 'w');
if fid > 0
    fprintf(fid, 'TRAINING HISTORY - Progressive CNN | Pack2 C_quota Sweep\n');
    fprintf(fid, 'Ablation: Pack2 C_quota sweep | C_quota=%d\n', C_quota);
    fprintf(fid, 'Date: %s\n', datestr(now));  %#ok<TNOW1,DATST>
    fprintf(fid, 'Target frequencies (Hz): %s\n', sprintf('%d ', targetHz));
    fprintf(fid, 'Ntrain=%d, Nval=%d, nt=%d, dt=%.6g\n', Ntrain, Nval, nt, dt);
    fprintf(fid, 'C_quota=%d, detachUpTo=%d\n', C_quota, detachUpTo);
    fprintf(fid, 'Total parameters: %d (%.2f M)\n', nParams, nParams/1e6);
    fprintf(fid, 'Training time: %.1f sec (%.1f min)\n', trainTimeSec, trainTimeSec/60);
    fprintf(fid, 'Mean epoch time: %.3f sec\n', trainTimeSec / earlyStopEpoch);
    if isfield(bestNets, 'epoch')
        fprintf(fid, 'Best val loss: %.6f (epoch %d)\n', bestValLoss, bestNets.epoch);
    end
    if earlyStopEpoch < maxEpochs
        fprintf(fid, 'Early stopping triggered at epoch %d\n', earlyStopEpoch);
    end
    fprintf(fid, '\nPer-stage val RMSE (physical units):\n');
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

function net = buildQuotaGate(numPrevPreds, cQuota, stageIdx)
    tag = sprintf("q%d", stageIdx);
    layers = [
        sequenceInputLayer(numPrevPreds, Name=tag+"_input")
        convolution1dLayer(1, cQuota, Padding="same", Name=tag+"_conv")
        preluLayer(Name=tag+"_prelu")
    ];
    net = dlnetwork(layers);
end

function [loss, grads, encState, headStates, stageLossVals] = ...
        modelLoss(encNet, headNets, quotaNets, dlX, dlY, weights, detachUpTo)

    nStages = numel(headNets);
    [feat, encState] = forward(encNet, dlX);

    predsForInput = cell(nStages, 1);
    headStates    = cell(nStages, 1);
    stageLossVals = zeros(1, nStages);
    loss = dlarray(single(0));

    for s = 1:nStages
        if s == 1 || isempty(quotaNets{s})
            headIn = feat;
        else
            prevStack = cat(1, predsForInput{1:s-1});
            quotaOut  = forward(quotaNets{s}, prevStack);
            headIn    = cat(1, feat, quotaOut);
        end

        [pred, headStates{s}] = forward(headNets{s}, headIn);

        diff  = pred - dlY(s, :, :);
        sLoss = mean(diff.^2, 'all');
        loss  = loss + weights(s) * sLoss;
        stageLossVals(s) = double(extractdata(sLoss));

        if s <= detachUpTo
            predsForInput{s} = dlarray(extractdata(pred), dims(pred));
        else
            predsForInput{s} = pred;
        end
    end

    quotaIdxList = find(~cellfun(@isempty, quotaNets));
    nNets = 1 + nStages + numel(quotaIdxList);
    learnablesList = cell(1, nNets);
    learnablesList{1} = encNet.Learnables;
    for s = 1:nStages
        learnablesList{1+s} = headNets{s}.Learnables;
    end
    for k = 1:numel(quotaIdxList)
        learnablesList{1+nStages+k} = quotaNets{quotaIdxList(k)}.Learnables;
    end

    grads = cell(1, nNets);
    [grads{:}] = dlgradient(loss, learnablesList{:});
end

function [valLoss, valStageLosses] = evaluateValidation( ...
        encNet, headNets, quotaNets, X_val_n, Y_val_n, ...
        weights, nt, Nval, miniBatchSize, useGPU)

    nStages = numel(headNets);
    valStageLosses = zeros(1, nStages);
    numValBatches  = ceil(Nval / miniBatchSize);

    for b = 1:numValBatches
        si = (b-1)*miniBatchSize + 1;
        ei = min(b*miniBatchSize, Nval);
        bSize = ei - si + 1;

        xb = reshape(X_val_n(:, si:ei).', [1, bSize, nt]);
        yb = permute(Y_val_n(:, si:ei, :), [3 2 1]);
        dlX = dlarray(xb, 'CBT');
        dlY = dlarray(yb, 'CBT');
        if useGPU, dlX = gpuArray(dlX); dlY = gpuArray(dlY); end

        feat = predict(encNet, dlX);
        predsForInput = cell(nStages, 1);

        for s = 1:nStages
            if s == 1 || isempty(quotaNets{s})
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

function n = countParams(net)
    n = 0;
    for i = 1:height(net.Learnables)
        n = n + numel(net.Learnables.Value{i});
    end
end

function gradTbl = applyL2(gradTbl, paramTbl, l2Reg)
    for i = 1:height(gradTbl)
        gradTbl.Value{i} = gradTbl.Value{i} + l2Reg * paramTbl.Value{i};
    end
end

function tf = canUseGPU()
    try
        tf = (gpuDeviceCount("available") > 0);
    catch
        tf = false;
    end
end
