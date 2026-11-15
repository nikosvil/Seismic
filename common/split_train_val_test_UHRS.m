% Title: split_train_val_test_UHRS.m
% Author: N. D. Vilanakis
%
% A progressive multi-output 1-D convolutional network for lower-frequency
% trace reconstruction of UHRS seismic data
% N. D. Vilanakis, N. Economou, A. Vafidis
% Computers and Geosciences
%
% User guide
%   Purpose : Splits the generated traces into training, validation and test
%             sets. The split is made by reflectivity group, so no reflectivity
%             series appears in more than one set.
%   Inputs  : dataset_multi_freq_UHRS.mat, written by the UHRS generator in common/.
%   Outputs : dataset_train_val_test_UHRS.mat, holding X_train, X_val, X_test,
%             the ten target bands Y_train1..10, Y_val1..10, Y_test1..10,
%             freq_list, t and dt. The sizes are 22500 / 3000 / 4500 traces.

S = load('dataset_multi_freq_UHRS.mat','freqmat500', 'freqmat600', 'freqmat800', 'freqmat1100',...
     'freqmat1500','freqmat2000','freqmat2600','freqmat3300',...
     'freqmat4500','freqmat5500','freqmat6400','g_trace','t','dt');

XHF = S.freqmat6400;       
YLF1 = S.freqmat500;
YLF2 = S.freqmat600;    
YLF3 = S.freqmat800;    
YLF4 = S.freqmat1100;    
YLF5 = S.freqmat1500;
YLF6 = S.freqmat2000;    
YLF7 = S.freqmat2600;    
YLF8 = S.freqmat3300;
YLF9 = S.freqmat4500;    
YLF10= S.freqmat5500;

g = S.g_trace(:);          

N = size(XHF,2);

assert(size(YLF1,2)  == N, 'HF and LF must have same #columns');
assert(size(YLF2,2)  == N, 'HF and LF must have same #columns');
assert(size(YLF3,2)  == N, 'HF and LF must have same #columns');
assert(size(YLF4,2)  == N, 'HF and LF must have same #columns');
assert(size(YLF5,2)  == N, 'HF and LF must have same #columns');
assert(size(YLF6,2)  == N, 'HF and LF must have same #columns');
assert(size(YLF7,2)  == N, 'HF and LF must have same #columns');
assert(size(YLF8,2)  == N, 'HF and LF must have same #columns');
assert(size(YLF9,2)  == N, 'HF and LF must have same #columns');
assert(size(YLF10,2) == N, 'HF and LF must have same #columns');
assert(numel(g) == N, 'g_trace length must equal #columns');

%% ------------------------------------------------------------------------
% GROUPED SPLIT BY g_trace
% 75% train / 10% validation / 15% test
% -------------------------------------------------------------------------
G = unique(g);

rng('default');
G = G(randperm(numel(G)));

nG = numel(G);

nTrainG = round(0.75 * nG);
nValG   = round(0.10 * nG);

% Make sure remainder goes to test
nTestG  = nG - nTrainG - nValG;

% Safety checks
assert(nTrainG > 0, 'Training group count is zero.');
assert(nValG   > 0, 'Validation group count is zero.');
assert(nTestG  > 0, 'Test group count is zero.');

Gtrain = G(1:nTrainG);
Gval   = G(nTrainG+1 : nTrainG+nValG);
Gtest  = G(nTrainG+nValG+1 : end);

trainIdx = find(ismember(g, Gtrain));
valIdx   = find(ismember(g, Gval));
testIdx  = find(ismember(g, Gtest));

% Extra safety: no overlap
assert(isempty(intersect(trainIdx, valIdx)),  'trainIdx and valIdx overlap!');
assert(isempty(intersect(trainIdx, testIdx)), 'trainIdx and testIdx overlap!');
assert(isempty(intersect(valIdx, testIdx)),   'valIdx and testIdx overlap!');

%% ------------------------------------------------------------------------
% BUILD TRAIN SET
% -------------------------------------------------------------------------
X_train   = XHF(:, trainIdx);
Y_train1  = YLF1(:, trainIdx);
Y_train2  = YLF2(:, trainIdx);
Y_train3  = YLF3(:, trainIdx);
Y_train4  = YLF4(:, trainIdx);
Y_train5  = YLF5(:, trainIdx);
Y_train6  = YLF6(:, trainIdx);
Y_train7  = YLF7(:, trainIdx);
Y_train8  = YLF8(:, trainIdx);
Y_train9  = YLF9(:, trainIdx);
Y_train10 = YLF10(:, trainIdx);

%% ------------------------------------------------------------------------
% BUILD VALIDATION SET
% -------------------------------------------------------------------------
X_val   = XHF(:, valIdx);
Y_val1  = YLF1(:, valIdx);
Y_val2  = YLF2(:, valIdx);
Y_val3  = YLF3(:, valIdx);
Y_val4  = YLF4(:, valIdx);
Y_val5  = YLF5(:, valIdx);
Y_val6  = YLF6(:, valIdx);
Y_val7  = YLF7(:, valIdx);
Y_val8  = YLF8(:, valIdx);
Y_val9  = YLF9(:, valIdx);
Y_val10 = YLF10(:, valIdx);

%% ------------------------------------------------------------------------
% BUILD TEST SET
% -------------------------------------------------------------------------
X_test   = XHF(:, testIdx);
Y_test1  = YLF1(:, testIdx);
Y_test2  = YLF2(:, testIdx);
Y_test3  = YLF3(:, testIdx);
Y_test4  = YLF4(:, testIdx);
Y_test5  = YLF5(:, testIdx);
Y_test6  = YLF6(:, testIdx);
Y_test7  = YLF7(:, testIdx);
Y_test8  = YLF8(:, testIdx);
Y_test9  = YLF9(:, testIdx);
Y_test10 = YLF10(:, testIdx);

t  = S.t;
dt = S.dt;

%% ------------------------------------------------------------------------
% SAVE
% -------------------------------------------------------------------------
save('dataset_train_val_test_UHRS.mat', ...
    'X_train','X_val','X_test', ...
    'Y_train1','Y_val1','Y_test1', ...
    'Y_train2','Y_val2','Y_test2', ...
    'Y_train3','Y_val3','Y_test3', ...
    'Y_train4','Y_val4','Y_test4', ...
    'Y_train5','Y_val5','Y_test5', ...
    'Y_train6','Y_val6','Y_test6', ...
    'Y_train7','Y_val7','Y_test7', ...
    'Y_train8','Y_val8','Y_test8', ...
    'Y_train9','Y_val9','Y_test9', ...
    'Y_train10','Y_val10','Y_test10', ...
    'trainIdx','valIdx','testIdx', ...
    'Gtrain','Gval','Gtest', ...
    't','dt', '-v7.3');

%% ------------------------------------------------------------------------
% REPORT
% -------------------------------------------------------------------------
fprintf('Grouped split done.\n');
fprintf('Train traces: %d\n', numel(trainIdx));
fprintf('Val traces:   %d\n', numel(valIdx));
fprintf('Test traces:  %d\n', numel(testIdx));

fprintf('Train groups: %d\n', numel(Gtrain));
fprintf('Val groups:   %d\n', numel(Gval));
fprintf('Test groups:  %d\n', numel(Gtest));

fprintf('Percentages by groups -> Train: %.2f%% | Val: %.2f%% | Test: %.2f%%\n', ...
    100*numel(Gtrain)/nG, 100*numel(Gval)/nG, 100*numel(Gtest)/nG);