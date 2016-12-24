% cvpr13_fig_jumps
%% settings
uncVideoRoot = fullfile(diemDataRoot, 'video_unc');
gazeDataRoot = fullfile(diemDataRoot, 'gaze');
visRoot = fullfileCreate(diemDataRoot, 'cvpr13', 'jump_test_v6');
outRoot = fullfileCreate(diemDataRoot, 'cvpr13');
modelFile = fullfile(uncVideoRoot, '00_trained_model_cvpr13_v5_3.mat'); % CVPR13

% videoIdx = 34; frIdx = 2150;
videoIdx = 21; frIdx = 565;

candScale = 2;

% cache settings
cache.root = fullfile(diemDataRoot, 'cache');
cache.frameRoot = fullfile(diemDataRoot, 'cache');
cache.featureRoot = fullfileCreate(cache.root, '00_features_v6');
cache.gazeRoot = fullfileCreate(cache.root, '00_gaze');
cache.renew = false; % use in case the preprocessing mechanism updated
cache.renewFeatures = false; % use in case the feature extraction is updated
cache.renewJumps = false; % recalculate the final result

% gaze settings
pointSigma = 10;

sourceType = 'rect';
before = -15;
outWidth = 800;

%% loading
videos = videoListLoad(diemDataRoot);
videoName = videos{videoIdx};
% load(fullfile(visRoot, '00_similarity.mat'));

[gbvsParam, ofParam, poseletModel] = configureDetectors();

% load gaze data
s = load(fullfile(gazeDataRoot, sprintf('%s.mat', videoName)));
gazeData = s.gaze.data;
clear s;

s = load(modelFile, 'options');
options = s.options;
clear s;

%% run
vr = VideoReader(fullfile(uncVideoRoot, sprintf('%s.avi', videoName)));
m = vr.Height;
n = vr.Width;

srcFr = preprocessFrames(vr, frIdx+before, gbvsParam, ofParam, poseletModel, cache);
dstFr = preprocessFrames(vr, frIdx, gbvsParam, ofParam, poseletModel, cache);

% source candidates
srcGazeMap = points2GaussMap(gazeData{frIdx+before}', ones(1, size(gazeData{frIdx+before}, 1)), 0, [n, m], pointSigma);
[srcCands, ~, ~, ~] = sourceCandidates(srcFr, srcGazeMap, options, sourceType);

% destination candidates
dstGazeMap = points2GaussMap(gazeData{frIdx}', ones(1, size(gazeData{frIdx}, 1)), 0, [n, m], pointSigma);
maps = cat(3, (dstFr.ofx.^2 + dstFr.ofy.^2), dstFr.saliency);
[dstCands, dstPts, dstScore, dstType] = jumpCandidates(dstFr.faces, dstFr.poselet_hit, maps, options);

% features
options.dstGroundTruth = dstGazeMap;
options.dstGroundTruthPts = gazeData{frIdx};
[f, d, l, jumps] = jumpPairwiseFeatures6(srcFr, srcCands, dstFr, dstCands, options);
% [f, d, l, jumps] = jumpPairwiseFeatures6(srcFr, srcCands, dstFr, dstCands, options, cache);

frout = visJumpsSbs(vr, frIdx+before, srcGazeMap, frIdx, dstGazeMap, jumps, options);
frout = imresize(frout, [nan, outWidth]);
imwrite(frout, fullfile(outRoot, sprintf('%s_%d_jump.png', videoName, frIdx)), 'png');
