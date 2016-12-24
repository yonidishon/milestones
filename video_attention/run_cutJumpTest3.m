% run_cutJumpTest3

%% settings
uncVideoRoot = fullfile(diemDataRoot, 'video_unc');
facesFile = fullfile(uncVideoRoot, '00_faces.mat');
% humansFile = fullfile(uncVideoRoot, '00_humans.mat');
% saliencyFile = fullfile(uncVideoRoot, '00_saliency.mat');
% motionFile = fullfile(uncVideoRoot, '00_motion.mat');
% gazeFile = fullfile(uncVideoRoot, '00_gaze.mat');
modelFile = fullfile(uncVideoRoot, '00_trained_model.mat');
% saliencyRoot = fullfile(diemDataRoot, 'cuts_saliency');
% motionRoot = fullfile(diemDataRoot, 'cuts_motion');
gazeRoot = fullfile(diemDataRoot, 'cuts_gaze');
visVideoRoot = fullfile(diemDataRoot, 'vis_jumptrack', 'jump_v3');

options.nonMaxSuprRad = 2;
options.humanTh = 1;
options.humanMinSzRat = 0.3; % filter human smaller that this of maximum
options.humanMidSz = 0.3; % below this size (X height) create one candidate
options.humanTrackRat = 0.7; % tracks with ration of human bounding box
options.useLabel = false;
options.candCovScale = 1;
% for features
options.motionScales = [0, 2, 4]; % point, 5x5, 9x9 windows
options.saliencyScales = [0, 2, 4]; % point, 5x5, 9x9 windows
options.sigmaScale = 0.5;
options.nSample = 10;
options.posPer = 0.1;
options.negPer = 0.3;
options.motionTh = 2;
options.topCandsNum = 5;
options.topCandsUse = 5; % number of tracked candidates
options.minTrackSize = 20; % minimum size of the side of tracking rectangle

% caching
cache.frameRoot = fullfile(diemDataRoot, 'cache');
cache.renew = false; % use in case the preprocessing mechanism updated

useCenter = true;
jumpFromCenter = true;

cutBefore = -1; % frames to use before cut
cutAfter = 15; % frames to update the map after cut
predRectTh = 0.03; % percentage for rectangle tracking
rect2DenseScale = 0.5; % in conversion of rectangles to map
rectSz = [20 20]; % rectangle around each candidate to track
nTrackRect = 5; % number of rectangles to track
nTrackDeadTh = 3; % if this number of tracks are dead - jump

% testing set
% testIdx = 40; % 34 - Harry Potter, 39 - Alice, 1:84 - all, 40 - Ice Age
testIdx = [13 21 28 40 41 67 80];
% testIdx = 34;

%% prepare
s = load(facesFile);
videos = s.videos;
faces = s.faces;
% s = load(humansFile);
% humans = s.humans;
% s = load(saliencyFile);
% saliency = s.saliency;
% s = load(motionFile);
% motion = s.motion;
% s = load(gazeFile);
% gaze = s.gaze;
clear s;

s = load(modelFile);
rf = s.rf;
clear s;

param = loadDefaultParams;
param.inputDir = []; % to use video reader
param.debug = 0;
param.scoreTh = 6; % normal: 2, failure: 5...10

% configure all detectors
[gbvsParam, ofParam, poseletModel] = configureDetectors();

%% run
nt = length(testIdx);
% jumps = cell(nt, 1);
% denseJumpsMap = cell(nt, 1);
sim = cell(nt, 1);

for i = 1:length(testIdx)
    iv = testIdx(i);
    fprintf('Testing on %s...\n', videos{iv}); tic;
    
    param.videoReader = VideoReader(fullfile(uncVideoRoot, sprintf('%s.avi', videos{iv})));
    cuts = faces{iv}.cuts;
    nc = length(cuts);
    m = faces{iv}.height;
    n = faces{iv}.width;

    cands = cell(nc, 1); % candidates per frame
    predMaps = zeros(m, n, nc); % dense maps
    
%     jumps{i} = cell(nc, 1);
    
%     % load gaze, motion, saliency
    s = load(fullfile(gazeRoot, sprintf('%s.mat', videos{iv})));
    gaze = s.gaze;
    clear s;
%     s = load(fullfile(saliencyRoot, sprintf('%s.mat', videos{iv})));
%     saliency = s.saliency;
%     clear s;
%     s = load(fullfile(motionRoot, sprintf('%s.mat', videos{iv})));
%     motion = s.motion;
%     clear s;

%     % jump to every pixel
%     [X, Y] = meshgrid(1:faces{iv}.width, 1:faces{iv}.height);
%     X = X(1:10:end, 1:10:end);
%     Y = Y(1:10:end, 1:10:end);
%     dstX = X(:);
%     dstY = Y(:);
    
    for ic = 1:nc
        if ((cuts(ic) + cutBefore >= 3) && (cuts(ic) + cutAfter <= faces{iv}.length) && ~isempty(gaze.before{ic}))
            % source frame
            if (jumpFromCenter)
                srcCands = {struct('point', [n/2, m/2], 'score', 1)}; % dummy source at center
            else % jump from gaze map
                [srcPts, srcScore, ~] = sourceCands(gaze.before{ic}.gazeProb, options);
                srcCands = {struct('point', srcPts, 'score', srcScore)};
            end
            
            dstCands = jumpPerform(srcCands, cuts(ic)+cutBefore, cuts(ic)+cutAfter, param, options, gbvsParam, ofParam, poseletModel, rf, cache);
            predMaps(:,:,ic) = candidate2map(dstCands, [n, m]);
            cands{ic} = dstCands;
        end
    end
    
    % compare
    frames = cuts+cutAfter;
    indFr = find(frames <= param.videoReader.numberOfFrames);
    sim{i} = visSideBySideCompareVideo(diemDataRoot, visVideoRoot, iv, frames(indFr), predMaps(:,:,indFr), cands(indFr));
    
    fprintf('%f sec\n', toc);
end

% save(fullfile(uncVideoRoot, '00_test_set1.mat'), 'jumps', 'denseJumpsMap', 'testIdx');
