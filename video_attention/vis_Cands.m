% vis_Cands

warning('off', 'stats:gmdistribution:FailedToConverge');

%% settings
uncVideoRoot = fullfile(diemDataRoot, 'video_unc');
modelFile = fullfile(uncVideoRoot, '00_trained_model.mat');
gazeRoot = fullfile(diemDataRoot, 'cuts_gaze');
saliencyRoot = fullfile(diemDataRoot, 'cuts_saliency');
motionRoot = fullfile(diemDataRoot, 'cuts_motion');
facesFile = fullfile(uncVideoRoot, '00_faces.mat');
humansFile = fullfile(uncVideoRoot, '00_humans.mat');
visVideoRoot = fullfile(diemDataRoot, 'vis_jumptrack', 'cands_v2');

% for candidates
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
% options.posPer = 0.1;
% options.negPer = 0.3;
options.motionTh = 2;
options.topCandsNum = 5;

evalOp.isDense = false;
evalOp.sigma = 10;
evalOp.pThresh = 0;

useCenter = true;

cutBefore = -1; % frames to use before cut
cutAfter = 15; % frames to update the map after cut
predRectTh = 0.03; % percentage for rectangle tracking
rect2DenseScale = 0.5; % in conversion of rectangles to map

gazeTh = 0.1;

% idx = 40; % ice age
idx = 34; % harry potter
% idx = 21; % chilly plasters
% idx = 1:84;

% caching
cache.frameRoot = fullfile(diemDataRoot, 'cache');
cache.renew = false; % use in case the preprocessing mechanism updated

%% prepare
s = load(facesFile);
videos = s.videos;
faces = s.faces;
clear s;

% configure all detectors
[gbvsParam, ofParam, poseletModel] = configureDetectors();

%% run
nidx = length(idx);
dstCands = cell(nidx, 1);
dstMean = zeros(nidx, 1);

for i = 1:nidx
    iv = idx(i);
    fprintf('Checking candidates for %s... ', videos{iv}); tic;
    
    vr = VideoReader(fullfile(uncVideoRoot, sprintf('%s.avi', videos{iv})));
    m = faces{iv}.height;
    n = faces{iv}.width;
    
    % find the cuts
    cuts = faces{iv}.cuts;
    nc = length(cuts);
    
%     for ic = 16
    for ic = 1:nc
        if (cuts(ic)+cutAfter > vr.NumberOfFrames), break; end;
        dstFr = preprocessFrames(vr, cuts(ic)+cutAfter, gbvsParam, ofParam, poseletModel, cache);
        
        maps = cat(3, (dstFr.ofx.^2 + dstFr.ofy.^2), dstFr.saliency);
        [dCands, dPts, dScore, dType] = jumpCandidates(dstFr.faces, dstFr.poselet_hit, maps, options);

        fr = read(vr, cuts(ic)+cutAfter);
        frOut = visCandidates(fr, dCands);
        imwrite(frOut, fullfile(visVideoRoot, sprintf('%s_cut_%d.png', videos{iv}, ic)), 'png');
    end
    
    fprintf('%f sec\n', toc);
end
