% run_cutJumpTrain2
% global dropbox;
% global config;
% 
% gbvsPath = fullfile(gdrive, 'Software', 'Saliency', 'gbvs');
% addpath(gbvsPath);
% 
% faceDetectorPath = fullfile(dropbox, 'Software', 'face_detector_adobe', 'win64new');
% addpath(faceDetectorPath);

%% settings
uncVideoRoot = fullfile(diemDataRoot, 'video_unc');
facesFile = fullfile(uncVideoRoot, '00_faces.mat');
humansFile = fullfile(uncVideoRoot, '00_humans.mat');
saliencyFile = fullfile(uncVideoRoot, '00_saliency.mat');
motionFile = fullfile(uncVideoRoot, '00_motion.mat');
gazeFile = fullfile(uncVideoRoot, '00_gaze.mat');
modelFile = fullfile(uncVideoRoot, '00_trained_model.mat');
saliencyRoot = fullfile(diemDataRoot, 'cuts_saliency');
motionRoot = fullfile(diemDataRoot, 'cuts_motion');
gazeRoot = fullfile(diemDataRoot, 'cuts_gaze');

cutTh = 10; % frames to consider cuts as the same
cutBefore = -1; % frames to sample before cut
cutAfter = 15; % frames to sample after cut

% for candidates
options.nonMaxSuprRad = 2;
options.humanTh = 1;
options.useGaze = true;
% for features
options.useLabel = true;
options.motionScales = [0, 2, 4]; % point, 5x5, 9x9 windows
options.saliencyScales = [0, 2, 4]; % point, 5x5, 9x9 windows
options.sigmaScale = 0.5;
options.gazeThreshold = 10;
options.nSample = 10;
options.posPer = 0.1;
options.negPer = 0.3;

useCenter = true;

% training set
trainIdx = [28, 34, 41, 65];

s = load(facesFile);
videos = s.videos;
faces = s.faces;
s = load(humansFile);
humans = s.humans;
s = load(saliencyFile);
saliency = s.saliency;
s = load(motionFile);
motion = s.motion;
s = load(gazeFile);
gaze = s.gaze;
clear s;

%% gather features
features = [];
labels = [];
curFeat = 1;

for i = 1:length(trainIdx)
    iv = trainIdx(i);
    fprintf('Extracting features from %s...\n', videos{iv}); tic;
    
    nc = length(faces{iv}.cuts);
    
    % load gaze, motion, saliency
    s = load(fullfile(gazeRoot, sprintf('%s.mat', videos{iv})));
    gaze = s.gaze;
    clear s;
    s = load(fullfile(saliencyRoot, sprintf('%s.mat', videos{iv})));
    saliency = s.saliency;
    clear s;
    s = load(fullfile(motionRoot, sprintf('%s.mat', videos{iv})));
    motion = s.motion;
    clear s;

    for ic = 1:nc
        fprintf('\tCut %d/%d\n', ic, nc);

        if ((faces{iv}.cuts(ic) + cutBefore >= 3) && (faces{iv}.cuts(ic) + cutAfter <= faces{iv}.length) && ~isempty(gaze.before{ic}) && ~isempty(gaze.after{ic}) && ~isempty(saliency.before{ic}) && ~isempty(saliency.after{ic}))
            % source frame
            [srcPts, srcScore, ~] = sourceCands(gaze.before{ic}.gazeProb, options);
            
            % destination frame
            maps = cat(3, motion.after{ic}.magnitude, saliency.after{ic}.saliency, gaze.after{ic}.gazeProb);
            [dstPts, dstScore, dstType] = jumpCands(faces{iv}.after{ic}, humans{iv}.after{ic}, useCenter, maps, options);
            
            % prepare frames
            srcFr.width = faces{iv}.width;
            srcFr.height = faces{iv}.height;
            srcFr.faces = faces{iv}.before{ic};
            srcFr.poselet_hit = humans{iv}.before{ic};
            srcFr.ofx = motion.before{ic}.ofx;
            srcFr.ofy = motion.before{ic}.ofy;
            srcFr.saliency = saliency.before{ic}.saliency;
            
            dstFr.width = faces{iv}.width;
            dstFr.height = faces{iv}.height;
            dstFr.faces = faces{iv}.after{ic};
            dstFr.poselet_hit = humans{iv}.after{ic};
            dstFr.ofx = motion.after{ic}.ofx;
            dstFr.ofy = motion.after{ic}.ofy;
            dstFr.saliency = saliency.after{ic}.saliency;

            % features
            [f, l] = jumpPairwiseFeatures(srcFr, srcPts, dstFr, dstPts, dstType, options);
            features = [features, f];
            labels = [labels, l];
            
        end
    end
    
    fprintf('%f sec\n', toc);
end

%% train model
features(isnan(features)) = 0;
rf = regRF_train(features', labels);
save(modelFile, 'rf', 'features', 'labels', 'options');
