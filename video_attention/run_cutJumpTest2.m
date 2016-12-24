% run_cutJumpTest2

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
visVideoRoot = fullfile(diemDataRoot, 'vis_cut_jump', 'RF_sparse_sbs');

cutTh = 10; % frames to consider cuts as the same
cutBefore = -1; % frames to sample before cut
cutAfter = 15; % frames to sample after cut

% for candidates
options.nonMaxSuprRad = 2;
options.humanTh = 1;
options.useLabel = false;
% for features
options.motionScales = [0, 2, 4]; % point, 5x5, 9x9 windows
options.saliencyScales = [0, 2, 4]; % point, 5x5, 9x9 windows
options.sigmaScale = 0.5;
options.nSample = 10;
options.posPer = 0.1;
options.negPer = 0.3;

evalOp.isDense = false;
evalOp.denseStep = 10;
evalOp.sigma = 10;
evalOp.pThresh = 0;

useCenter = true;

% testing set
% testIdx = 40; % 34 - Harry Potter, 39 - Alice, 1:84 - all, 40 - Ice Age
testIdx = [13 21 28 34 40 41 67 80];

%% prepare
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

s = load(modelFile);
rf = s.rf;
clear s;

%% gather features
nt = length(testIdx);
jumps = cell(nt, 1);
denseJumpsMap = cell(nt, 1);
cjSim = cell(nt, 1);
ssSim = cell(nt, 1);

for i = 1:length(testIdx)
    iv = testIdx(i);
    fprintf('Testing on %s...\n', videos{iv}); tic;
    
%     vr = VideoReader(fullfile(uncVideoRoot, sprintf('%s.avi', videos{iv})));
    nc = length(faces{iv}.cuts);
    m = faces{iv}.height;
    n = faces{iv}.width;
    jumps{i} = cell(nc, 1);
    
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

%     % jump to every pixel
%     [X, Y] = meshgrid(1:faces{iv}.width, 1:faces{iv}.height);
%     X = X(1:10:end, 1:10:end);
%     Y = Y(1:10:end, 1:10:end);
%     dstX = X(:);
%     dstY = Y(:);
    
    for ic = 1:nc
%         fprintf('\tCut %d/%d\n', ic, nc);

        if ((faces{iv}.cuts(ic) + cutBefore >= 3) && (faces{iv}.cuts(ic) + cutAfter <= faces{iv}.length) && ~isempty(gaze.before{ic}))
            % source frame
            [srcPts, srcScore, ~] = sourceCands(gaze.before{ic}.gazeProb, options);
            nSrc = size(srcPts, 1);
            
            if (isempty(saliency.before{ic}))
                srcFr.saliency = zeros(m, n);
            else
                srcFr.saliency = saliency.before{ic}.saliency;
            end
            
            % destination frame
            if (isempty(saliency.after{ic}))
                dstFr.saliency = zeros(m, n);
            else
                dstFr.saliency = saliency.after{ic}.saliency;
            end
            maps = cat(3, motion.after{ic}.magnitude, dstFr.saliency);
            [dstPts, dstScore, dstType] = jumpCands(faces{iv}.after{ic}, humans{iv}.after{ic}, useCenter, maps, options);
            nDst = size(dstPts, 1);
            
            % prepare frames
            srcFr.width = n;
            srcFr.height = m;
            srcFr.faces = faces{iv}.before{ic};
            srcFr.poselet_hit = humans{iv}.before{ic};
            srcFr.ofx = motion.before{ic}.ofx;
            srcFr.ofy = motion.before{ic}.ofy;
%             srcFr.saliency = saliency.before{ic}.saliency;
            
            dstFr.width = n;
            dstFr.height = m;
            dstFr.faces = faces{iv}.after{ic};
            dstFr.poselet_hit = humans{iv}.after{ic};
            dstFr.ofx = motion.after{ic}.ofx;
            dstFr.ofy = motion.after{ic}.ofy;
%             dstFr.saliency = saliency.after{ic}.saliency;

            % features
            [features, ~] = jumpPairwiseFeatures(srcFr, srcPts, dstFr, dstPts, dstType, options);
            jumps{i}{ic}.src = srcPts';
            jumps{i}{ic}.dst = dstPts';
            jumps{i}{ic}.srcW = srcScore(:)';
            jumps{i}{ic}.p = zeros(nSrc, nDst);

            % pairwise jumps
            for isrc = 1:nSrc % predict for every source
                for idst = 1:nDst
                    lbl = regRF_predict(features(:, nDst*(isrc-1)+idst)', rf); % RF reg
                    jumps{i}{ic}.p(isrc, idst) = lbl;
                end
            end
        end
    end
    
    % create dense map
    denseJumpsMap{i} = evalCutJump([faces{iv}.width, faces{iv}.height], jumps{i}, evalOp);
    [cjSim{i}, ssSim{i}] = visCutJumpVideo(diemDataRoot, visVideoRoot, iv, denseJumpsMap{i});
    
    fprintf('%f sec\n', toc);
end

% cd(curDir);

save(fullfile(uncVideoRoot, '00_test_set1.mat'), 'jumps', 'denseJumpsMap', 'testIdx');

%% visualize
% visCutJumps(jumps, testIdx, {faces{testIdx(1)}.cuts}, videos, uncVideoRoot, fullfile(diemDataRoot, 'vis_cut_jump', 'RF_sparse_test'), false);
