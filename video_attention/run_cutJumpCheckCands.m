% run_cutJumpCheckCands

%% settings
uncVideoRoot = fullfile(diemDataRoot, 'video_unc');
facesFile = fullfile(uncVideoRoot, '00_faces.mat');
humansFile = fullfile(uncVideoRoot, '00_humans.mat');
saliencyRoot = fullfile(diemDataRoot, 'cuts_saliency');
motionRoot = fullfile(diemDataRoot, 'cuts_motion');
gazeRoot = fullfile(diemDataRoot, 'cuts_gaze');
visVideoRoot = fullfile(diemDataRoot, 'vis_jumptrack', 'cands_v2_sbs');

cutTh = 10; % frames to consider cuts as the same
cutBefore = -1; % frames to sample before cut
cutAfter = 15; % frames to sample after cut

% for candidates
options.nonMaxSuprRad = 2;
options.humanTh = 1;
options.humanMinSzRat = 0.3; % filter human smaller that this of maximum
options.humanMidSz = 0.3; % below this size (X height) create one candidate
options.humanTrackRat = 0.7; % tracks with ration of human bounding box
options.useGaze = true;
options.candCovScale = 1;
% for features
options.motionScales = [0, 2, 4]; % point, 5x5, 9x9 windows
options.saliencyScales = [0, 2, 4]; % point, 5x5, 9x9 windows
options.sigmaScale = 0.5;
options.gazeThreshold = 10;
options.nSample = 10;
options.posPer = 0.1;
options.negPer = 0.3;
options.motionTh = 2;
options.topCandsNum = 5;
options.topCandsUse = 5; % number of tracked candidates
options.minTrackSize = 20; % minimum size of the side of tracking rectangle

evalOp.sigma = 10;
gazeTh = 0.1;

useCenter = true;
renderVideo = true;
idx = 1:84;
% idx = 34;

%% prepare
s = load(facesFile);
videos = s.videos;
faces = s.faces;
s = load(humansFile);
humans = s.humans;
clear s;

%% gather features
nidx = length(idx);
dstCands = cell(nidx, 1);
dstMean = zeros(nidx, 1);

for i = 1:nidx
    iv = idx(i);
    fprintf('Checking candidates for %s... ', videos{iv}); tic;
    
    m = faces{iv}.height;
    n = faces{iv}.width;
    nc = length(faces{iv}.cuts);
    dstCands{i} = nan(nc, 1);
    
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

    mps = zeros(m, n, nc);
    gz = zeros(m, n, nc);
    for ic = 1:nc
        if ((faces{iv}.cuts(ic) + cutBefore >= 3) && (faces{iv}.cuts(ic) + cutAfter <= faces{iv}.length))
            % destination frame
            if (isempty(saliency.after{ic}))
                maps = cat(3, motion.after{ic}.magnitude, zeros(m, n));
            else
                maps = cat(3, motion.after{ic}.magnitude, saliency.after{ic}.saliency);
            end
            [cands, dstPts, dstScore, ~] = jumpCandidates(faces{iv}.after{ic}, humans{iv}.after{ic}, maps, options);
%             [dstPts, dstScore, ~] = jumpCands(faces{iv}.after{ic}, humans{iv}.after{ic}, useCenter, maps, options);
            
            pm = points2GaussMap(dstPts', dstScore', 0, [n, m], evalOp.sigma, true);
            if (~isempty(gaze.after{ic}))
                dstCands{i}(ic) = probMapSimilarity(dstPts, gaze.after{ic}.gazeProb, 'ishit', gazeTh);
                mps(:,:,ic) = pm;
                gz(:,:,ic) = gaze.after{ic}.gazeProb;
            end
        end
    end
    
    dstMean(i) = mean(dstCands{i}(~isnan(dstCands{i})));

    if (renderVideo)
        visCutJumpSideBySide(uncVideoRoot, visVideoRoot, iv, faces{iv}.cuts, videos, mps, gz);
    end

    fprintf('%f sec\n', toc);
end

save(fullfile(visVideoRoot, '00_data.mat'), 'dstCands', 'dstMean');

%% visualize
dstCandsAll = []; 
for i = 1:length(dstCands)
    dstCandsAll = [dstCandsAll; dstCands{i}(:)];
end

dstCandsAll = dstCandsAll(dstCandsAll > 0); 
[h, x] = hist(dstCandsAll, 10); 
figure; 
bar(x, h/sum(h)); 
xlabel('Coverage of gaze'); 
ylabel('Percentage'); 
title('Candidate statistics, DIEM cuts');
