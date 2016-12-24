% cvpr13_validateCands
%% settings
uncVideoRoot = fullfile(diemDataRoot, 'video_unc');
gazeDataRoot = fullfile(diemDataRoot, 'gaze');
visRoot = fullfileCreate(diemDataRoot, 'cvpr13', 'jump_test_v6');
outRoot = fullfileCreate(diemDataRoot, 'cvpr13');
modelFile = fullfile(uncVideoRoot, '00_trained_model_cvpr13_v5_3.mat'); % CVPR13

videoIdx = [6,8,10,11,12,14,15,16,34,42,44,48,53,54,55,59,70,74,83,84]; % used by Borji;

candScale = 2;
visVideo = false;

% cache settings
cache.root = fullfile(diemDataRoot, 'cache');
cache.frameRoot = fullfile(diemDataRoot, 'cache');
cache.featureRoot = fullfileCreate(cache.root, '00_features_v6');
cache.gazeRoot = fullfileCreate(cache.root, '00_gaze');
cache.renew = false; % use in case the preprocessing mechanism updated
cache.renewFeatures = false; % use in case the feature extraction is updated
cache.renewJumps = false; % recalculate the final result

% gaze settings
gazeParam.pointSigma = 10;

rocBins = 100;

% visualization
cmap = jet(256);
colors = [1 0 0;
    0 1 0;
    0 0 1;
    1 1 0;
    1 0 1;
    0 1 1];

%% loading
nv = length(videoIdx);
videos = videoListLoad(diemDataRoot);
load(fullfile(visRoot, '00_similarity.mat'));

[gbvsParam, ofParam, poseletModel] = configureDetectors();

%% run
dst = cell(nv, 1);
hit = cell(nv, 1);
for i = 1:nv
    iv = videoIdx(i);
    videoName = videos{iv};

    fprintf('Processing %s... ', videoName); tic;

    % load candidates
    s = load(fullfile(visRoot, sprintf('%s.mat', videoName)));
    [m, n, nfr] = size(s.predMaps);
    cands = s.cands;
    clear s;
    
    % load gaze data
    s = load(fullfile(cache.gazeRoot, sprintf('%s.mat', videoName)));
    gazeData = s.data;
    clear s;

    dst{i} = zeros(nfr, 1);
    hit{i} = zeros(nfr, 1);
    for ifr = 1:nfr
        [dst{i}(ifr), h] = candDistGaze(gazeData.points{ifr}, cands{ifr});
        hit{i}(ifr) = mean(h);
    end
    
    clear cands;
    
    fprintf('%f sec\n', toc);
end


%% results
dstMean = zeros(nv ,1);
hitMean = zeros(nv ,1);
for i = 1:nv
    dstMean(i) = mean(dst{i}(~isnan(dst{i})));
    hitMean(i) = mean(hit{i}(~isnan(hit{i})));
    fprintf('%s\t%fpx\t%f\n', videos{videoIdx(i)}, dstMean(i), hitMean(i));
end

dstAll = cell2mat(dst);
dstAll = dstAll(~isnan(dstAll));
dstAllMean = mean(dstAll);
hitAll = cell2mat(hit);
hitAll = hitAll(~isnan(hitAll));
hitAllMean = mean(hitAll);

fprintf('\nMean distance: %fpx\n', dstAllMean);
fprintf('Mean hit rate: %f%%\n', hitAllMean * 100);

fs = 18;
% [h, x] = hist(dstAll, 25);
% figure, bar(x, h / sum(h));
% set(gca, 'FontSize', fs);
% xlabel('Distance, [px]');
% ylabel('Percentage');
% set(findobj(gca,'Type','text'),'FontSize',fs);
% print('-dpng', fullfile(outRoot, 'min_dist.png'));

[h, x] = hist(hitAll, 25);
h = h / sum(h);
h = cumsum(h);
figure, bar(x, h);
set(gca, 'FontSize', fs);
xlabel('Hit rate');
ylabel('Cumulative histogram of frames per hit-rate value');
set(findobj(gca,'Type','text'),'FontSize',fs);
print('-dpng', fullfile(outRoot, 'hit_rate.png'));

%% visual comparison
% videoName = 'harry_potter_6_trailer_1280x544'; frameIdx = 514;
% videoName = 'BBC_life_in_cold_blood_1278x710'; frameIdx = 1384;
videoName = 'DIY_SOS_1280x712'; frameIdx = 1120;
candCol = [1 1 0];
gazeCol = [0 1 0];
lineWidth = 1;
ptsSz = 3;
outWidth = 800; 

[gbvsParam, ofParam, poseletModel] = configureDetectors();

% load model
s = load(modelFile, 'options');
% rf = s.rf;
options = s.options;
options.useLabel = false; % no need in label while testing
clear s;

% calculate
vr = VideoReader(fullfile(uncVideoRoot, sprintf('%s.avi', videoName)));
fr = preprocessFrames(vr, frameIdx, gbvsParam, ofParam, poseletModel, cache);
maps = cat(3, (fr.ofx.^2 + fr.ofy.^2), fr.saliency);
cands = jumpCandidates3(fr.faces, fr.poselet_hit, maps, options);

% visualize
frout = im2double(rgb2gray(fr.image));
frout = repmat(frout, [1 1 3]);

% candidates
frcand = repmat(reshape(candCol, [1 1 3]), [fr.height, fr.width, 1]);
for ic = 1:length(cands)
    msk = maskEllipse(fr.width, fr.height, cands{ic}.point, cands{ic}.cov, 1)';
    msk = bwperim(msk, 8);
    if (lineWidth > 1)
        msk = imdilate(msk, strel('disk', lineWidth));
    end
    msk = repmat(double(msk), [1 1 3]);
    
    frout = (1-msk) .* frout + msk .* frcand;
end

% gaze
s = load(fullfile(gazeDataRoot, sprintf('%s.mat', videoName)));
gazePts = s.gaze.data{frameIdx};
clear s;

gazePts = gazePts(~isnan(gazePts(:,1)), :);
msk = points2binaryMap(gazePts, [fr.width, fr.height]);
if (ptsSz > 1)
    msk = imdilate(msk, strel('diamond', lineWidth));
end
msk = repmat(double(msk), [1 1 3]);
frGaze = repmat(reshape(gazeCol, [1 1 3]), [fr.height, fr.width, 1]);
frout = (1-msk) .* frout + msk .* frGaze;

% save
frout = imresize(frout, [nan, outWidth]);
imwrite(frout, fullfile(outRoot, sprintf('%s_%d_valid.png', videoName, frameIdx)), 'png');
