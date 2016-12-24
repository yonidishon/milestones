% cvpr13_fig_visRes
clear outWidth;

%% settings
uncVideoRoot = fullfile(diemDataRoot, 'video_unc');
gazeDataRoot = fullfile(diemDataRoot, 'gaze');
visRoot = fullfileCreate(diemDataRoot, 'cvpr13', 'jump_test_v6');
outRoot = fullfileCreate(diemDataRoot, 'cvpr13');
modelFile = fullfile(uncVideoRoot, '00_trained_model_cvpr13_v5_3.mat'); % CVPR13

% videoIdx = 34; frIdx = 2180;
% videoIdx = 34; frIdx = 2538;
% videoIdx = 34; frIdx = 2422; % harry potter
% videoIdx = 6; frIdx = 240; % BBC
% videoIdx = 6; frIdx = 250;
% videoIdx = 70; frIdx = 750;
% videoIdx = 21; frIdx = 565;
videoIdx = 10; frIdx = 534; % DIY

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
measures = {'chisq', 'auc'};
methods = {'ours', 'self', 'center', 'gbvs', 'pqft', 'hou'};

ncol = 256;
cmap = jet(ncol);
outWidth = 1280;

%% loading
videos = videoListLoad(diemDataRoot);
videoName = videos{videoIdx};

[gbvsParam, ofParam, poseletModel] = configureDetectors();

% load gaze data
s = load(fullfile(cache.gazeRoot, sprintf('%s.mat', videoName)));
gazeData = s.data;
clear s;
% gazeParam.gazeData = gazeData.points;

s = load(modelFile, 'options');
options = s.options;
clear s;

s = load(fullfile(visRoot, sprintf('%s.mat', videoName)));
frames = s.frames;
indFr = s.indFr;
cands = s.cands;
predMaps = s.predMaps;
clear s;

%% run
i = find(frames == frIdx);
ifr = find(indFr == i);
vr = VideoReader(fullfile(uncVideoRoot, sprintf('%s.avi', videoName)));
n = vr.Width;

fr = preprocessFrames(vr, frames(indFr(ifr)), gbvsParam, ofParam, poseletModel, cache);
gazeData.index = frames(indFr(ifr));

[sim, outMaps] = similarityFrame3(predMaps(:,:,indFr(ifr)), gazeData, measures, ...
    'self', ...
    struct('method', 'center', 'cov', [(n/16)^2, 0; 0, (n/16)^2]), ...
    struct('method', 'saliency_GBVS', 'map', fr.saliency), ...
    struct('method', 'saliency_PQFT', 'map', fr.saliencyPqft), ...
    struct('method', 'saliency_Hou', 'map', fr.saliencyHou));

for i = 1:length(methods)
    gim = rgb2gray(fr.image);
    gim = im2double(imadjust(gim, [0; 1], [0.3 0.7]));
    gf = repmat(gim, [1 1 3]);
    
    % add gaze map
    mg = max(max(outMaps(:,:,i)));
    if (mg > 0)
        prob = outMaps(:,:,i) ./ mg;
        alphaMap = 0.7 * repmat(prob, [1 1 3]);
        rgbHM = ind2rgb(round(prob * ncol), cmap);
        outfr = im2uint8(rgbHM .* alphaMap + gf .* (1-alphaMap));
    else
        outfr = im2uint8(gf);
    end

    if (exist('outWidth', 'var') && outWidth > 0)
        outfr = imresize(outfr, [nan, outWidth]);
    end
    imwrite(outfr, fullfile(outRoot, sprintf('exp_%s_%d_%s.png', videoName, frIdx, methods{i})), 'png');
end
