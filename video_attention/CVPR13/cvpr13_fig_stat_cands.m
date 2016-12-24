% CVPR13
% cvpr13_fig_stat_cands
%% settings
uncVideoRoot = fullfile(diemDataRoot, 'video_unc');
gazeDataRoot = fullfile(diemDataRoot, 'gaze');
modelFile = fullfile(uncVideoRoot, '00_trained_model_validation_v5_3.mat');
visRoot = fullfile(diemDataRoot, 'cvpr13');

% videoName = 'advert_bbc4_bees_1024x576'; frameIdx = 122; candType = 5; % static
videoName = 'DIY_SOS_1280x712'; frameIdx = 100; candType = 5; % static
% videoName = 'harry_potter_6_trailer_1280x544'; frameIdx = 685; candType = 5; % static
% videoName = 'DIY_SOS_1280x712'; frameIdx = 320; candType = 4; % motion

cache.root = fullfile(diemDataRoot, 'cache');
cache.frameRoot = fullfile(diemDataRoot, 'cache');
cache.featureRoot = fullfile(cache.root, '00_features_v5');
cache.renew = false; % use in case the preprocessing mechanism updated
cache.renewFeatures = false; % use in case the feature extraction is updated
cache.renewJumps = false; % recalculate the final result

ncol = 256;
candCol = [0 0 0];
lineWidth = 1;
outWidth = 800; 

%% prepare
[gbvsParam, ofParam, poseletModel] = configureDetectors();

% load model
s = load(modelFile, 'options');
% rf = s.rf;
options = s.options;
options.useLabel = false; % no need in label while testing
clear s;

cmap = jet(ncol);

%% run
vr = VideoReader(fullfile(uncVideoRoot, sprintf('%s.avi', videoName)));
fr = preprocessFrames(vr, frameIdx, gbvsParam, ofParam, poseletModel, cache);
maps = cat(3, (fr.ofx.^2 + fr.ofy.^2), fr.saliency);
cands = jumpCandidates3(fr.faces, fr.poselet_hit, maps, options);

%% visualize
% heatmap
if (candType == 5)
    prob = fr.saliency;
elseif (candType == 4);
    prob = (fr.ofx.^2 + fr.ofy.^2);
end

if (sum(prob(:) > 0))
    prob = prob ./ max(prob(:));
end

alphaMap = 0.7 * repmat(prob, [1 1 3]);
rgbHM = ind2rgb(round(prob * ncol), cmap);
gim = rgb2gray(fr.image);
gim = imadjust(gim, [0; 1], [0.3 0.7]);
gf = repmat(gim, [1 1 3]);
frout = rgbHM .* alphaMap + im2double(gf) .* (1-alphaMap);

% candidates
frcand = repmat(reshape(candCol, [1 1 3]), [fr.height, fr.width, 1]);
for ic = 1:length(cands)
    if (cands{ic}.type == candType)
        msk = maskEllipse(fr.width, fr.height, cands{ic}.point, 2*cands{ic}.cov, 1)';
        msk = bwperim(msk, 8);
        if (lineWidth > 1)
            msk = imdilate(msk, strel('disk', lineWidth));
        end
        msk = repmat(double(msk), [1 1 3]);
        
        frout = (1-msk) .* frout + msk .* frcand;
    end
end

frout = imresize(frout, [nan, outWidth]);
imwrite(frout, fullfile(visRoot, sprintf('%s_%d_type-%d.png', videoName, frameIdx, candType)), 'png');
