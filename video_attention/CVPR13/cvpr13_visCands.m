% CVPR13
% cvpr13_visCands
%% settings
uncVideoRoot = fullfile(diemDataRoot, 'video_unc');
gazeDataRoot = fullfile(diemDataRoot, 'gaze');
modelFile = fullfile(uncVideoRoot, '00_trained_model_validation_v5_3.mat');
gazePredRoot = fullfile(diemDataRoot, 'vis_jump', 'borji_test_v5_32'); % gaze predictions results
visRoot = fullfileCreate(diemDataRoot, 'cvpr13', 'cands');

% videoName = 'advert_bbc4_bees_1024x576'; frameIdx = 122; candType = 5; % static
videoName = 'harry_potter_6_trailer_1280x544';

cache.root = fullfile(diemDataRoot, 'cache');
cache.frameRoot = fullfile(diemDataRoot, 'cache');
cache.featureRoot = fullfile(cache.root, '00_features_v5');
cache.renew = false; % use in case the preprocessing mechanism updated
cache.renewFeatures = false; % use in case the feature extraction is updated

ptsSigma = 10;
ncol = 256;
% candCol = [0 0 0];

cmap = jet(ncol);
colors = [1 0 0;
    0 1 0;
    0 0 1;
    1 1 0;
    1 0 1;
    0 1 1];

%% prepare
% [gbvsParam, ofParam, poseletModel] = configureDetectors();

% load model
% s = load(modelFile, 'options');
% % rf = s.rf;
% options = s.options;
% options.useLabel = false; % no need in label while testing
% clear s;

% load gaze
s = load(fullfile(gazeDataRoot, sprintf('%s.mat', videoName)));
gazeData = s.gaze.data;
clear s;

% load predictions
s = load(fullfile(gazePredRoot, sprintf('%s.mat', videoName)));

[m, n, nfr] = size(s.predMaps);

%% run
vr = VideoReader(fullfile(uncVideoRoot, sprintf('%s.avi', videoName)));
vw = VideoWriter(fullfile(visRoot, sprintf('%s.avi', videoName)), 'Motion JPEG AVI'); % 'Motion JPEG AVI' or 'Uncompressed AVI' or 'MPEG-4' on 2012a.
open(vw);

try
    for ifr = 1:nfr
        % frame
        fr = read(vr, s.frames(ifr));
        
        % gaze map
        gazePts = gazeData{s.frames(s.indFr(ifr))};
        gazeMap = points2GaussMap(gazePts', ones(1, size(gazePts, 1)), 0, [n, m], ptsSigma);
        
        % cands
        nc = length(s.cands{ifr});
        candMap = zeros(m, n);
        alpha = 0.5;
        rad = 2;

        for ic = 1:nc % for each candidate
            msk = maskEllipse(n, m, s.cands{ifr}{ic}.point, s.cands{ifr}{ic}.cov, rad)';
            msk = bwperim(msk, 8);
            mskd = double(msk);
            candMap = candMap .* (1 - mskd) + (floor((s.cands{ifr}{ic}.type-1)/6 * ncol/2 + ncol/2)) .* mskd;
%             candMap = candMap .* (1 - mskd) + (s.cands{ifr}{ic}.score) .* mskd;
        end

        outMaps = cat(3, s.predMaps(:,:,ifr), gazeMap, candMap);
%         fr = preprocessFrames(vr, frames(ifr), gbvsParam, ofParam, poseletModel, cache);
%         maps = cat(3, (fr.ofx.^2 + fr.ofy.^2), fr.saliency);
%         cands = jumpCandidates3(fr.faces, fr.poselet_hit, maps, options);
        
        outfr = renderSideBySide(fr, outMaps, colors, cmap);
        writeVideo(vw, outfr);
    end
    
catch me
    close(vw);
    rethrow(me);
end

close(vw);
