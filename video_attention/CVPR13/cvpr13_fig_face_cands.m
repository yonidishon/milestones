% CVPR13
% cvpr13_fig_face_cands
%% settings
uncVideoRoot = fullfile(diemDataRoot, 'video_unc');
gazeDataRoot = fullfile(diemDataRoot, 'gaze');
modelFile = fullfile(uncVideoRoot, '00_trained_model_validation_v5_3.mat');
visRoot = fullfile(diemDataRoot, 'cvpr13');

% videoName = 'advert_bbc4_bees_1024x576'; frameIdx = 122; candType = 5; % static
% videoName = 'movie_trailer_quantum_of_solace_1280x688'; frameIdx = 2630; candType = [1 2 3]; % cneter, face, person
% videoName = 'harry_potter_6_trailer_1280x544'; frameIdx = 577; candType = [1 2 3];
% videoName = 'harry_potter_6_trailer_1280x544'; frameIdx = 599; candType = [1 2 3];
videoName = 'movie_trailer_alice_in_wonderland_1280x682'; frameIdx = 48; candType = [1 2 3];
% videoName = 'movie_trailer_alice_in_wonderland_1280x682'; frameIdx = 1873; candType = [1 2 3];

cache.root = fullfile(diemDataRoot, 'cache');
cache.frameRoot = fullfile(diemDataRoot, 'cache');
cache.renew = false; % use in case the preprocessing mechanism updated

ncol = 256;
candCol = [1 0 0;   % center
    0 1 0           % face
    0 0 1];         % body
scale = [1 8 1];
lineWidth = 1;
outWidth = 800; 

%% prepare
[gbvsParam, ofParam, poseletModel] = configureDetectors();

% load model
s = load(modelFile, 'options');
% rf = s.rf;
options = s.options;
options.useLabel = false; % no need in label while testing
options.useCenter = 1;
clear s;

cmap = jet(ncol);

%% run
vr = VideoReader(fullfile(uncVideoRoot, sprintf('%s.avi', videoName)));
fr = preprocessFrames(vr, frameIdx, gbvsParam, ofParam, poseletModel, cache);
maps = cat(3, (fr.ofx.^2 + fr.ofy.^2), fr.saliency);
cands = jumpCandidates3(fr.faces, fr.poselet_hit, maps, options);

%% visualize
gim = rgb2gray(fr.image);
gim = imadjust(gim, [0; 1], [0.2 0.8]);
gf = repmat(gim, [1 1 3]);
frout = im2double(gf);

% candidates
for ic = 1:length(cands)
    cind = find(candType == cands{ic}.type);
    if (~isempty(cind))
        frcand = repmat(reshape(candCol(cind, :), [1 1 3]), [fr.height, fr.width, 1]);
        msk = maskEllipse(fr.width, fr.height, cands{ic}.point, cands{ic}.cov * scale(cind), 1)';
        msk = bwperim(msk, 8);
        if (lineWidth > 1)
            msk = imdilate(msk, strel('disk', lineWidth));
        end
        msk = repmat(double(msk), [1 1 3]);
        
        frout = (1-msk) .* frout + msk .* frcand;
    end
end

frout = imresize(frout, [nan, outWidth]);
imwrite(frout, fullfile(visRoot, sprintf('%s_%d_semantic.png', videoName, frameIdx)), 'png');
