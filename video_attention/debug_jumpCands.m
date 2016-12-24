% debug_jumpCands
uncVideoRoot = fullfile(diemDataRoot, 'video_unc');
gazeDataRoot = fullfile(diemDataRoot, 'gaze');
modelFile = fullfile(uncVideoRoot, '00_trained_model_validation_v5_3.mat');
gazePredRoot = fullfile(diemDataRoot, 'vis_jump', 'borji_test_v5_32'); % gaze predictions results
visRoot = fullfileCreate(diemDataRoot, 'cvpr13', 'cands');

videoName = 'harry_potter_6_trailer_1280x544'; frIdx = 484;

cache.root = fullfile(diemDataRoot, 'cache');
cache.frameRoot = fullfile(diemDataRoot, 'cache');
cache.featureRoot = fullfile(cache.root, '00_features_v5');
cache.renew = false; % use in case the preprocessing mechanism updated
cache.renewFeatures = false; % use in case the feature extraction is updated

s = load(fullfile(gazePredRoot, sprintf('%s.mat', videoName)), 'frames', 'cands', 'indFr');
cands = s.cands;
frames = s.frames;
indFr = s.indFr;
clear s;

vr = VideoReader(fullfile(uncVideoRoot, sprintf('%s.avi', videoName)));
fr = read(vr, frames(frIdx));

fro = visCandidates(fr, cands{frIdx});
figure, imshow(fro);

