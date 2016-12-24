% CVPR13
% cvpr13_fig_DiemGaze
%% settings
uncVideoRoot = fullfile(diemDataRoot, 'video');
gazeRoot = fullfile(diemDataRoot, 'gaze');
modelFile = fullfile(uncVideoRoot, '00_trained_model_validation_v5_3.mat');
visRoot = fullfile(diemDataRoot, 'cvpr13');

% videoName = 'tv_uni_challenge_final_1280x712'; frameIdx = 430;
% videoName = 'tv_uni_challenge_final_1280x712'; frameIdx = 407;
videoName = 'tv_uni_challenge_final_1280x712'; frameIdx = 1022;
% videoName = 'DIY_SOS_1280x712'; frameIdx = 182;
% videoName = 'DIY_SOS_1280x712'; frameIdx = 508;

scale = 5;
sigma = 10;
ncol = 256;
outWidth = 800; 
cmap = jet(ncol);

%% run
vr = VideoReader(fullfile(uncVideoRoot, sprintf('%s.mp4', videoName)));
img = read(vr, frameIdx);
[m, n, ~] = size(img);

s = load(fullfile(gazeRoot, sprintf('%s.mat', videoName)));
gaze = s.gaze;
clear s;

gazePts = gaze.data{frameIdx};
prob = points2GaussMap(gazePts', ones(1, size(gazePts, 1)), 0, [gaze.width, gaze.height], sigma);
prob = imresize(prob, [m, n]);

if (sum(prob(:) > 0))
    prob = prob ./ max(prob(:));
end

alphaMap = 0.6 * repmat(prob, [1 1 3]);
rgbHM = ind2rgb(round(prob * ncol), cmap);
gim = rgb2gray(img);
gim = imadjust(gim, [0; 1], [0.3 0.7]);
gf = repmat(gim, [1 1 3]);
frout = rgbHM .* alphaMap + im2double(gf) .* (1-alphaMap);

imwrite(frout, fullfile(visRoot, sprintf('%s_%d_video_gaze.png', videoName, frameIdx)));
