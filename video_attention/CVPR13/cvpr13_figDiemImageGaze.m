% cvpr13_figDiemImageGaze
%% settings
dataRoot = fullfile(dropbox, 'CrowdGazeImage');
videoRoot = fullfile(diemDataRoot, 'video');
outRoot = fullfile(diemDataRoot, 'cvpr13');

imgName = 'tv_uni_challenge_final_1280x712_2.jpg'; videoName = 'tv_uni_challenge_final_1280x712'; frameIdx = 1022;
% imgName = 'tv_uni_challenge_final_1280x712_1.jpg'; videoName = 'tv_uni_challenge_final_1280x712'; frameIdx = 407;
% imgName = 'DIY_SOS_1280x712_2.jpg'; videoName = 'DIY_SOS_1280x712'; frameIdx = 508;

% sigma = 10;
ncol = 256;
cmap = jet(ncol);

%% run
vr = VideoReader(fullfile(videoRoot, sprintf('%s.mp4', videoName)));
img = read(vr, frameIdx);
[m, n, ~] = size(img);

data = importCrowdGazeImage(dataRoot, [n m]);
idx = find(strcmp(data.imageNames, imgName));
gazePts = data.gaze{idx};
sigma = sqrt(mean(data.error{idx}));

% binMap = points2binaryMap(gazePts, [n, m]);
prob = points2GaussMap(gazePts', ones(1, size(gazePts, 1)), 0, [n, m], sigma * m / 144);

alphaMap = 0.6 * repmat(prob, [1 1 3]);
rgbHM = ind2rgb(round(prob * ncol), cmap);
gim = rgb2gray(img);
gim = imadjust(gim, [0; 1], [0.3 0.7]);
gf = repmat(gim, [1 1 3]);
frout = rgbHM .* alphaMap + im2double(gf) .* (1-alphaMap);

imwrite(frout, fullfile(outRoot, sprintf('%s_%d_img_gaze.png', videoName, frameIdx)));
