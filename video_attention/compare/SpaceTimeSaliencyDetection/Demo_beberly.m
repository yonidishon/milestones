% Space-time Saliency detection Demo
% [HISTORY]
% Apr 25, 2011 : created by Hae Jong Seo

% clear all;
% close all;
% clc;

for i = 1:40
    img = double(imread(['beberly/' num2str(i) '.png']));
    Y(:,:,i) = img;
    S = imresize(img,[64 64],'bilinear');
    Seq(:,:,i) = S/std(S(:));
end
Seq = Seq/std(Seq(:));

%%Parameters
param.wsize = 3; % LARK spatial window size
param.wsize_t = 3; % LARK temporal window size
param.alpha = 0.42; % LARK sensitivity parameter
param.h = 1;  % smoothing parameter for LARK
param.sigma = 0.7; % fall-off parameter for self-resemblamnce

% Compute 3-D LARKs
tic;
LARK = ThreeDLARK(Seq,param);
disp(['3-D LARK computation : ' num2str(toc) 'sec']);
% Compute space-time saliency
tic;
SM = SpaceTimeSaliencyMap(Seq,LARK,param.wsize,param.wsize_t,param.sigma);
disp(['SpaceTimeSaliencyMap : ' num2str(toc) 'sec']);

%% Visualize saliency values on top of video frame

figure(1),
for i = 1:size(Seq,3)
    a = imresize(SM(:,:,i),[size(Y,1) size(Y,2)]);
    sc(cat(3,a, Y(:,:,i)),'prob_jet');
    pause(.01);
end



