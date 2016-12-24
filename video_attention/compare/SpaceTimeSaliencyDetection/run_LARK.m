vr = VideoReader('C:\Users\dmitryr\Documents\Datasets\DIEM\video_unc\movie_trailer_quantum_of_solace_1280x688.avi');
frs = read(vr, [1420 1499]);
Seq = zeros(64, 64, size(frs, 4)); 
for i = 1:size(frs, 4)
    Seq(:,:,i) = im2double(rgb2gray(imresize(frs(:,:,:,i), [64 64]))); 
end; 
Seq = Seq ./ std(Seq(:));

param.wsize = 3; % LARK spatial window size
param.wsize_t = 3; % LARK temporal window size
param.alpha = 0.42; % LARK sensitivity parameter
param.h = 1;  % smoothing parameter for LARK
param.sigma = 0.7; % fall-off parameter for self-resemblamnce

LARK = ThreeDLARK(Seq,param);
SM = SpaceTimeSaliencyMap(Seq,LARK,param.wsize,param.wsize_t,param.sigma);

salmap = zeros(size(frs, 1), size(frs, 2), size(frs, 4)); 
for i = 1:size(frs, 4)
    salmap(:,:,i) = imresize(SM(:,:,i), [size(frs, 1), size(frs, 2)]); 
end

salmap = salmap - min(salmap(:)); salmap = salmap ./ max(salmap(:));
