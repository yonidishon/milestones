close all;
clear all;
clc;
img=imread('pattern1.jpg');
scale = 0.5;

figure;
imshow(img);


figure;
imshow(img);
saliencyMap=saliency(img,scale);
title('Saliency by spectral residual');

img=im2double(rgb2gray(img));
img=imresize(img, scale, 'bilinear');

% show phase only image
f=fft2(img);
ff=f./abs(f);
ii=abs(ifft2(ff)).^2;
sm=imfilter(ii, fspecial('disk', 3));

figure;
imshow(mat2gray((sm)))
title('Saliency by phase only');



%% phase-correlation saliency
I = img;
I2 = zeros(size(I));
I2(1,1) = 1;
r = ifft2( (fft2(I2).*conj(fft2(I)))./abs(fft2(I2).*conj(fft2(I))) );
s2 =imfilter(abs(r).^2, fspecial('disk', 3));
figure;
imagesc(r);
colormap(gray);
title('Phase correlation with delta');
figure;
imagesc(s2);
colormap(gray);
title('Saliency by Phase correlation with delta');


























