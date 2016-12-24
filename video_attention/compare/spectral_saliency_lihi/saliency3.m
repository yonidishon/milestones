function smap=saliency3(m,scale);

% Calculate a saliency  map using 3D-DFT phase  (spectral residual alternatively).
% Inputs:
% m = grayscale video matrix sized: [height,width,frames]
% scale = a scaling factor <=1 
% Output:
% smap = a grayscale saliency video matrix sized: [height*scale,width*scale ,frames]

for k=1:size(m,3)
    inVid(:,:,k) = single(imresize(m(:,:,k), scale, 'bilinear'));
end;

%% Spectral Residual
% myFFT = fftn(inVid);
% myLogAmplitude = log(abs(myFFT));
% myPhase = angle(myFFT);
% filt=repmat( fspecial('average', 3),[1 1 3]);
% filt=filt/sum(filt(:));
% mySmooth = imfilter(myLogAmplitude, filt, 'replicate');
% mySpectralResidual = myLogAmplitude - mySmooth;
% saliencyMap = abs(ifftn(exp(mySpectralResidual + j*myPhase))).^2;

%% Phase
myFFT = fftn(inVid);
myFFT = myFFT./abs(myFFT);
saliencyMap = abs(ifftn(myFFT)).^2;

%% Post processing - smoothing
filt=repmat( fspecial('disk', 3),[1 1 6]);
filt=filt/sum(filt(:));

smap = imfilter(saliencyMap, filt);
