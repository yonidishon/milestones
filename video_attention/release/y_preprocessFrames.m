function frs = y_preprocessFrames(vr, frIdx, ofParam)
% Preprocesses video frames by calculating statis saliency, optical flow,
% face and poselet detector. Can work with frame cache.
%
% frs = preprocessFrames(vr, frIdx, gbvsParam, ofParam, poseletModel, cache)
% frs = preprocessFrames(vr, frIdx, gbvsParam, ofParam, poseletModel)
%
% INPUT
%   vr              video stream as VideoReader object
%   frIdx           indices of the relevant frames inside the video stream
%   gbvsParam       options for static saliency (GBVS) calculation. Output
%                   of configureDetectors() function
%   ofParam         options for optical flow calculation. Output of
%                   configureDetectors() function
%   poseletModel    model for poselet detection. Output of
%                   configureDetectors() function
%   cache           if set the cache will be used. This is a structure:
%       .frameRoot      root folder for the frame cache
%       .renew          if set to true the cache data will be overwritten
%
% OUTPUT
%   frs             cell array of preprocessed frames. If there is only one
%                   frame cell array is not created. Includes
%       .width, .height     the dimentions of the frame
%       .image              original frame
%       .videoName          video file name
%       .index              frame index in video
%       .ofx, .ofy          optical flow
%       .saliency           static saliency
%       .faces              detected face rectangles
%       .poselet_hit        poselet hits

nfr = length(frIdx);
frs = cell(nfr, 1);

for ifr = 1:nfr
    ind = frIdx(ifr); 
    data.width = vr.Width;
    data.height = vr.Height;
    f = read(vr, ind);
    data.image = f;
    
    % motion
    fp = read(vr, ind - 2);
    [data.ofx, data.ofy] = Coarse2FineTwoFrames(f, fp, ofParam);
    if (data.height < 128)
        scale = 2;
        ff = imresize(f, scale);
    else
        scale = 1;
        ff = f;
    end
    frs{ifr} = data;
end

if (nfr == 1)
    frs = frs{1};
end
