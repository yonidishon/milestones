function frs = xxx_preprocessFramesPartial(vr, frIdx, gbvsParam, ofParam, cache)
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

if (exist('cache', 'var'))
    cacheDir = fullfile(fullfile(cache.frameRoot, vr.Name));
    if (~exist(cacheDir, 'dir'))
        mkdir(cacheDir);
    end
else
    cacheDir = [];
end

nfr = length(frIdx);
frs = cell(nfr, 1);

for ifr = 1:nfr
    ind = frIdx(ifr);
    cacheFile = fullfile(cacheDir, sprintf('frame_%06d.mat', ind));
    
    if (exist('cache', 'var') && exist(cacheFile, 'file') && ~cache.renew) % load from cache
        s = load(cacheFile);
        data = s.data;
        % support image data
        if (~isfield(s.data, 'image')) % there is no image data - add it
            data.image = read(vr, ind);
        end
        if (~isfield(s.data, 'index')) % add frame index
            data.index = frIdx;
            data.videoName = vr.name;
        end

        clear s;
        frs{ifr} = data;
        continue;
    end
    
    data.width = vr.Width;
    data.height = vr.Height;
    
    f = read(vr, ind);
    data.image = f;
    
    % motion
    if (ind > 3)
        fp = read(vr, ind - 2);
        [data.ofx, data.ofy] = Coarse2FineTwoFrames(f, fp, ofParam);
    else
        data.ofx = zeros(data.height,data.width);
        data.ofy = data.ofx;
    end
    
    % saliency
%     if ((max(f(:)) - min(f(:))) < 25)
    fg = rgb2gray(f);
    if ((max(fg(:)) - min(fg(:))) < 40 || sum(imhist(fg) == 0) > 200) % not contrast enough
        data.saliency = zeros(data.height, data.width);
    else
        if (data.height < 128)
            scale = 2;
            ff = imresize(f, scale);
        else
            scale = 1;
            ff = f;
        end
        
        salOut = gbvs(ff, gbvsParam);
        if (scale > 1)
            salOut.master_map_resized = imresize(salOut.master_map_resized, 1/scale);
        end
        data.saliency = salOut.master_map_resized;
    end
       
    % video data
    data.index = frIdx;
    data.videoName = vr.name;
    
    % save
    frs{ifr} = data;
%     if (exist('cache', 'var'))
%         save(cacheFile, 'data');
%     end
end

if (nfr == 1)
    frs = frs{1};
end
