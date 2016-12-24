function [sim, outMaps] = similarityFrame(predMap, gazePts, meas, varargin)
% Compute similarity of the predicted map and other approaches to the
% ground truth gaze data
%
% sim = similarityFrame(predMap, gazePts, meas, ...)
%
% INPUT
%   predMap     prediced attention map height X width array
%   gazePts     gaze data points, npts X 2 array
%   meas        measures to use. Supported are 'auc' and 'chisq'
%   varargin    structs or strings that define another methods. If method
%               does not need any additional parameters, just name can be
%               used. Otherwise the struct includes a field 'method' with
%               the name of the method and additional method-dependent
%               fields. Supported methods:
%       * center    just Gaussian in the center. Requires a 'cov' field
%       * self      gaze self-similarity
%       * saliency* additional saliency measures (can be followed by name).
%                   Require a 'map' field, sized as predMap above
%
% OUTPUT
%   sim         comparison result, (1+nMethods) X nMeas array. The order of
%               measures corresponds to 'meas'. The first method is always
%               predMap <-> gaze map, followed by additional methods, as
%               they apper in 'varargin'
%   outMaps     all the used maps, in order of first dimention of 'sim'. In
%               case of 'self' method gaze map is used. Size height X width
%               X (1+nMethods)

ptsSigma = 10;

if (ischar(meas)) % single measure
    meas = {meas};
end

[m, n] = size(predMap);

nMeas = length(meas);
nMethods = length(varargin);
sim = zeros(1+nMethods, nMeas);
outMaps = zeros(m, n, 1+nMethods);

% produce gaze map
gazeMap = points2GaussMap(gazePts', ones(1, size(gazePts, 1)), 0, [n, m], ptsSigma);
gaze = struct('points', gazePts, 'denseMap', gazeMap, 'binaryMap', points2binaryMap(gazePts, [n, m]));

% predited <-> gaze
outMaps(:,:,1) = predMap;
sim(1, :) = similarityCalc(outMaps(:,:,1), gaze, meas);
% for imeas = 1:nMeas
%     if (strcmp(meas{imeas}, 'chisq'))
%         sim(1, imeas) = probMapSimilarity(gazeMap, predMap, meas{imeas});
%     elseif (strcmp(meas{imeas}, 'auc'))
%         sim(1, imeas) = probMapSimilarity(gazePts, predMap, meas{imeas});
%     end
% end

for imt = 1:nMethods
    if (ischar(varargin{imt}))
        mname = varargin{imt};
    else
        mname = varargin{imt}.method;
    end
    
    if (strcmp(mname, 'self'))
        outMaps(:,:,1+imt) = gazeMap;
        sim(1+imt, :) = gazeSelfSimilarity(gazePts, [n, m], meas);
%         for imeas = 1:nMeas
%             sim(1+imt, imeas) = gazeSelfSimilarity(gazePts, [n, m], meas{imeas});
%         end
        
    elseif (strcmp(mname, 'center'))
        [X, Y] = meshgrid(1:n, 1:m);
        cp = [n/2, m/2];
        s = inv(varargin{imt}.cov);
        cg = exp(-1/2 .* (s(1,1).*(X-cp(1)).^2 + s(2,2).*(Y-cp(2)).^2 + (s(2,1)+s(1,2)).*(X-cp(1)).*(Y-cp(2))));
        cg = cg ./ max(cg(:));
        outMaps(:,:,1+imt) = cg;
        
        sim(1+imt, :) = similarityCalc(outMaps(:,:,1+imt), gaze, meas);
%         for imeas = 1:nMeas
%             if (strcmp(meas{imeas}, 'chisq'))
%                 sim(1+imt, imeas) = probMapSimilarity(gazeMap, cg, meas{imeas});
%             elseif (strcmp(meas{imeas}, 'auc'))
%                 sim(1+imt, imeas) = probMapSimilarity(gazePts, cg, meas{imeas});
%             end
%         end
        
    elseif (strncmp(mname, 'saliency', 8))
        outMaps(:,:,1+imt) = varargin{imt}.map;
        sim(1+imt, :) = similarityCalc(outMaps(:,:,1+imt), gaze, meas);
%         for imeas = 1:nMeas
%             if (strcmp(meas{imeas}, 'chisq'))
%                 sim(1+imt, imeas) = probMapSimilarity(gazeMap, varargin{imt}.map, meas{imeas});
%             elseif (strcmp(meas{imeas}, 'auc'))
%                 sim(1+imt, imeas) = probMapSimilarity(gazePts, varargin{imt}.map, meas{imeas});
%             end
%         end
        
    else
        warning('Unsupported method for frame similarity: %s', mname);
    end

end
