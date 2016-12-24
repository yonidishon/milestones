function [sim, outMaps, extra] = x_similarityFrame3(predMap, gaze, meas, varargin)
% Compute similarity of the predicted map and other approaches to the
% ground truth gaze data
%
% sim = similarityFrame(predMap, gazePts, meas, ...)
%
% INPUT
%   predMap     prediced attention map height X width array
%   gaze        gaze data with following fields:
%       .pointSigma
%       .points
%       .denseMap
%       .binaryMap
%       .otherMap
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

if (ischar(meas)) % single measure
    meas = {meas};
end

[m, n] = size(predMap);

nMeas = length(meas);
nMethods = length(varargin);
sim = zeros(1+nMethods, nMeas);
outMaps = zeros(m, n, 1+nMethods);
if (nargout > 2), extra.roc = cell(nMethods, 1); end;

if (isfield(gaze, 'binaryMaps') && isfield(gaze, 'index') && iscell(gaze.points)) % gaze data is list - sample
    gz.points = gaze.points{gaze.index};

    gazePts = gz.points(~isnan(gz.points(:,1)), :);
    gz.denseMap = points2GaussMap(gazePts', ones(1, size(gazePts, 1)), 0, [n, m], gaze.pointSigma);

    gz.binaryMap = gaze.binaryMaps(:,:,gaze.index);
    gz.otherMap = gaze.otherMaps(:,:,gaze.index);
    if (isfield(gaze, 'selfSimilarity'))
        gz.selfSimilarity = gaze.selfSimilarity(:,gaze.index);
    end
else
    gz = gaze;
    gazePts = gz.points(~isnan(gz.points(:,1)), :);
    gz.denseMap = points2GaussMap(gazePts', ones(1, size(gazePts, 1)), 0, [n, m], gaze.pointSigma);
end

% predited <-> gaze
outMaps(:,:,1) = predMap;
if (nargout > 2)
    [sim(1, :), extra.roc{1}] = x_similarityCalc(outMaps(:,:,1), gz, meas);
else
    sim(1, :) = x_similarityCalc(outMaps(:,:,1), gz, meas);
end

for imt = 1:nMethods
    if (ischar(varargin{imt}))
        mname = varargin{imt};
    else
        mname = varargin{imt}.method;
    end
    
    if (strcmp(mname, 'self'))
        outMaps(:,:,1+imt) = gz.denseMap;
        if 1 %(~isfield(gz, 'selfSimilarity'))
            sim(1+imt, :) = gazeSelfSimilarity(gazePts, [n, m], meas, gaze.pointSigma);
        else
            sim(1+imt, :) = gz.selfSimilarity;
        end
        
    elseif (strcmp(mname, 'center'))
        [X, Y] = meshgrid(1:n, 1:m);
        cp = [n/2, m/2];
        s = inv(varargin{imt}.cov);
        cg = exp(-1/2 .* (s(1,1).*(X-cp(1)).^2 + s(2,2).*(Y-cp(2)).^2 + (s(2,1)+s(1,2)).*(X-cp(1)).*(Y-cp(2))));
        cg = cg ./ max(cg(:));
        outMaps(:,:,1+imt) = cg;
        
        if (nargout > 2)
            [sim(1+imt, :), extra.roc{1+imt}] = x_similarityCalc(outMaps(:,:,1+imt), gz, meas);
        else
            x_sim(1+imt, :) = similarityCalc(outMaps(:,:,1+imt), gz, meas);
        end
        
    elseif (strncmp(mname, 'saliency', 8))
        outMaps(:,:,1+imt) = varargin{imt}.map;
        if (nargout > 2)
            [sim(1+imt, :), extra.roc{1+imt}] = x_similarityCalc(outMaps(:,:,1+imt), gz, meas);
        else
            sim(1+imt, :) = x_similarityCalc(outMaps(:,:,1+imt), gz, meas);
        end
        
    else
        warning('Unsupported method for frame similarity: %s', mname);
    end

end
