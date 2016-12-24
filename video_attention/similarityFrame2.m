function [sim, outMaps, extra] = similarityFrame2(predMap, gazePts, otherGazePts, meas, varargin)
% Compute similarity of the predicted map and other approaches to the
% ground truth gaze data
%
% sim = similarityFrame(predMap, gazePts, meas, ...)
%
% INPUT
%   predMap     prediced attention map height X width array
%   gazePts     gaze data points, npts X 2 array
%   otherGazePts    gaze points for other frames. Used for AUC calculation.
%                   Cell array (cell per frame) of npts X 2 arrays
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
extra.roc = cell(nMethods, 1);

% produce gaze map
gazePts = gazePts(~isnan(gazePts(:,1)), :);
gazeMap = points2GaussMap(gazePts', ones(1, size(gazePts, 1)), 0, [n, m], ptsSigma);
gaze = struct('points', gazePts, 'denseMap', gazeMap, 'binaryMap', points2binaryMap(gazePts, [n, m]));
if (~isempty(otherGazePts)) % use other gaze points (aggregated)
    otherMap = false(m, n);
    for i = 1:length(otherGazePts)
        ogp = otherGazePts{i};
        ogp = ogp(~isnan(ogp(:,1)), :);
        otherMap = otherMap | points2binaryMap(ogp, [n, m]);
    end
    gaze.otherMap = otherMap;
end

% predited <-> gaze
outMaps(:,:,1) = predMap;
[sim(1, :), extra.roc{1}] = similarityCalc(outMaps(:,:,1), gaze, meas);

for imt = 1:nMethods
    if (ischar(varargin{imt}))
        mname = varargin{imt};
    else
        mname = varargin{imt}.method;
    end
    
    if (strcmp(mname, 'self'))
        outMaps(:,:,1+imt) = gazeMap;
        sim(1+imt, :) = gazeSelfSimilarity(gazePts, [n, m], meas);
        
    elseif (strcmp(mname, 'center'))
        [X, Y] = meshgrid(1:n, 1:m);
        cp = [n/2, m/2];
        s = inv(varargin{imt}.cov);
        cg = exp(-1/2 .* (s(1,1).*(X-cp(1)).^2 + s(2,2).*(Y-cp(2)).^2 + (s(2,1)+s(1,2)).*(X-cp(1)).*(Y-cp(2))));
        cg = cg ./ max(cg(:));
        outMaps(:,:,1+imt) = cg;
        
        [sim(1+imt, :), extra.roc{1+imt}] = similarityCalc(outMaps(:,:,1+imt), gaze, meas);
        
    elseif (strncmp(mname, 'saliency', 8))
        outMaps(:,:,1+imt) = varargin{imt}.map;
        [sim(1+imt, :), extra.roc{1+imt}] = similarityCalc(outMaps(:,:,1+imt), gaze, meas);
        
    else
        warning('Unsupported method for frame similarity: %s', mname);
    end

end
