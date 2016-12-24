function sim = gazeSelfSimilarity(pts, frameSz, meas, ptsSigma)
% Calculates self-similarity of a heatmap created by points. The method is
% to place a constant sized Gaussian (10 px) at every and check the
% distance between two random halves. The result is averages several (10)
% times
%
% sim = gazeSelfSimilarity(pts, frameSz, meas)
%
% INPUT
%   pts         data points, each point in a row. (npts X 2) array
%   frameSz     frame size, (w X h)
%   meas        measure to use, 'chisq' for Chi-Square, 'auc' for AUC
% 
% OUTPUT
%   sim         similarity result

if (~exist('ptsSigma', 'var'))
    ptsSigma = 10; % sigma of Gaussian around every point
end
nTst = 10; % number of tests to average

if (ischar(meas)), meas = {meas}; end;

nmeas = length(meas);
sim = zeros(nTst, nmeas);
npts = size(pts, 1);
if (npts == 0)
    sim = nan(1, nmeas);
    return;
end

trn = 1:ceil(npts/2);
tst = trn(end)+1:npts;
wtrn = ones(length(trn));
% wtst = ones(length(tst));

for it = 1:nTst
    % randomly split the set of points into two halfs
    p = randperm(npts);
    gzTrn = points2GaussMap(pts(p(trn), :)', wtrn, 0, frameSz, ptsSigma);
    
    % calculate similarity
    
    sim(it, :) = similarityCalc(gzTrn, struct('points', pts(p(tst), :)), meas);
%     if (strcmp(meas, 'chisq'))
%         gzTst = points2GaussMap(pts(p(tst), :)', wtst, 0, frameSz, ptsSigma);
%         sim(it) = probMapSimilarity(gzTst, gzTrn, 'chisq');
%     else
%         sim(it) = probMapSimilarity(pts(p(tst), :), gzTrn, 'auc');
%     end
end

sim = mean(sim);
