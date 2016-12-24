function featVec = jumpFeaturePoint(srcFr, srcPt, dstFr, dstPt, options)
% builds feature vector at a single source and destination (candidate)
% point. The structure of feature vector:
% 1       ...       3+3*nMS+nSS     source properties
% 3+3*nMS+nSS+1 ... 2*(3+3*nMS+nSS) same destination properties
% 31                                distance (Euclidian) of jump
% 32                                direction of jump (arctan of the vector angle)
% Source / destination properties are:
% 1                         distance (Euclidian) to the center of the frame
% 2...3*nMS+1               mean motion magnitude and direction in region
% 3*nMS+2 ... 3*nMS+nSS+1   mean saliency in region
% 3*nMS+nSS+2               probability of face
% 3*nMS+nSS+3               probability of person
%
% featVec = jumpFeaturePoint(srcFr, srcPt, dstFr, dstPt, options)
%
% INPUT
%   srcFr, dstFr    source and destination frames. Each includes:
%                   - width, height
%                   - ofx, ofy: optical flow
%                   - saliency
%                   - faces
%                   - poselet_hit
%   srcPt, dstPt    source and destination points
%   options         calculation settions, includes
%                   - motionScales: list of half window sizes for motion
%                   averaging, length nMS
%                   - saliencyScales: list of half window sizes for
%                   saliency averaging, length nSS
%                   - sigmaScale: scaling of Gaussian sigma for face /
%                   human scoring
% OUTPUT
%   featVec         created feature vector, [nfeat X 1]

vp = framePoint2Vec(srcFr, srcPt, options);
vn = framePoint2Vec(dstFr, dstPt, options);
v = srcPt - dstPt;
dist = sqrt(abs(sum(v.^2)));
drc = atan(v(2)/v(1));

featVec = [vp; vn; dist; drc];

function vec = framePoint2Vec(fr, pt, options)

pt = round(pt);

% preallocate
nf = 3 + 3*length(options.motionScales) + length(options.saliencyScales);

vec = zeros(nf, 1);

curI = 1;
% distance to center
cen = [fr.width/2, fr.height/2];
vec(curI) = sqrt(abs(sum(pt - cen).^2));
curI = curI + 1;

% motion energy
ofm = sqrt(fr.ofx.^2 + fr.ofy.^2);
nsc = length(options.motionScales);
for i = 1:nsc
    ri = round(max(1, pt(2)-options.motionScales(i)):min(fr.height, pt(2)+options.motionScales(i)));
    rj = round(max(1, pt(1)-options.motionScales(i)):min(fr.width, pt(1)+options.motionScales(i)));
    vec(curI-1+i) = mean(mean(ofm(ri, rj)));
    vec(curI-1+nsc+i) = mean(mean(fr.ofx(ri, rj)));
    vec(curI-1+2*nsc+i) = mean(mean(fr.ofy(ri, rj)));
end
curI = curI + 3*nsc;

% static saliency
nsc = length(options.saliencyScales);
if (isempty(fr.saliency))
    vec(curI:curI+nsc-1) = 0;
else
    for i = 1:nsc
        ri = round(max(1, pt(2)-options.saliencyScales(i)):min(fr.height, pt(2)+options.saliencyScales(i)));
        rj = round(max(1, pt(1)-options.saliencyScales(i)):min(fr.width, pt(1)+options.saliencyScales(i)));
        vec(curI-1+i) = mean(mean(fr.saliency(ri, rj)));
    end
end
curI = curI + nsc;

if (pt(1) >= 1 && pt(1) <= fr.width && pt(2) >= 1 && pt(2) <= fr.height) % if inside
    % score of face
    p = rectDet2GaussMap(fr.faces(:,1:4), fr.faces(:, 6) ./ max(fr.faces(:, 6)), [fr.width, fr.height], options.sigmaScale);
    vec(curI) = p(pt(2), pt(1));
    curI = curI + 1;
    
    % score of person
    p = rectDet2GaussMap(double(fr.poselet_hit.bounds'), double(fr.poselet_hit.score(:)) ./ max(double(fr.poselet_hit.score(:))), [fr.width, fr.height], options.sigmaScale);
    vec(curI) = p(pt(2), pt(1));
    % curI = curI + 1;
end
