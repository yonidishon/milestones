function featVec = jumpFeaturePoint6(srcFr, srcCand, dstFr, dstCand, options)
% builds feature vector at a single source and destination candidate.
% Optimized for CVPR13 model and does not use srcFr. It can be set to empty
% to save preprocessing
% The structure of feature vector:
% 1       ...           5+3*nMS+nSS+nCS         source properties
% 5+3*nMS+nSS+nCS+1 ... 2*(5+3*nMS+nSS+nCS)     same destination properties
% 31                                            distance (Euclidian) of jump
% 32                                            direction of jump (arctan of the vector angle)
% Source / destination properties are:
% 1                         distance (Euclidian) to the center of the frame
% 2...3*nMS+1               mean motion magnitude and direction in region
% 3*nMS+2 ... 3*nMS+nSS+1   mean saliency in region
% 3*nMS+nSS+2               probability of face
% 3*nMS+nSS+3                       probability of person
% 3*nMS+nSS+4                       candidate type
% 3*nMS+nSS+5 ... 3*nMS+nSS+nCS+4   mean relative contrast in region
% 3*nMS+nSS+nCS+5                   candidate size
%
% featVec = jumpFeaturePoint(srcFr, srcCand, dstFr, dstCand, options)
%
% INPUT
%   srcFr, dstFr        source and destination frames. Each includes:
%       .width, .height
%       .image
%       .ofx, .ofy          optical flow
%       .saliency
%       .faces
%       .poselet_hit
%   srcCand, dstCand    source and destination candidate. Each includes
%       .point
%       .type
%       .candCov
%   options         calculation settions, includes
%       .motionScales: list of half window sizes for motion averaging, length nMS
%       .saliencyScales: list of half window sizes for saliency averaging, length nSS
%       .contrastScales: list of half window sizes for contrast calculation, length nCS
%       .sigmaScale: scaling of Gaussian sigma for face / human scoring
% OUTPUT
%   featVec         created feature vector, [nfeat X 1]

vn = cand2Vec(dstFr, dstCand, options);
v = srcCand.point - dstCand.point;
dist = sqrt(abs(sum(v.^2)));
drc = atan(v(2)/v(1));

featVec = [vn; dist; drc];

function vec = cand2Vec(fr, cand, options)

pt = round(cand.point);

% preallocate
nf = 5 + 3*length(options.motionScales) + length(options.saliencyScales) + length(options.contrastScales);
vec = zeros(nf, 1);

curI = 1;
% distance to center
cen = [fr.width/2, fr.height/2];
vec(curI) = sqrt(abs(sum((pt - cen).^2)));
curI = curI + 1;

% v4
% motion energy
g1 = fspecial('gaussian', [51 51], 10);
g2 = fspecial('gaussian', [51 51], 20);
ofx = abs(imfilter(fr.ofx, g2, 'symmetric') - imfilter(fr.ofx, g1, 'symmetric'));
ofy = abs(imfilter(fr.ofy, g2, 'symmetric') - imfilter(fr.ofy, g1, 'symmetric'));
ofm = sqrt(fr.ofx.^2 + fr.ofy.^2);
ofm = abs(imfilter(ofm, g2, 'symmetric') - imfilter(ofm, g1, 'symmetric'));
nsc = length(options.motionScales);
for i = 1:nsc
    ri = round(max(1, pt(2)-options.motionScales(i)):min(fr.height, pt(2)+options.motionScales(i)));
    rj = round(max(1, pt(1)-options.motionScales(i)):min(fr.width, pt(1)+options.motionScales(i)));
    vec(curI-1+i) = mean(mean(ofm(ri, rj)));
    vec(curI-1+nsc+i) = mean(mean(ofx(ri, rj)));
    vec(curI-1+2*nsc+i) = mean(mean(ofy(ri, rj)));
end
curI = curI + 3*nsc;

% v3
% % motion energy
% ofm = sqrt(fr.ofx.^2 + fr.ofy.^2);
% nsc = length(options.motionScales);
% for i = 1:nsc
%     ri = round(max(1, pt(2)-options.motionScales(i)):min(fr.height, pt(2)+options.motionScales(i)));
%     rj = round(max(1, pt(1)-options.motionScales(i)):min(fr.width, pt(1)+options.motionScales(i)));
%     vec(curI-1+i) = mean(mean(ofm(ri, rj)));
%     vec(curI-1+nsc+i) = mean(mean(fr.ofx(ri, rj)));
%     vec(curI-1+2*nsc+i) = mean(mean(fr.ofy(ri, rj)));
% end
% curI = curI + 3*nsc;

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
    curI = curI + 1;
end

% candidate type
vec(curI) = cand.type;
curI = curI + 1;

% relative contrast
nsc = length(options.contrastScales);
img = im2double(rgb2gray(fr.image));
lmin = min(img(:));
lmax = max(img(:));
if (lmax == lmin)
    vec(curI:curI+nsc-1) = 0;
else
    gc = (lmax - lmin) / (lmax + lmin);
    for i = 1:nsc
        ri = round(max(1, pt(2)-options.contrastScales(i)):min(fr.height, pt(2)+options.contrastScales(i)));
        rj = round(max(1, pt(1)-options.contrastScales(i)):min(fr.width, pt(1)+options.contrastScales(i)));
        reg = img(ri, rj);
        rmax = max(reg(:));
        rmin = min(reg(:));
        vec(curI-1+i) = ((rmax - rmin) / (rmax + rmin)) / gc;
    end
end
curI = curI + nsc;

% candidate size
vec(curI) = sqrt(mean(diag(cand.candCov)));
