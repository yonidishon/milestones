function [sim, extra] = similarityCalc(predMap, gaze, meas)
% Calculates similarity between predicted map and gaze fixations using
% several measures
% 
% sim = similarityCalc(predMap, gaze, meas)
%
% INPUT
%   predMap     predicted heat map
%   gaze        structure with gaze data. Fields:
%       .points     
%       .denseMap   
%       .binaryMap  
%       .otherMap   
%   meas        string or cell array of measures to use. Supported are: 
%       'chisq'     Chi Square
%       'cc'        Linear Correlation Coefficient
%       'nss'       Normalized Scanpath Saliency
%       'auc'       Area Under Curve (ROC)
%
% OUTPUT
%   sim         similarity of the maps, vector corresponding to number of
%               measures

[m, n] = size(predMap);
if (ischar(meas)), meas = {meas}; end;
nmeas = length(meas);
sim = zeros(1, nmeas);

if (isempty(gaze.points))
    sim = nan(1, nmeas);
    return;
end

if (~isfield(gaze, 'otherMap'))
    gaze.otherMap = [];
end

for im = 1:nmeas
    if (strcmpi(meas{im}, 'chisq')) % Chi square measure
        if (~isfield(gaze, 'denseMap'))
            gaze.denseMap = points2GaussMap(gaze.points', ones(1, size(gaze.points, 1)), 0, [n, m], 10);
        end
        
        p1 = predMap(:) / sum(predMap(:));
        p2 = gaze.denseMap(:) / sum(gaze.denseMap(:));
        sim(im) = sum((p1 - p2) .^ 2 ./ (p1 + p2+eps)) ./ 2;
        %sim(im) = sum((p1 - p2) .^ 2 ./ (p1 + p2)) ./ 2;

    elseif (strcmpi(meas{im}, 'cc')); % Linear Correlation Coefficient
        if (~isfield(gaze, 'binaryMap'))
            gaze.binaryMap = points2binaryMap(gaze.points, [n, m]);
        end
        
        sim(im) = mean(calcCCscore(predMap, gaze.binaryMap));
    elseif (strcmpi(meas{im}, 'nss')); % Normalized Scanpath Saliency
        if (~isfield(gaze, 'binaryMap'))
            gaze.binaryMap = points2binaryMap(gaze.points, [n, m]);
        end
        
        sim(im) = mean(calcNSSscore(predMap, gaze.binaryMap));
            
    elseif (strcmpi(meas{im}, 'auc')); % Area Under Curve
        if (~isfield(gaze, 'binaryMap'))
            gaze.binaryMap = points2binaryMap(gaze.points, [n, m]);
        end
        
        if (~any(gaze.binaryMap))
            sim(im) = nan;
        else
            [ac, R] = calcAUCscore(predMap, gaze.binaryMap, gaze.otherMap); % Shuffle AUC
            %[ac, R] = calcAUCscore(predMap, gaze.binaryMap); % AUC
            sim(im) = mean(ac);
            nb = 100;
            b = linspace(0, 1, nb);
            
            if (nargout > 1)
                nt = length(R);
                rocBins = zeros(nb, nt);
                rocVals = zeros(nb, nt);
                for i = 1:nt
                    s = timeseries(flipdim(R{i}(:,2),1),flipdim(R{i}(:,1),1));
                    s1 = resample(s, b);
                    rocVals(:,i) = s1.data;
                end
                extra.rocBins = b;
                extra.rocVals = mean(rocVals, 2);
            end
        end
        
    else
        warning('Unsupported similarity measure: %s', meas{im});
        sim(im) = nan;
    end
end
