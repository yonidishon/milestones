function topCands = candidatesTopSelect(cands, options)
% Selects a given number of top candidates. Usually used to track them
% later
%
% topCands = candidatesTopSelect(cands, options)
%
% INPUT
%   cands       cell array of candidates, each must have:
%       .score      candidate score. According to this the candidates are
%                   selected
%   options     options struct with
%       .topCandsUse    number of candidates to select

nc = length(cands);
if (nc >= options.topCandsUse) % select top scored
    scores = zeros(nc, 1);
    
    for ic = 1:nc
        scores(ic) = cands{ic}.score;
    end
    
    [~, idx] = sort(scores, 'descend');
    topCands = cands(idx(1:options.topCandsUse));
else % add dummy candidates
    dc = cands{1};
    dc.score = 0;
    
    topCands = cell(options.topCandsUse, 1);
    for i = 1:options.topCandsUse
        if (i <= nc)
            topCands{i} = cands{i};
        else
            topCands{i} = dc;
        end
    end
end
