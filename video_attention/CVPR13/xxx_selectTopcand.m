function [topcands] = xxx_selectTopcand(cands,topnum)
% Date: 18/11/2016
% Author: Yonatan Dishon.
% Input::
%   cands = cell array of frame cands
%   topnum = int - number of candidate to filter out of cands according to
%       cands score
% Output::
%   topcands - cell array of only topcands
topcands = cell(length(cands),1);
for ii =1:length(cands) % all frames
    filt_idx = zeros(length(cands{ii}),1);
    % Static 
    for jj = 1:length(cands{ii})
        if ismember(cands{ii}{jj}.type,[5]) 
            filt_idx(jj)=1;
        end 
    end
    if ~any(filt_idx) % This should never happend TODO check why it happens
        %scands = innertop([],topnum); 
        scands = cands{ii}; % not really true but will be okay by now
    else
        scands = innertop(cands{ii}(find(filt_idx)),topnum);
    end
    
    % motion
    filt_idx = zeros(length(cands{ii}),1);
    for jj = 1:length(cands{ii})
        if ismember(cands{ii}{jj}.type,[4])
            filt_idx(jj)=1;
        end
    end
    if ~any(filt_idx)
        mcands = innertop([],topnum); 
    else
        mcands = innertop(cands{ii}(find(filt_idx)),topnum);
    end
    topcands{ii}=[scands,mcands];
end
end

function [selcand] = innertop(innercands,numtop)
if isempty(innercands)
    %warning('xxx_selectTopcand:: Candidates are empty');
    selcand =[];
    return;
end
if length(innercands) <= numtop
    selcand = innercands;
else    
    scores = cellfun(@(x)extractfield(x,'score'),innercands);
    [~, swi] = sort(scores, 'descend');
    idx = swi(1:numtop);
    selcand = innercands(idx);
end
end