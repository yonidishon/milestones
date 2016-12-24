function [cands] = xxx_sourceCandidates_onlysimple(fr, options)
% replaces sourceCands
%
%   options     options for calculation, used fields:
%       .nSample    number of points within gaze region ('random')
%       .posPer     positive percentage
%       .negPer     negative percentage ('random')
%       .rectSzTh   rectangles below this threshold are rejected ('rect')
%   type        candidate creation type, supported
%       'random'    candidates are randomly sampled from options.posPer
%                   percentage of gaze map. options.nSample number of
%                   candidates is created
%       'rect'      candidates are created by converting the gaze map to
%                   rectangles. Points are sampled at rectangle center.
%                   This does not creates negative samples (yet)
%       'cand'      
         maps = cat(3, (fr.ofx.^2 + fr.ofy.^2), fr.saliency);
        cands = xxx_jumpCandidates3simple(maps, options); %Yonatan - changed this line from jumpCnadidates to jumpCandidates3            
end
