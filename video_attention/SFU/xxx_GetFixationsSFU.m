% Written by S. Hossein Khatoonabadi (skhatoon@sfu.ca)
%
% get fixations of a sequence as a 3-D discrete map
%
% Input
%     SEQ_DIR: (string) root directory
%     SEQ_NAME: (string) input sequence
%     FRMS_CNT: (integer value) number of frames in the sequence
%     IMG_H: (integer value) number of horizontal pixels (height)
%     IMG_W: (integer value) number of vertical pixels (width)
%
% Output
%     fixMap1: (3-D matrix) fixation map of the first viewing
%     fixMap2: (3-D matrix) fixation map of the second viewing

function [fixMap1] = xxx_GetFixationsSFU(SEQ_DIR,SEQ_NAME,FRMS_CNT)

% number of subjects: 15
fixations1 = zeros(FRMS_CNT,15,2); % (the number of frames, x and y)
numFix1 = zeros(FRMS_CNT,1); % the number of valid fixations for view 1
filesep ='\';
load([SEQ_DIR SEQ_NAME filesep 'gazemask.mat'])
gazeMask = logical(gazeMask);
load([SEQ_DIR SEQ_NAME filesep 'gazeloc.mat']);

% gazeMask = logical(xlsread([SEQ_DIR SEQ_NAME filesep 'gazemask.csv']));
% gazeLoc = xlsread([SEQ_DIR SEQ_NAME filesep 'gazeloc.csv']);

for frame=1:FRMS_CNT
    N1 = 0; 
    for i=1:4:4*15
        flag = gazeMask(frame,i:i+3);
        if flag(1)
            N1 = N1+1;
            fixations1(frame,N1,1) = gazeLoc(frame,i+1)/2;
            fixations1(frame,N1,2) = gazeLoc(frame,i)/2;
        end
    end
    numFix1(frame) = N1;
end

% create a map of fixations
fixMap1 = cell(FRMS_CNT,1);
for frame=1:FRMS_CNT
fixMap1{frame} = [fixations1(frame,1:numFix1(frame),1);fixations1(frame,1:numFix1(frame),2)];
end
