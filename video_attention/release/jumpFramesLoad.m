function [jf, before, after] = jumpFramesLoad(dataRoot, videoIndex, type, varargin)
% loads information about jump frames
%
% [jf, before, after] = jumpFramesLoad(dataRoot, type)
%
% INPUT
%   dataRoot    
%   videoIndex  
%   type        
%
% OUTPUT
%   jf          
%   before      
%   after       

if (strcmp(type, 'cut'))
    facesFile = fullfile(dataRoot, 'video_unc', '00_faces.mat');
    s = load(facesFile);
    jf = s.faces{videoIndex}.cuts;
    before = -1;
    after = 15;
elseif (strcmp(type, 'cut+neg'))
    facesFile = fullfile(dataRoot, 'video_unc', '00_faces.mat');
    s = load(facesFile);
    jf = s.faces{videoIndex}.cuts;
    if (~isempty(jf))
        njf = round((jf(2:end) + jf(1:end-1)) / 2);
        jf = sort([jf; njf]);
    end
    before = -1;
    after = 15;
elseif (strcmp(type, 'gaze_jump'))
    jumpLen = 10;
    videos = videoListLoad(dataRoot, 'DIEM');
    jumpFile = fullfile(dataRoot, 'cache', '00_tracking_deviation', sprintf('%s.mat', videos{videoIndex}));
    s = load(jumpFile);
    d = [0; diff(s.jumpFrames)];
    jf = s.jumpFrames(d > jumpLen);
    
    before = -5;
    after = 10;
elseif (strcmp(type, 'random'))
    frNum = varargin{1};
    frSt = varargin{2};
    frEn = varargin{3};
    jf = sort(floor((frEn-frSt) * rand(frNum, 1) + frSt));
    
    before = -1;
    after = 0;
elseif (strcmp(type, 'all'))
%     frNum = varargin{1};
    frSt = varargin{2};
    frEn = varargin{3};
    jf = frSt:frEn;
    
    before = -1;
    after = 0;
else
    error('Not supported jump frames type: %s', type);
end
