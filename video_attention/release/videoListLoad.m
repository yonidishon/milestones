function videos = videoListLoad(dataRoot, dataSet)
% Imports the list of videos in a data set
% 
% videos = videoListLoad(dataRoot, dataSet)
% videos = videoListLoad(dataRoot)
%
% INPUT
%   dataRoot    root of the dataset
%   dataSet     [DIEM] which dataset is this. Supported values are: DIEM
%
% OUTPUT
%   videos      cell array of video names

if (~exist('dataSet' , 'var'))
    dataSet = 'DIEM';
end

if (strcmp(dataSet, 'DIEM'))
    videos = importdata(fullfile(dataRoot, 'list.txt'));
else
    error('Unsupported dataset %s', dataSet);
end
