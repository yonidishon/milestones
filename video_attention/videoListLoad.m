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
%hosts = {'CGM-AYELLET-1', 'CGM7', 'CGM16', 'CGM22', 'CGM38', 'CGM44', 'CGM46', 'CGM47'};

if (strcmp(dataSet, 'DIEM'))
    videos = importdata(fullfile(dataRoot, 'list.txt'));
    % STATIC ALLOCATION OF VIDEOS PER HOSTS
    %nv = length(videos);
    %nh = length(hosts);
    %vPerHost =  ones(nh, 1) * floor(nv/nh);
    %vLeft = nv - sum(vPerHost);
    %vPerHost(1:vLeft) = vPerHost(1:vLeft) + 1;
    %assert(nv  == sum(vPerHost));
    %vIndPerHost = [1; cumsum(vPerHost)];
%     for h = 1:nh
%         hostname = char( getHostName( java.net.InetAddress.getLocalHost ));
%         if strcmpi(hostname, hosts{h})
%             videos = videos([vIndPerHost(h) : vIndPerHost(h+1)])
%         end;
%     end;
else
    error('Unsupported dataset %s', dataSet);
end
