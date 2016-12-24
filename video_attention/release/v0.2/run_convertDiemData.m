% ADOBE PROPRIETARY INFORMATION
% 
% Use is governed by the license in the attached LICENSE.TXT file
% 
% 
% Copyright 2011 Adobe Systems Incorporated
% All Rights Reserved.
% 
% NOTICE:  All information contained herein is, and remains
% the property of Adobe Systems Incorporated and its suppliers,
% if any.  The intellectual and technical concepts contained
% herein are proprietary to Adobe Systems Incorporated and its
% suppliers and may be covered by U.S. and Foreign Patents,
% patents in process, and are protected by trade secret or copyright law.
% Dissemination of this information or reproduction of this material
% is strictly forbidden unless prior written permission is obtained
% from Adobe Systems Incorporated.
% 

% run_convertDiemData
%% settings
matlabDiemRoot = fullfile(saveRoot, 'diem'); % w/o saccades
useSingleExp = true;

%% video files
files = dir(fullfile(diemDataRoot, '*.txt'));
nfl = length(files);
videos = cell(nfl, 1);
for ifl = 1:nfl
    und = find(files(ifl).name == '_');
    st = und(2) + 1;
    en = find(files(ifl).name == '.', 1, 'last') - 1;
    videos{ifl} = files(ifl).name(st:en);
end

uniqueVideos = unique(videos);
nuv = length(uniqueVideos);

%% convert
for iv = 1:nuv
    fprintf('Converting %s...', uniqueVideos{iv});
    tic;
    
    idx = find(strcmp(uniqueVideos{iv}, videos));
    
    % filter multiple experiments
    if (useSingleExp)
        nf = length(idx);
        expNum = zeros(nf, 1);
        for i = 1:nf
            expNum(i) = str2double(files(idx(i)).name(5));
        end
        snum = sort(expNum);
        idx = idx(expNum == snum(end));
        fprintf(' using %d points from take %d...', length(idx), snum(end));
    end
    
    nf = length(idx);
    
    % read each file
    dt = cell(nf, 1);
    nfr = 0;
    for ifl = 1:nf
        dt{ifl} = load(fullfile(diemDataRoot, files(idx(ifl)).name));
        nfr = max(nfr, dt{ifl}(end, 1));
    end
    
    data = nan(nf, 2, nfr);
    for ifl = 1:nf
        vidx = (dt{ifl}(:, 5) == 1 & dt{ifl}(:, 9) == 1);
        n = length(vidx);
        x = (dt{ifl}(:, 2) + dt{ifl}(:, 6)) / 2;
        y = (dt{ifl}(:, 3) + dt{ifl}(:, 7)) / 2;
        data(ifl, 1, 1:n) = reshape(x, [1 1 n]);
        data(ifl, 1, ~vidx) = nan;
        data(ifl, 2, 1:n) = reshape(y, [1 1 n]);
        data(ifl, 2, ~vidx) = nan;
    end
    
    % save
    save(fullfile(matlabDiemRoot, sprintf('%s.mat', uniqueVideos{iv})), 'data');
    
    fprintf(' %f sec\n', toc);
end

