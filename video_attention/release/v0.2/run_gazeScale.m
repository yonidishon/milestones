% run_gazeScale
% sg.data is stored for each DIEM video previously, it is an
% <number_of_experiments> X <2> X <nuber_of_frames> array

%% settings
uncVideoRoot = fullfile(diemDataRoot, 'video_unc');
gazeRoot = fullfile(saveRoot, 'diem');
outGazeRoot = fullfile(diemDataRoot, 'gaze');
% gazeFile = fullfile(uncVideoRoot, '00_gaze.mat');

diemScreen = [1280 960];
kerRad = 52;
scaleFactor = 5;

diemScrCen = diemScreen / 2;

%% run
s = load(facesFile);
videos = videoListLoad(diemDataRoot);
faces = s.faces;
clear s;

for iv = 1:length(videos)
    fprintf('Scaling gaze for %s... ', videos{iv}); tic;
    
    % load gaze data
    sg = load(fullfile(gazeRoot, sprintf('%s.mat', videos{iv})));
    [ng, ~, nfr] = size(sg.data);

    % video size
    vr = VideoReader(fullfile(uncVideoRoot, sprintf('%s.avi', videos{iv})));
    
    gaze.height = vr.Height;
    gaze.width = vr.Width;
    gaze.length = nfr;
    gaze.data = cell(gaze.length, 1);
    
    frCen = [gaze.width, gaze.height] / 2;
    
    gazeData = 1/scaleFactor * (sg.data - repmat(diemScrCen, [ng, 1, nfr])) + repmat(frCen, [ng, 1, nfr]);
    
    for i = 1:nfr
        gazeLoc = gazeData(:,:,i);
        gaze.data{i} = gazeLoc(~isnan(gazeLoc(:,1)), :);
    end
    
    save(fullfile(outGazeRoot, sprintf('%s.mat', videos{iv})), 'gaze');
    fprintf('%f sec\n', toc);
end
