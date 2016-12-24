% cvpr13_figJuddGaze
%% settings
juddRoot = 'C:\Users\dmitryr\Documents\Datasets\Judd';
jpgRoot = fullfile(juddRoot, 'ALLSTIMULI');
dataRoot = fullfile(juddRoot, 'DATA');
outRoot = fullfile(diemDataRoot, 'cvpr13');

addpath(fullfile(juddRoot, 'DatabaseCode'));

imgName = 'i1267668332';
numFix = 8;
sigma = 10;
ncol = 256;
cmap = jet(ncol);

users = {'CNG', 'ajs', 'emb', 'ems', 'ff', 'hp', 'jcw', 'jw', 'kae', 'krl', 'po', 'tmj', 'tu', 'ya', 'zb'};

%% run
img = imread(fullfile(jpgRoot, sprintf('%s.jpeg', imgName)));
[m, n, ~] = size(img);
gazePts = [];

for j = 1:length(users)
    user = users{j};
    
    % Get eyetracking data for this image
    datafolder = fullfile(dataRoot, user);
    
    datafile = strcat(imgName, '.mat');
    load(fullfile(datafolder, datafile));
    stimFile = eval([datafile(1:end-4)]);
    eyeData = stimFile.DATA(1).eyeData;
    [eyeData Fix Sac] = checkFixations(eyeData);
    s=find(eyeData(:, 3)==2, 1)+1; % to avoid the first fixation
    eyeData=eyeData(s:end, :);
    fixs = find(eyeData(:,3)==0);
    
    % Add numbers and initials to indicate which and whos fixation is displayed
    if (numFix <= length(Fix.medianXY))
        appropFix = floor(Fix.medianXY(2:numFix, :));
        
        gazePts = [gazePts; appropFix(:, 1:2)];
    end
end

% binMap = points2binaryMap(gazePts, [n, m]);
prob = points2GaussMap(gazePts', ones(1, size(gazePts, 1)), 0, [n, m], sigma * m / 144);

alphaMap = 0.7 * repmat(prob, [1 1 3]);
rgbHM = ind2rgb(round(prob * ncol), cmap);
gim = rgb2gray(img);
gim = imadjust(gim, [0; 1], [0.3 0.7]);
gf = repmat(gim, [1 1 3]);
frout = rgbHM .* alphaMap + im2double(gf) .* (1-alphaMap);

imwrite(frout, fullfile(outRoot, 'img_gaze.png'));
