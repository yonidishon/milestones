% script to copy all SFU from CGM10 to CGM47
ROOTDIR ='\\cgm10\Users\ydishon\Downloads\howManyBitsforSaliency\DATA\SEQ_SFU';
DSTROOTDIR = 'D:\Competition_Dataset\SFU\ResultsCompressedAlgo';
FOLDS = dir(ROOTDIR);
ISDIR = cell2mat(extractfield(FOLDS,'isdir'));
FOLDS = FOLDS(ISDIR);FOLDSNM=extractfield(FOLDS,'name');
FOLDSNM = FOLDSNM(3:end);
for ii = 1:length(FOLDSNM)
    if ~exist(fullfile(DSTROOTDIR,FOLDSNM{ii}),'dir')
        mkdir(fullfile(DSTROOTDIR,FOLDSNM{ii}));
    end
    resfiles = extractfield(dir(fullfile(ROOTDIR,FOLDSNM{ii},'result*.mat')),'name');
    for jj=1:length(resfiles)
       copyfile(fullfile(ROOTDIR,FOLDSNM{ii},resfiles{jj}),fullfile(DSTROOTDIR,FOLDSNM{ii},resfiles{jj})); 
    end
end