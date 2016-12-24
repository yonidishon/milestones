function data = importCrowdGazeImage(dataRoot, targetSize, verb)
% TODO doc

if (~exist('verb', 'var'));
    verb = false;
end

files = dir(fullfile(dataRoot, '*.xml'));
nf = length(files);
data.imageNames = {};
data.gaze = {};
data.error = {};

for i = 1:nf
    fname = fullfile(dataRoot, files(i).name);

    if (verb)
        fprintf('Parsing %s (%d/%d)...', files(i).name, i, nf);
        tic;
    end
    
    xmlDoc = xmlread(fname);

    % import data
    lg = getLetterGrid(xmlDoc);
    lgSz = [lg.width, lg.height];
    scale = max(targetSize ./ lgSz);
    lgC = lgSz ./ 2;
    tC = targetSize ./ 2;
    
    vname = char(getVideoName(xmlDoc));
    [~, loc, err, ~] = getUserInput(xmlDoc);
    loc = (loc - lgC) * scale + tC;
    err = err * scale;
    idx = find(strcmp(data.imageNames, vname));
    if (isempty(idx)) % new image
        data.imageNames = [data.imageNames; vname];
        data.gaze = [data.gaze; loc];
        data.error = [data.error; err];
    else
        data.gaze{idx} = [data.gaze{idx}; loc];
        data.error{idx} = [data.error{idx}; err];
    end
%     [hid, wid, aid] = getHITData(xmlDoc);
%     sessionId = getSessionData(xmlDoc);
%     [isTutorial, loc] = getTutorialData(xmlDoc);
%     adata.data{i}.letterGrid = getLetterGrid(xmlDoc);
%     adata.data{i}.input = getUserInput(xmlDoc);
%     [adata.data{i}.input, adata.data{i}.userLocation, adata.data{i}.userError, uvalid] = getUserInput(xmlDoc);
%     [adata.data{i}.clip, adata.data{i}.stopTime, adata.data{i}.trialNumber] = getClipData(xmlDoc);
%     [adata.data{i}.browserName, adata.data{i}.browserSize, adata.data{i}.screenSize] = getAssignmentData(xmlDoc);
% 
%     % assignment data
%     adata.data{i}.saveDateNum = files(i).datenum;
%     adata.data{i}.HITID = hid;
%     adata.data{i}.workerID = wid;
%     adata.data{i}.assignmentID = aid;
%     adata.data{i}.sessionID = sessionId;
%     adata.data{i}.isTutorial = isTutorial;
%     adata.data{i}.location = loc;
%     
%     % global data
%     adata.assignmentIds{i} = aid;
%     adata.workerIds{i} = wid;
%     adata.sessionIds{i} = sessionId;
%     adata.isTutorial(i) = isTutorial;
    
    if (verb)
        fprintf(' %f sec\n', toc);
    end
end
