% Locally Orderless Tracking main function
%
% Based on (this is the original implementation):
% [1] Locally Orderless Tracking. 
%     Shaul Oron, Aharon Bar-Hillel, Dan Levi and Shai Avidan
%	  Computer Vision and Pattern Recognition 2012
%
% The code provided here also uses and incluedes a re-distribution of the following code:
% - EMD mex downloaded from: http://www.mathworks.com/matlabcentral/fileexchange/12936-emd-earth-movers-distance-mex-interface
% Which is based on:
% [2] A Metric for Distributions with Applications to Image Databases. 
%     Y. Rubner, C. Tomasi, and L. J. Guibas.  ICCV 1998
% See also: http://ai.stanford.edu/~rubner/emd/default.htm
% - Turbopixels downloaded from: http://www.cs.toronto.edu/~babalex/research.html
% which is based on is based on:
% [3] TurboPixels: Fast Superpixels Using Geometric Flows. 
%     Alex Levinshtein, Adrian Stere, Kiriakos N. Kutulakos, David J. Fleet, Sven J. Dickinson, and Kaleem Siddiqi. TPAMI 2009
%
%
% [] = LocallyOrderlessTracking(param)
%
% Input:
%             param - Either a parameter file name of parameter structure
%                     if empty or not provided default parameters are
%                     loaded see 'loadDefaultParams.m' for more info.
%
% Getting started:
%       To get started you can simply run this function (i.e. LocallyOrderlessTracking)
%       A figure showing the first frame of the provided example sequence will open 
%       Mark the target with a rectangle and double-click to start tracking
%       Tracking results will be displayed in a new figure window
%       (*)  We suggest initially to use our parameter configuration
%       (**) If you have matlab parallel toolbox run matlabpool before starting for better performance
%
% This code is distributed under the GNU-GPL license.
%
% When using this code for academic purposes please cite [1]
%
% Writen by: Shaul Oron
% Last Updated: 19/04/2012
% shauloron@gmail.com
% Computer Vision Lab, Faculty of Engineering, Tel-Aviv University, Israel
%
% Modified by: Dmitry Rudoy
% Added support for:
%   param.scoreTh           score threshold. If total EMD score raises
%                           above this the tracking is stopped
%   param.minSize           tracking size threshold. If the size of the
%                           tracked rectangle drops below this (relatively
%                           to frame height) the tracking stops

function [tracking, score, validIdx] = LOT_trackTillFailure(param)

if ~isstruct(param)
    error('Param input is invalid should be file name or parameter structure');
end

% Load first frame
% DR, video
if (isempty(param.inputDir)) % use video reader
    [I] = loadFrameVideo(param.videoReader, param.initFrame, param.DS);
else % use image directory
    [I] = loadFrame(fullfile(param.inputDir,files(param.initFrame).name),param.DS);
end
Ihsv = rgb2hsv(I);

% Get image support
[maxY,maxX,~] = size(I);

% Check if traget position is defined o/w get from user
[target,T0] = setInitialTargetPosition(I,param);

% Initialize particles
w = ones(param.numOfParticles,1)/param.numOfParticles;
[particles,w] = predictParticles(ones(param.numOfParticles,1)*target,w,param,maxX,maxY);
ROI = getParticleROI(particles,[maxX maxY],param.dxySP);

% Calculate number of super pixels
param.nSP = round(prod(ROI(3:4))/prod(target(3:4))*param.targetSP);
param.nSP = max(min(param.nSP,param.maxSP),param.minSP);

% Perform oversegmentation i.e. superpixelization
idxImg = buildSuperPixelsIndexImage(I,param.nSP,ROI);

% Build template signature
[S0,W0,VL] = getSignatureFromSuperPixelImage(Ihsv,idxImg,target);


% Initialize required distance parameters
switch lower(param.emdDist)
    case 'gaussgauss'
        distParams.sigA = sqrt(param.priorVarA);
        distParams.sigL = sqrt(param.priorVarL);
    case 'gaussuniform'
        distParams.sigA = sqrt(param.priorVarA);
        distParams.RL = param.priorRL;
        distParams.alphaL = param.priorAlphaL;
        distParams.SL = VL;
    otherwise
        distParams = [];
end

% Allocate score array
scr = zeros(param.numOfParticles,1);

% DR, video
if (param.finFrame == -1 && ~isempty(param.inputDir)) 
    param.finFrame = length(files);
end

% DR
tracking = zeros(param.finFrame - param.initFrame + 1, 4); % tracked rectangles
tracking(1, :) = target;
score = zeros(param.finFrame - param.initFrame + 1, 1); % scores
validIdx = (1:param.finFrame - param.initFrame + 1)'; % indices of tracked frames

% Main Loop (Tracking)
k = 1;
for frame = param.initFrame+1:param.finFrame
    % DR
    if (isfield(param, 'debug') && param.debug >= 2)
        fprintf('----------------------- Frame #%d ----------------------\n',frame);
    end
    % Load new frame and partition to super pixels
    % DR, video
    if (isempty(param.inputDir)) % use video
        [I] = loadFrameVideo(param.videoReader, frame, param.DS);
    else % use images
        [I] = loadFrame(fullfile(param.inputDir,files(frame).name),param.DS);
    end
    
    Ihsv = rgb2hsv(I);  
    idxImg = buildSuperPixelsIndexImage(I,param.nSP,ROI);
        
    % Sample each particle from new frame and calc match score using EMD
    parfor p = 1:size(particles,1)
        % Get particle signature
        [S,W] = getSignatureFromSuperPixelImage(Ihsv,idxImg,particles(p,:));
        % Build cost matrix for EMD
        [C] = buildCostMatrix(S0,S,param.emdDist,distParams);
        % Calculate match score using EMD
        try
            [scr(p),flow] = emd_mex(W0',W',C);
        catch
            if (isfield(param, 'debug') && param.debug >= 2)
                fprintf('EMD error at particle %d\n',p);
            end
            scr(p) = inf;
        end  
    end    
    
    % Update particle weights
    w = exp(-scr(1:size(particles,1))*param.beta);
    w = w./sum(w);
        
    % Update target state integrating over all particles
    target = updateTargetState(target,particles,w,param,maxX,maxY);
    
    % stop if target includes NaNs or tracking all zeros
    if (any(isnan(target)) || target(3) == 0 || target(4) == 0)
        if (isfield(param, 'debug') && param.debug >= 1)
            fprintf('Tracking stopped at frame %d (target is NaN)\n', frame);
        end
        break;
    end

    % Get final target signature
    [S,W,VL] = getSignatureFromSuperPixelImage(Ihsv,idxImg,target);
    distParams.SL = VL;
    
    % Build cost matrix for EMD
    [C] = buildCostMatrix(S0,S,param.emdDist,distParams);
    
    % Calculate match score using EMD
    try
        [Tscr,flow] = emd_mex(W0',W',C);
        emdPass = 1;
    catch
%         warndlg(sprintf('EMD for final target has failed\n at frame # %d\n',frame));
        flow = [];
        emdPass = 0;
        Tscr = inf;
    end    
    
    if (isfield(param, 'debug') && param.debug >= 1)
        fprintf('Score: %f\n', Tscr);
    end
    
    % Update distance parameters
    if param.updateParams && emdPass
        switch lower(param.emdDist)
            case 'gaussgauss'
                [distParams] = estimateGaussGaussParams(S0,S,flow,param,distParams);
            case 'gaussuniform'
                [distParams] = estimateGaussUniformParams(S0,S,flow,param,distParams);            
        end
    end 
    
    % Update particles for next frame
    [particles,ROI] = updateParticles(particles,w,maxX,maxY,param);
        
    % DR, save results
    tracking(frame-param.initFrame+1, :) = target;
    score(frame-param.initFrame+1) = Tscr;
    
    % stop if score threshold exceeded
    if (isfield(param, 'scoreTh') && param.scoreTh > 0 && Tscr > param.scoreTh)
        if (isfield(param, 'debug') && param.debug >= 1)
            fprintf('Tracking stopped at frame %d (score is too high)\n', frame);
        end
        break;
    end
    
    % stop if rectangle is too small
    tsz = sqrt(target(3)^2 + target(4)^2);
    if (isfield(param, 'minSize') && param.minSize > 0 && tsz < maxY * param.minSize)
        if (isfield(param, 'debug') && param.debug >= 1)
            fprintf('Tracking stopped at frame %d (target is too small)\n', frame);
        end
        break;
    end
    
    k = k+1;
end % Main Loop

validIdx = validIdx(1:k);
tracking = tracking(validIdx, :);
score = score(validIdx);
