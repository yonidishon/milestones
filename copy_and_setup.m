%clc; 
clear all; 
close all force;
rmdir('C:\Users\gleifman\My Documents\DimaCode\DepthDB', 's');
copyfile('Z:\RGB-D\DimaCode\DepthDB', 'C:\Users\gleifman\My Documents\DimaCode\DepthDB')
cd Z:\RGB-D\DimaCode\Code4George\Dropbox\Matlab; 
setup_videoSaliencyLenovo;
cd video_attention\; 
precalc_GazeData; 
run_trackGazeDev;
cd video_attention\CVPR13\;
cvpr13_jumpTrain;
cvpr13_jumpTest;

