--------------------------------------------------------------------------------------------------
-- Matlab code for "Learning video saliency from human gaze using candidate selection", CVPR'13 --
--------------------------------------------------------------------------------------------------

DISCLAMER

If you find this code useful, or compare to it in your paper please cite:
@inproceedings{rudoylearning,
  author    = {Dmitry Rudoy and
               Dan B Goldman and
               Eli Shechtman and
               Lihi Zelnik-Manor},
  title     = {Learning video saliency from human gaze using candidate selection},
  booktitle = {CVPR},
  year      = {2013},
}

The paper, the code and others are available from http://dimarudoy.co.nf/publications/cvpr2013/video_saliency_project.html 
For question please email to Dmitry Rudoy, dmitry.rudoy@gmail.com

INSTALL
1. In configureDetectors define paths: faceDetectorPath, poseletsRoot



This code is checked on Matlab 2012b, but might work on earlier versions as well
To use the code you will need to download and install the following packages:

* Pyotr Dollar toolbox v2.6.1 from http://vision.ucsd.edu/~pdollar/toolbox/doc/. The latest version might work, but I did not check.
* MeanShiftCluster from http://www.mathworks.com/matlabcentral/fileexchange/10161-mean-shift-clustering. Download and put in your Matlab path.
* Gaussian2D from http://thermo.as.arizona.edu/hp4d1_gmt1/NEW%20SCOTS%20CODES%20FOR%20MIRROR%20LAB/Gaussian2D.m. Add to your Matlab path. You can also copy and paste from http://jila.colorado.edu/bec/BEC_for_everyone/matlabfitting.htm if the above does not work (please name the file Gaussian2D.m and put in your Matlab path)
* randomforest-matlab from https://code.google.com/p/randomforest-matlab/. Follow the original installation instructions.
* Metrics AUC, NSS and CC are from https://sites.google.com/site/saliencyevaluation/. Follow their installation.
* GBVS code from http://www.klab.caltech.edu/~harel/share/gbvs.php
	The function makeGBVSParams should be updated as following:
	Add at the beginning of the function the lines:
	p = {};
	p.pathroot = fileparts(which('makeGBVSParams'));

* PQFT_2. Download it from http://www.mathworks.com/matlabcentral/fileexchange/36599-saliency-map-based-on-phase-quaternion-fourier-transform/ and put in your Matlab path.
* How2008 from http://www.klab.caltech.edu/~xhou/papers/nips08matlab.zip. Download and put in your Matlab path.
* Optical flow from Ce Liu. Download from http://people.csail.mit.edu/celiu/OpticalFlow/ and follow the intallation instructions.
* Viola-Jones face detector from http://www.mathworks.se/matlabcentral/fileexchange/19912-open-cv-viola-jones-face-detection-in-matlab. Please follow their installation procedure.
* Poselets. Download from http://www.eecs.berkeley.edu/~lbourdev/poselets/ and follow the installation instructions.


USING THE MODEL
For example usage see example_CRCNS.m. You're welcome to modify it as you wish.
Before using the model you should define the following variables (see example_CRCNS.m):
crcnsRoot = <path_to_downloaded_crcns>
crcnsOrigRoot = <path_to_CRCNS_original>, filled by run_convertGazeCRCNS
crcnsMtvRoot = <path_to_CRCNS_MTV>, filled by run_convertGazeCRCNS
diemDataRoot = <path_to_DIEM>

-- DATA --
The datasets are not provided and should be downloaded separately. Both DIEM and CRCNS are available online, see citations in out paper. 
Supported data sets: DIEM, CRCNS. CRCNS is used for this example.
Dataset directory structure:
<root>
	list.txt: a list of all videos in the set, line per video, without extension
	video_unc: directory with uncompressed videos (*.avi)
	gaze: folder with preprocesses matlab files that include the gaze tracking data. Created by run_convertGazeCRCNS (CRCNS) and comes with DIEM
	cache: folder that used for caching, do not modify


-- CRCNS --
* run_convertGazeCRCNS: converts the CRCNS data to our format, uses crcnsRoot as input and crcnsOrigRoot / crcnsMtvRoot as output
* precalc_GazeDataCRCNS: prepalculates all the gaze information for CRCNS data set and puts it in cache.	

-- DIEM --
* precalc_GazeData: prepalculates all the gaze information for DIEM data set and puts it in cache.	

-- Running --
The code is provided together with the trained model (files under data directory). But if you wish to re-train please run:
* run_jumpTrain: trains the model on DIEM set, uses diemDataRoot

To test the model on DIEM run:
* run_jumpTest: test the model on 64 movies from DIEM.

Enjoy,
Dmitry
