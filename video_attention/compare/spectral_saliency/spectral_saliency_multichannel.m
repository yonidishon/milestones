function S=spectral_saliency_multichannel(I,imsize,multichannel_method,smap_smoothing_filter_size,cmap_normalization,do_figures,do_channel_image_mattogrey)
  % SPECTRAL_SALIENCY_MULTICHANNEL is a (simple) multichannel
  %   implementation of the spectral saliency calculation for images.
  %
  %   The selected image size (imsize) at which the saliency is calculated 
  %   is the most important parameter. Just try different sizes and you 
  %   will see ...
  %
  %   There are two methods (multichannel_method) to calculate the 
  %   multichannel saliency:
  %   'simple':     Calculates the single-channel saliency for each channel
  %                 separately and then averages the result.
  %   ...:          I am currently preparing an extended version and update
  %                 (e.g., an extended implementation using Quaternions). 
  %                 Until then (because the internal interfaces change quite
  %                 frequently at the moment), I disabled this functionality.
  %                 Contact me, if you urgently need, e.g., a Quaternion-based
  %                 or multi-scale implementation.
  %                 
  %   Usage examples:
  %   - spectral_saliency_multichannel(imread(..some image path..))
  %     or as an example for other color spaces (e.g. ICOPP, Lab, ...)
  %   - spectral_saliency_multichannel(rgb2icopp(imread(..some image path..)))
  %
  %   Notes:
  %   - I kept the implementations as focused and simple as possible and
  %     thus they lack more advanced functionality, e.g. more complex 
  %     normalizations. However, I think that the provided functionality is
  %     more than sufficient for (a) people who want to get started in the
  %     field of visual attention (especially students), (b) practitioners
  %     who have heard about the spectral approach and want to try it, and
  %     (c) people who just need a fast, reliable, well-established visual 
  %     saliency algorithm (with a simple interface and not too many
  %     parameters) for their applications.
  %
  %   For more details on the method see:
  %   [1] X. Hou and L. Zhang, "Saliency Detection: A Spectral Residual
  %       Approach", in CVPR, 2007.
  %       (original paper)
  %   [2] C. Guo, Q. Ma, and L. Zhang, “Spatio-temporal saliency detection
  %       using phase spectrum of quaternion fourier transform,” in CVPR, 
  %       2008.
  %       (extension to quaternions; importance of the residual)
  %
  %   It has been applied quite a lot through the last years, e.g., see:
  %   [3] B. Schauerte, B. Kühn, K. Kroschel, R. Stiefelhagen, "Multimodal 
  %       Saliency-based Attention for Object-based Scene Analysis," in 
  %       IROS, 2011.
  %       ("simple" multi-channel and quaternion-based; Isophote-based
  %        saliency map segmentation)
  %   [4] B. Schauerte, J. Richarz, G. A. Fink,"Saliency-based 
  %       Identification and Recognition of Pointed-at Objects," in IROS,
  %       2010.
  %       (uses multi-channel on intensity, blue-yellow/red-green opponent)
  %   [5] B. Schauerte, G. A. Fink, "Focusing Computational Visual 
  %       Attention in Multi-Modal Human-Robot Interaction," in Proc. ICMI,
  %       2010
  %       (extended to a multi-scale and neuron-based approach that allows
  %        to incorporate information about the visual search target)
  %
  %   However, the underlying principle has been addressed long before:
  %   [6] A. Oppenheim and J. Lim, “The importance of phase in signals,”
  %       in Proc. IEEE, vol. 69, pp. 529–541, 1981.
  % 
  % @author: B. Schauerte
  % @date:   2009-2011
  % @url:    http://cvhci.anthropomatik.kit.edu/~bschauer/
  
  % Copyright 2009-2011 B. Schauerte. All rights reserved.
  % 
  % Redistribution and use in source and binary forms, with or without 
  % modification, are permitted provided that the following conditions are 
  % met:
  % 
  %    1. Redistributions of source code must retain the above copyright 
  %       notice, this list of conditions and the following disclaimer.
  % 
  %    2. Redistributions in binary form must reproduce the above copyright 
  %       notice, this list of conditions and the following disclaimer in 
  %       the documentation and/or other materials provided with the 
  %       distribution.
  % 
  % THIS SOFTWARE IS PROVIDED BY B. SCHAUERTE ''AS IS'' AND ANY EXPRESS OR 
  % IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
  % WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
  % DISCLAIMED. IN NO EVENT SHALL B. SCHAUERTE OR CONTRIBUTORS BE LIABLE 
  % FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
  % CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
  % SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR 
  % BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
  % WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
  % OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
  % ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
  % 
  % The views and conclusions contained in the software and documentation
  % are those of the authors and should not be interpreted as representing 
  % official policies, either expressed or implied, of B. Schauerte.

  %if nargin<2, imsize=1; end % use original image size
  if nargin<2, imsize=[NaN 64]; end
  if nargin<3, multichannel_method='simple'; end
  if nargin<4, smap_smoothing_filter_size=3; end
  if nargin<5, cmap_normalization=1; end
  if nargin<6, do_figures=true; end
  if nargin<7, do_channel_image_mattogrey=true; end
  
  if ~isfloat(I)
    I=im2double(I);
  end
  imorigsize=size(I);
  IR=imresize(I, imsize, 'bilinear');
  
  nchannels=size(IR,3);
  channel_saliency=zeros(size(IR));
  channel_phase=zeros(size(IR));
  channel_magnitude=zeros(size(IR));
  channel_saliency_smoothed=zeros(size(IR));
  
  switch multichannel_method
    % "simple" single-channel and averaging
    case {'simple'}
      % calculate "saliency" for each channel
      for i=1:1:nchannels
        [channel_saliency(:,:,i),channel_phase(:,:,i),channel_magnitude(:,:,i)]=spectral_saliency(IR(:,:,i));
        channel_saliency_smoothed(:,:,i)=imfilter(channel_saliency(:,:,i), fspecial('disk', smap_smoothing_filter_size));
        switch cmap_normalization % simple (range) normalization
          case {1}
            % simply normalize the value range
            cmin=min(channel_saliency_smoothed(:));
            cmax=max(channel_saliency_smoothed(:));
            if (cmin - cmax) > 0
              channel_saliency_smoothed=(channel_saliency_smoothed - cmin) / (cmax - cmin);
            end

          case {0}
            % do nothing
            
          otherwise
            error('unsupported normalization')
        end
      end
          
      % uniform linear combination of the channels
      S=mean(channel_saliency_smoothed,3);
          
      if do_figures
        figure('name','Saliency / Channel');
        for i=1:1:nchannels
          subplot(4,nchannels,0*nchannels+i);
          if do_channel_image_mattogrey
            subimage(mat2gray(IR(:,:,i))); 
          else
            subimage(IR(:,:,i));
          end
          title(['Channel ' int2str(i)]);
        end
        for i=1:1:nchannels
          subplot(4,nchannels,1*nchannels+i);
          subimage(mat2gray(channel_saliency_smoothed(:,:,i))); 
          title(['Channel ' int2str(i) ' Saliency']);
        end
        for i=1:1:nchannels
          subplot(4,nchannels,2*nchannels+i);
          subimage(mat2gray(channel_phase(:,:,i))); 
          title(['Channel ' int2str(i) ' Phase']);
        end
        for i=1:1:nchannels
          subplot(4,nchannels,3*nchannels+i);
          subimage(mat2gray(channel_magnitude(:,:,i))); 
          title(['Channel ' int2str(i) ' Magnitude']);
        end
      end
  
    % quaternion-based
    %case {'quaternion'}
    %  S=spectral_saliency_quaternion(IR);
    %  S=imfilter(S, fspecial('disk', smap_smoothing_filter_size));

    otherwise
      error('unsupported multichannel saliency calculation mode')
  end
  
  if do_figures
    figure('name','Saliency');
    subplot(1,2,1); imshow(I);
    subplot(1,2,2); imshow(mat2gray(imresize(S, imorigsize(1:2))));
  end
