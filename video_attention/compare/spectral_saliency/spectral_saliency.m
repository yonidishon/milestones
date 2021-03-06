function [S,P,M]=spectral_saliency(I,residual_filter_length)
  % SPECTRAL_SALIENCY implements the calculation of the visual saliency
  %   using pure spectral whitening (residual_filter_length=0) or
  %   the spectral residual (residual_filter_length>0).
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
  %       ("simple" multi-channel and quaternion-based)
  %   [4] B. Schauerte, J. Richarz, G. A. Fink,"Saliency-based 
  %       Identification and Recognition of Pointed-at Objects," in IROS,
  %       2010.
  %       (uses multi-channel on intensity, blue-yellow/red-green opponent)
  %   [5] B. Schauerte, G. A. Fink, "Focusing Computational Visual 
  %       Attention in Multi-Modal Human-Robot Interaction," in Proc. ICMI,
  %       2010
  %       (extended multi-scale and neuron-based approach that allows
  %        to incorporate information about the visual search target)
  %
  %   However, the underlying principle has been addressed long before:
  %   [6] A. Oppenheim and J. Lim, “The importance of phase in signals,”
  %       in Proc. IEEE, vol. 69, pp. 529–541, 1981.
  % 
  % @author: B. Schauerte
  % @date:   2009-2011
  
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
  
  assert(size(I,3)==1);
  
  if nargin<2
    residual_filter_length=0; 
  else
    assert(residual_filter_length>=0); 
  end
  
  FI = fft2(I);   % Fourier-transformed image representation
  P = angle(FI);  % Phase
  M = abs(FI);    % Magnitude
  if ~residual_filter_length
    % perform pure spectral whitening (see [2] and [1])
    S = abs(ifft2(exp(1i*P))).^2; % SP seems not to be the most important part
  else
    % perform spectral residual (see [1])
    L  = log(M);
    SR = L - imfilter(L, fspecial('average', residual_filter_length), 'replicate');
    S  = abs(ifft2(exp(SR + 1i*P))) .^ 2;
  end
