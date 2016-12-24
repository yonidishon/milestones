function SM = SpaceTimeSaliencyMap(varargin)
% Compute Space-time Self-Resemblance

% [RETURNS]
% SM   : Space-time Saliency Map
%
% [PARAMETERS]
% img   : Input image
% LARK  : A collection of LARK descriptors
% param : parameters

% [HISTORY]
% Apr 25, 2011 : created by Hae Jong

Seq = varargin{1};
LARK = varargin{2};
wsize = varargin{3};
size_t = varargin{4};
sigma = varargin{5};

win = (wsize-1)/2;
win_t = (size_t-1)/2;

% To avoid edge effect, we use mirror padding.
for i = 1:size(LARK,4)
    LARK1(:,:,:,i) = EdgeMirror3(LARK(:,:,:,i),[win,win,win_t]);   
end

% Precompute Norm of center matrices and surrounding matrices
for i = 1:size(LARK,1)
    for j = 1:size(LARK,2)
        for l = 1:size(LARK,3)
            norm_C(i,j,l) = norm(squeeze(LARK(i,j,l,:)));
        end
    end
end
norm_S(:,:,:) = EdgeMirror3(norm_C,[win,win,win_t]);


[M,N,T] = size(Seq);

Center = reshape(LARK,[size(LARK,1)*size(LARK,2)*size(LARK,3) size(LARK,4)]);
norm_C = reshape(norm_C,[size(LARK,1)*size(LARK,2)*size(LARK,3) 1]);


cnt = 0;
SM = zeros(size(norm_C));
for i = 1:wsize
    for j = 1:wsize
        for l = 1:size_t
            cnt = cnt + 1;
            temp = sum(Center.*reshape(LARK1(i:i+size(LARK,1)-1,j:j+size(LARK,2)-1,l:l+size(LARK,3)-1,:),[size(LARK,1)*size(LARK,2)*size(LARK,3) size(LARK,4)]),2); % compute inner product between a center and surrounding matrices
            temp = temp./(norm_C.*reshape(norm_S(i:i+size(LARK,1)-1,j:j+size(LARK,2)-1,l:l+size(LARK,3)-1),[size(LARK,1)*size(LARK,2)*size(LARK,3) 1])); 
            SM = SM+exp( (-1+temp)./(sigma^2)); % compute self-resemblance using matrix cosine similarity
        end
    end
end
SM = 1./SM; %Final saliency map values

SM = reshape(SM,[size(LARK,1) size(LARK,2) size(LARK,3)]);

end