function LARK = ThreeDLARK(varargin)

% Compute 3-D LARK descriptors

% [RETURNS]
% LARK   : 3-D LARK descriptors
%
% [PARAMETERS]
% input   : Input sequence
% wsize : LARK window size
% alpha : Sensitivity parameter
% h     : smoothing paramter

% [HISTORY]
% Apr 25, 2011 : created by Hae Jong

input = varargin{1};
param = varargin{2};

wsize = param.wsize;
h = param.h;
wsize_t = param.wsize_t;
alpha = param.alpha;

% Gradient calculation
[zx,zy,zt] = gradient(input);

[M,N,T] = size(input);
win = (wsize-1)/2;
win_t = (wsize_t-1)/2;
zx = EdgeMirror3(zx,[win,win,win_t]);
zy = EdgeMirror3(zy,[win,win,win_t]);
zt = EdgeMirror3(zt,[win,win,win_t]);


[x1,x2,x3] = meshgrid(-win:win,-win:win,-win_t:win_t);

for k = 1:wsize_t
    K(:,:,k) = fspecial('disk',win);
end
K = permute(K,[1 3 2]);
for k = 1:wsize_t
    K(:,:,k) = K(:,:,k).*fspecial('disk',win);
end
K = permute(K,[3 2 1]);
for k = 1:wsize_t
    K(:,:,k) = K(:,:,k).*fspecial('disk',win);
end
K = permute(K,[1 3 2]);

for k = 1:wsize_t
K(:,:,k) = K(:,:,k)./K(win+1,win+1,k);
end
lambda = 1;
% Covariance matrices computation
for i = 1 : M
    for j = 1 : N
        for k = 1 : T
            gx = zx(i:i+wsize-1, j:j+wsize-1, k:k+wsize_t-1).*K;
            gy = zy(i:i+wsize-1, j:j+wsize-1, k:k+wsize_t-1).*K;
            gt = zt(i:i+wsize-1, j:j+wsize-1, k:k+wsize_t-1).*K;
            G = [gx(:), gy(:), gt(:)];
            len = sum(K(:));
            [u s v] = svd(G,'econ');
            S(1) = (s(1,1) + lambda) / ( sqrt(s(2,2)*s(3,3)) + lambda);
            S(2) = (s(2,2) + lambda) / ( sqrt(s(1,1)*s(3,3)) + lambda);
            S(3) = (s(3,3) + lambda) / ( sqrt(s(2,2)*s(1,1)) + lambda);
            tmp = (S(1) * v(:,1) * v(:,1).' + S(2) * v(:,2) * v(:,2).' + S(3) * v(:,3) * v(:,3).')  * ((s(1,1) * s(2,2)*s(3,3) + 0.0000001) / len)^alpha;
            C11(i,j,k) = tmp(1,1);
            C12(i,j,k) = tmp(1,2);
            C22(i,j,k) = tmp(2,2);
            C13(i,j,k) = tmp(1,3);
            C23(i,j,k) = tmp(2,3);
            C33(i,j,k) = tmp(3,3);
            
        end
    end
end


C11 = EdgeMirror3(C11,[win,win,win_t]);
C12 = EdgeMirror3(C12,[win,win,win_t]);
C22 = EdgeMirror3(C22,[win,win,win_t]);
C23 = EdgeMirror3(C23,[win,win,win_t]);
C33 = EdgeMirror3(C33,[win,win,win_t]);
C13 = EdgeMirror3(C13,[win,win,win_t]);

x13 = 2*x1.*x3;
x12 = 2*x1.*x2;
x23 = 2*x2.*x3;
x11 = x1.^2;
x22 = x2.^2;
x33 = x3.^2;

x1x1 = reshape(repmat(reshape(x11,[1 wsize^2*wsize_t]),M*N*T,1),[M,N,T, wsize wsize wsize_t]);
x2x2 = reshape(repmat(reshape(x22,[1 wsize^2*wsize_t]),M*N*T,1),[M,N,T, wsize wsize wsize_t]);
x3x3 = reshape(repmat(reshape(x33,[1 wsize^2*wsize_t]),M*N*T,1),[M,N,T, wsize wsize wsize_t]);
x1x2 = reshape(repmat(reshape(x12,[1 wsize^2*wsize_t]),M*N*T,1),[M,N,T, wsize wsize wsize_t]);
x1x3 = reshape(repmat(reshape(x13,[1 wsize^2*wsize_t]),M*N*T,1),[M,N,T, wsize wsize wsize_t]);
x2x3 = reshape(repmat(reshape(x23,[1 wsize^2*wsize_t]),M*N*T,1),[M,N,T, wsize wsize wsize_t]);

% Geodesic distance computation between a center and surrounding voxels
LARK = zeros(M,N,T, wsize,wsize,wsize_t);
for i = 1:wsize
    for j = 1:wsize
        for k = 1:wsize_t
        temp = C11(i:i+M-1,j:j+N-1,k:k+T-1).*x1x1(:,:,:,i,j,k)+ C22(i:i+M-1,j:j+N-1,k:k+T-1).*x2x2(:,:,:,i,j,k) + ...
        C33(i:i+M-1,j:j+N-1,k:k+T-1).*x3x3(:,:,:,i,j,k) + C12(i:i+M-1,j:j+N-1,k:k+T-1).*x1x2(:,:,:,i,j,k)+ ...
        C13(i:i+M-1,j:j+N-1,k:k+T-1).*x1x3(:,:,:,i,j,k) + C23(i:i+M-1,j:j+N-1,k:k+T-1).*x2x3(:,:,:,i,j,k);
        LARK(:,:,:,i,j,k) = temp;
        end
    end
end
% Convert geodesic distance to self-similarity

 LARK = exp(-LARK*0.5/h^2);
 LARK = reshape(LARK,[M N T wsize^2*wsize_t]);

 

end