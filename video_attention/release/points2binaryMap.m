function mp = points2binaryMap(pts, sz)

m = sz(2);
n = sz(1);
mp = false(m, n);
pts = round(pts);
vi = find(~isnan(pts(:,1)) & pts(:,1) >= 1 & pts(:,1) <= n & pts(:,2) >= 1 & pts(:,2) <= m);
ind = sub2ind([m, n], pts(vi, 2), pts(vi, 1));
mp(ind) = true;
