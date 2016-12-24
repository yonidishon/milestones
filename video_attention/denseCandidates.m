function cands = denseCandidates(srcFr, step)

x = min(floor(step/2), 1):step:srcFr.width;
y = min(floor(step/2), 1):step:srcFr.height;
nx = length(x);
ny = length(y);

cands = cell(nx * ny, 1);
ax = srcFr.height/8;

for xx = 1:nx
    for yy = 1:ny
        ic = yy + ny * (xx-1);
        cands{ic}.point = [x(xx), y(yy)];
        cands{ic}.type = 7;
        cands{ic}.score = 1;
        cands{ic}.cov = [ax^2, 0; 0, ax^2];
        cands{ic}.candCov = [ax^2, 0; 0, ax^2];
    end
end
