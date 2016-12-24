function imout = embedLine(im, pt1, pt2, col, cmap)

pt1 = round(pt1);
pt2 = round(pt2);
imout = im;
[m, n, ~] = size(im);
nl = size(pt1, 1);
if (isscalar(col))
    col = col .* ones(nl, 1);
end

for il = 1:nl
    c = reshape(cmap(col(il), :), [1 1 3]);
    
    if (pt2(il,1) == pt1(il,1)) % vertical line
        ys = min(pt1(il,2), pt2(il,2));
        ye = max(pt1(il,2), pt2(il,2));
        imout(ys:ye, pt1(il,1), :) = repmat(c, [ye-ys+1, 1, 1]);
    elseif (pt2(il,2) == pt1(il,2)) % horisontal line
        xs = min(pt1(il,1), pt2(il,1));
        xe = max(pt1(il,1), pt2(il,1));
        imout(pt1(il,2), xs:xe, :) = repmat(c, [1, xe-xs+1, 1]);
    else % other lines
        ts = 1 / max(abs(pt2(il,1) - pt1(il,1)), abs(pt2(il,2) - pt1(il,2)));
        t = 0:ts:1;
        x = round(pt1(il,1) + t .* ((pt2(il,1) - pt1(il,1))));
        y = round(pt1(il,2) + t .* ((pt2(il,2) - pt1(il,2))));
        ind = sub2ind([m, n], y, x);
        for i = 1:3
            imc = imout(:,:,i);
            imc(ind) = c(i);
            imout(:,:,i) = imc;
        end
    end
end
