function cc = filterCandidates(c, varargin)
%

na = length(varargin);
nc = length(c);
if (na == 0)
    cc = c;
else
    for i = 1:2:na
        if (strcmp(varargin{i}, 'scoreLow')) % filter by score
            score = zeros(nc, 1);
            val = varargin{i+1};
            for ic = 1:nc;
                score(ic) = c{ic}.score;
            end
            cc = c(score > val);
        end
    end
end
