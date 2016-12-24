function f = fullfileCreate(varargin)

narginchk(1, Inf);

fs = filesep; 
f = varargin{1};
bIsPC = ispc;

for i=2:nargin,
   part = varargin{i};
   if isempty(f) || isempty(part)
      f = [f part]; %#ok<AGROW>
   else
      % Handle the three possible cases
      if (f(end)==fs) && (part(1)==fs),
         f = [f part(2:end)]; %#ok<AGROW>
      elseif (f(end)==fs) || (part(1)==fs || (bIsPC && (f(end)=='/' || part(1)=='/')) )
         f = [f part]; %#ok<AGROW>
      else
         f = [f fs part]; %#ok<AGROW>
      end
   end
end

% Be robust to / or \ on PC
if bIsPC
   f = strrep(f,'/','\');
end

if (~exist(f, 'dir'))
    mkdir(f);
end
