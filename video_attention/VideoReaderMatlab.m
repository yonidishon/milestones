classdef (CaseInsensitiveProperties=true, TruncatedProperties=true) ...
        VideoReaderMatlab < hgsetget

    %------------------------------------------------------------------
    % General properties (in alphabetic order)
    %------------------------------------------------------------------
    properties(GetAccess='public')
        Duration        % Total length of file in seconds.
        Name            % Name of the file to be read.
        Path            % Path of the file to be read.
    end
    
    properties(GetAccess='public', SetAccess='public')
        Tag = '';       % Generic string for the user to set.
    end
    
    properties(GetAccess='public', Dependent) 
        Type            % Classname of the object.
    end
    
    properties(GetAccess='public', SetAccess='public')
        UserData        % Generic field for any user-defined data.
    end
    
    properties(GetAccess='private', SetAccess='private')
        frames          % the frames of the video
    end
    %------------------------------------------------------------------
    % Video properties (in alphabetic order)
    %------------------------------------------------------------------
    properties(GetAccess='public')
        BitsPerPixel    % Bits per pixel of the video data.
        FrameRate       % Frame rate of the video in frames per second.
        Height          % Height of the video frame in pixels.
        NumberOfFrames  % Total number of frames in the video stream. 
        Width           % Width of the video frame in pixels.
    end

    %------------------------------------------------------------------
    % Documented methods
    %------------------------------------------------------------------    
    methods(Access='public')
        %------------------------------------------------------------------
        % Lifetime
        %------------------------------------------------------------------
        function obj = VideoReaderMatlab(fileName)
            % If no file name provided.
            if nargin == 0
                error(message('MATLAB:audiovideo:VideoReaderMatlab:noFile'));
            end

            % Initialize the object.
            s = load(fileName);
            obj.frames = s.frames;
            clear s;
            [obj.Path, obj.Name] = fileparts(fileName);
            obj.Name = [obj.Name, '.avi'];
            obj.FrameRate = 30;
        end
        
        function v = read(obj, varargin)
            if (isempty(varargin))
                frs = 1:obj.NumberOfFrames;
            else
                if (isscalar(varargin{1}))
                    frs = varargin{1};
                else
                    frs = varargin{1}(1):varargin{1}(2);
                end
            end
            
            v = obj.frames(:,:,:,frs);
        end
    end
    
    methods(Static, Hidden)
        %------------------------------------------------------------------
        % Persistence
        %------------------------------------------------------------------        
        obj = loadobj(B)
    end

    %------------------------------------------------------------------
    % Custom Getters/Setters
    %------------------------------------------------------------------
    methods
        % Properties that are not dependent on underlying object.
        function set.Tag(obj, value)
            if ~(ischar(value) || isempty(value))
                error(message('MATLAB:audiovideo:VideoReader:TagMustBeString'));
            end
            obj.Tag = value;
        end
        
        function value = get.Type(obj)
            value = class(obj);
        end
        
        % Properties that are dependent on underlying object.
        function value = get.Duration(obj)
            value = size(obj.frames, 4) / obj.FrameRate;
        end
        
        function value = get.BitsPerPixel(obj)
            value = 8 * size(obj.frames, 3);
        end
        
        function value = get.Height(obj)
            value = size(obj.frames, 1);
        end
        
        function value = get.NumberOfFrames(obj)
            value = size(obj.frames, 4);
        end
        
        function value = get.Width(obj)
            value = size(obj.frames, 2);
        end
    end
end
