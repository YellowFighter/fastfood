function dbgmsg( format,varargin )
ts = datestr(now,'HH:MM:SS');
fprintf('[%s] ',ts);
fprintf(format,varargin{:});

