function dbgmsg( format,varargin )
ts = datestr(now,'HH:MM:SS');
printf('[%s] ',ts);
printf(format,varargin{:});

