function dbgmsg( format,varargin )
ts = datestr(now,'HH:MM:SS');
format = sprintf('[%s] %s\n',ts,format);
printf(format,varargin{:});

