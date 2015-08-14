function dbgmsg( format,varargin )
c = clock;
ts = datestr(datenum(c{:}));
format = sprintf('[%s] %s\n',ts,format);
printf(format,varargin);

