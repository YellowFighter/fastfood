function dbgmsg( varargin )
global DEBUG
if DEBUG
    fprintf(varargin);
end