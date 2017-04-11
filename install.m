
if isunix || ismac
    cmd = 'cp -a vllab_dag_loss.m matconvnet/matlab/+dagnn';
else % ispc
    cmd = 'copy vllab_dag_loss.m matconvnet/matlab/+dagnn';
end
fprintf('%s\n', cmd);
system(cmd);


addpath('matconvnet/matlab');
vl_compilenn('enableGPu', true);

