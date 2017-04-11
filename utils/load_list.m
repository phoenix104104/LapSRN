function list = load_list(list_name)
% -------------------------------------------------------------------------
%   Description:
%       load a list file that each row is a string/name
%
%   Input:
%       - list_name: file name of the list
%
%   Output:
%       - list  : a cell array
%
%   Citation: 
%       Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution
%       Wei-Sheng Lai, Jia-Bin Huang, Narendra Ahuja, and Ming-Hsuan Yang
%       IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017
%
%   Contact:
%       Wei-Sheng Lai
%       wlai24@ucmerced.edu
%       University of California, Merced
% -------------------------------------------------------------------------

    f = fopen(list_name);
    if( f == -1 )
        error('%s does not exist!', list_name);
    end
    C = textscan(f, '%s', 'CommentStyle', '#');
    list = C{1};
    fclose(f);
end