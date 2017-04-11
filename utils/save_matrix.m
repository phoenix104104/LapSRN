function save_matrix(Q, filename, precision)
% -------------------------------------------------------------------------
%   Description:
%       save a 2D array into text file
%
%   Input:
%       - Q         : input matrix (2D array)
%       - filename  : output file name
%       - precision : float point precision [default = 7]
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

    if( ~exist('precision', 'var') )
        precision = 7;
    end
    
    file = fopen(filename, 'w');
    
    if( size(Q, 2) == 1 )
        dlmwrite(filename, Q, 'precision', precision, 'delimiter', '\n', 'newline', 'unix');
    else
        for i = 1:size(Q, 1)
            fprintf(file, sprintf('%%.%df\t', precision), Q(i, 1:end-1));
            fprintf(file, sprintf('%%.%df', precision), Q(i, end));
            fprintf(file, '\n');
        end
    end
    fclose(file);
    
end