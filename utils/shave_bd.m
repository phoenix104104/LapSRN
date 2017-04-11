function img = shave_bd(img, bd)
% -------------------------------------------------------------------------
%   Description:
%       crop image boundaries for 'bd' pixels
%
%   Input:
%       - img   : input image
%       - bd    : pixels to be cropped
%
%   Output:
%       - img   : output image
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

    img = img(1+bd:end-bd, 1+bd:end-bd, :);
    
end