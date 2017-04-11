function img = mod_crop(img, scale)
% -------------------------------------------------------------------------
%   Description:
%       crop image boundary to be the multiple of 'scale'
%
%   Input:
%       - img   : input image
%       - scale : upsampling scale
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

if size(img, 3) == 1
    sz = size(img);
    sz = sz - mod(sz, scale);
    img = img(1:sz(1), 1:sz(2));
else
    tmpsz = size(img);
    sz = tmpsz(1:2);
    sz = sz - mod(sz, scale);
    img = img(1:sz(1), 1:sz(2),:);
end

