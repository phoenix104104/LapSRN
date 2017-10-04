function img_list = batch_imread(batch)
% -------------------------------------------------------------------------
%   Description:
%       read a batch of images
%
%   Input:
%       - batch : array of ID to fetch
%
%   Output:
%       - img_list: batch of images
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

    img_list = cell(length(batch), 1);
    
    for i = 1:length(batch)
         img = imread(batch{i});
         
         if( size(img, 3) > 1 )
             img = rgb2ycbcr(img);
             img = img(:, :, 1);
         end
         
         img_list{i} = im2single(img);
    end

end