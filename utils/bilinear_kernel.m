function f = bilinear_kernel(k, num_input, num_output)
% -------------------------------------------------------------------------
%   Description:
%       create bilinear interpolation kernel for the convt (deconv) layer
%
%   Input:
%       - k             : kernel size k x k
%       - num_input     : number of input channels
%       - num_output    : number of output channels
%
%   Output:
%       - f             : bilinear filter
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


    radius = ceil(k / 2);
    
    if rem(k, 2) == 1
        center = radius;
    else
        center = radius + 0.5;
    end
    
    C = 1:k;
    f = (ones(1, k) - abs(C - center) ./ radius)' ...
      * (ones(1, k) - abs(C - center) ./ radius);
    
    f = repmat(f, 1, 1, num_input, num_output);


end

