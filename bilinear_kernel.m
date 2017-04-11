function f = bilinear_kernel(k, numGroups, numClasses)
% -------------------------------------------------------------------------
%   Description:
%       create bilinear interpolation kernel for the convt (deconv) layer
%
%   Input:
%       - k             : kernel size k x k
%       - numGroups     : number of filter groups convt layer
%       - numClasses    : number of input channels
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


    factor = floor((k + 1) / 2);
    
    if rem(k, 2) == 1
        center = factor;
    else
      center = factor + 0.5;
    end
    
    C = 1:k;
    f = zeros(k, k, numGroups, numClasses);

    for i = 1:numGroups
        for j = 1:numClasses
            f(:, :, i, j) = ...
                (ones(1, k) - abs(C - center) ./ factor)' ...
              * (ones(1, k) - abs(C - center) ./ factor);
        end
    end


end

