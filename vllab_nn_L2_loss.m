function Y = vllab_nn_L2_loss(X, Z, dzdy)
% -------------------------------------------------------------------------
%   Description:
%       L2 (MSE) loss function used in MatConvNet NN
%       forward : Y = vllab_nn_L2_loss(X, Z)
%       backward: Y = vllab_nn_L2_loss(X, Z, dzdy)
%
%   Input:
%       - X     : predicted data
%       - Z     : ground truth data
%       - dzdy  : the derivative of the output
%
%   Output:
%       - Y     : loss when forward, derivative of loss when backward
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

    if nargin <= 2
        diff = (X - Z) .^ 2;
        Y = 0.5 * sum(diff(:));
    else
        Y = (X - Z) * dzdy;
    end
end
