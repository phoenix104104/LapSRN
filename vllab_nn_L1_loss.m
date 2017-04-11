function Y = vllab_nn_L1_loss(X, Z, dzdy)
% -------------------------------------------------------------------------
%   Description:
%       L1 (Charbonnier) loss function
%       forward : Y = vllab_nn_L1_loss(X, Z)
%       backward: Y = vllab_nn_L1_loss(X, Z, dzdy)
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

    eps = 1e-6;
    d = X - Z;
    e = sqrt( d.^2 + eps );
    
    if nargin <= 2
        Y = sum(e(:));
    else
        Y = d ./ e;
        Y = Y .* dzdy;
    end
    
end
