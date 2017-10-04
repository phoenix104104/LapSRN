function N = count_network_parameters(net)
% -------------------------------------------------------------------------
%   Description:
%       Count the total number of network parameters
%
%   Input:
%       - net   : network
%
%   Output:
%       - N     : #parameters
%
%   Citation: 
%       Fast and Accurate Image Super-Resolution with Deep Laplacian Pyramid Networks
%       Wei-Sheng Lai, Jia-Bin Huang, Narendra Ahuja, and Ming-Hsuan Yang
%       arXiv, 2017
%
%   Contact:
%       Wei-Sheng Lai
%       wlai24@ucmerced.edu
%       University of California, Merced
% -------------------------------------------------------------------------

    N = 0;
    for i = 1:length(net.params)

        N = N + numel(net.params(i).value);

    end


end