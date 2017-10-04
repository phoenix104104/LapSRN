function [img_HR, time] = SR_MSLapSRN(img_LR, net, model_scale, test_scale, gpu)
% -------------------------------------------------------------------------
%   Description:
%       function to apply SR with MS-LapSRN
%
%   Input:
%       - img_LR        : low-resolution image
%       - net           : MS-LapSRN model
%       - model_scale   : model upsampling scale for constructing pyramid
%       - test_scale    : image upsampling scale
%       - gpu           : GPU ID
%
%   Output:
%       - img_HR: high-resolution image
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

    %% setup
    net.mode = 'test' ;
    output_var = sprintf('x%dSR_%dx_output', model_scale, model_scale);
    output_index = net.getVarIndex(output_var);
    net.vars(output_index).precious = 1;
    
    % RGB to YUV
    if( size(img_LR, 3) > 1 )
        img_LR = rgb2ycbcr(img_LR);
    end
    
    % extract Y
    y = single(img_LR(:, :, 1));
    
    if( gpu )
        y = gpuArray(y);
    end
    
    % bicubic upsample UV
    img_HR = imresize(img_LR, test_scale);
    

    % forward
    inputs = {sprintf('x%dSR_LR', model_scale), y};
    tic;
    net.eval(inputs);
    time = toc;
    y = gather(net.vars(output_index).value);

        
    % resize if size does not match the output image
    if( size(y, 1) ~= size(img_HR, 1) )
        y = imresize(y, [size(img_HR, 1), size(img_HR, 2)]);
    end
    
    img_HR(:, :, 1) = double(y);
        
    % YUV to RGB
    if( size(img_HR, 3) > 1 )
        img_HR = ycbcr2rgb(img_HR);
    end
        
    
end
