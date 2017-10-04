function [img_HR, time] = SR_patch_MSLapSRN(img_LR, net, model_scale, test_scale, gpu)
% -------------------------------------------------------------------------
%   Description:
%       function to apply patch-based SR with MS-LapSRN
%       We split input image into 4 overlapped sub-regions and apply SR
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
%
%   Contact:
%       Wei-Sheng Lai
%       wlai24@ucmerced.edu
%       University of California, Merced
% -------------------------------------------------------------------------

    %% setup
    net.mode = 'test';
    output_var = sprintf('x%dSR_%dx_output', model_scale, model_scale);
    output_index = net.getVarIndex(output_var);
    net.vars(output_index).precious = 1;
    
    % RGB to YUV
    if( size(img_LR, 3) > 1 )
        img_LR = rgb2ycbcr(img_LR);
    end
    
    % extract Y
    y_LR = single(img_LR(:, :, 1));
    if( gpu )
        y_LR = gpuArray(y_LR);
    end

    H = size(y_LR, 1);
    W = size(y_LR, 2);

    pw = ceil(W / 2);
    ph = ceil(H / 2);

    rf = 40;

    patch_ul = y_LR(      1 : ph + rf,       1 : pw + rf);
    patch_ur = y_LR(      1 : ph + rf, pw - rf + 1 : end);
    patch_dl = y_LR(ph - rf + 1 : end,       1 : pw + rf);
    patch_dr = y_LR(ph - rf + 1 : end, pw - rf + 1 : end);

    tic;
    % forward ul
    inputs = {sprintf('x%dSR_LR', model_scale), patch_ul};
    net.eval(inputs);
    patch_ul = gather(net.vars(output_index).value);
    patch_ul = patch_ul(1 : ph * model_scale, 1 : pw * model_scale);
            
    % forward ur
    inputs = {sprintf('x%dSR_LR', model_scale), patch_ur};
    net.eval(inputs);
    patch_ur = gather(net.vars(output_index).value);
    patch_ur = patch_ur(1 : ph * model_scale, rf * model_scale + 1 : end);

    % forward dl
    inputs = {sprintf('x%dSR_LR', model_scale), patch_dl};
    net.eval(inputs);
    patch_dl = gather(net.vars(output_index).value);
    patch_dl = patch_dl(rf * model_scale + 1 : end, 1 : pw * model_scale);

    % forward dr
    inputs = {sprintf('x%dSR_LR', model_scale), patch_dr};
    net.eval(inputs);
    patch_dr = gather(net.vars(output_index).value);
    patch_dr = patch_dr(rf * model_scale + 1 : end, rf * model_scale + 1 : end);


    % reconstruct output
    y_HR = cat(1, cat(2, patch_ul, patch_ur), cat(2, patch_dl, patch_dr));
    
    time = toc;

    % bicubic upsample UV
    img_HR = imresize(img_LR, test_scale);

    if( size(y_HR, 1) ~= size(img_HR, 1) )
        y_HR = imresize(y_HR, [size(img_HR, 1), size(img_HR, 2)]);
    end
    
    img_HR(:, :, 1) = double(y_HR);
        
    % YUV to RGB
    if( size(img_HR, 3) > 1 )
        img_HR = ycbcr2rgb(img_HR);
    end
        
    
end
