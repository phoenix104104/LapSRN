function [PSNR, SSIM, IFC] = evaluate_SR(img_GT, img_HR, scale, compute_ifc)
% -------------------------------------------------------------------------
%   Description:
%       Compute PSNR, SSIM and IFC for SR
%       We convert RGB image to grayscale and crop boundaries for 'scale'
%       pixels
%
%   Input:
%       - img_GT        : Ground truth image
%       - img_HR        : predicted HR image
%       - scale         : upsampling scale
%       - compute_ifc   : evaluate IFC [default = 0 since it's slow]
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

    if ~exist('compute_ifc', 'var')
        compute_ifc = 0;
    end
    
    %% quantize pixel values
    img_GT = im2double(im2uint8(img_GT)); 
    img_HR = im2double(im2uint8(img_HR)); 
        
    %% convert to gray scale
    if( size(img_GT, 3) > 1 )
        img_GT = rgb2ycbcr(img_GT); img_GT = img_GT(:, :, 1);
        img_HR = rgb2ycbcr(img_HR); img_HR = img_HR(:, :, 1);
    end
    
    %% crop boundary
    img_GT = shave_bd(img_GT, scale);
    img_HR = shave_bd(img_HR, scale);
    
    % evaluate
    PSNR = psnr(img_GT, img_HR);
    SSIM = ssim(img_GT, img_HR);
    
    % comment IFC to speed up testing
    IFC = 0;
    if compute_ifc
        IFC = ifcvec(img_GT, img_HR);
        if( ~isreal(IFC) )
            IFC = 0;
        end
    end

end