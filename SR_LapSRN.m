function img_HR = SR_LapSRN(img_LR, net, opts)
    
    %% setup
    net.mode = 'test' ;
    output_var = 'level1_output';
    output_index = net.getVarIndex(output_var);
    net.vars(output_index).precious = 1;
    
    % RGB to YUV
    if( size(img_LR, 3) > 1 )
        img_LR = rgb2ycbcr(img_LR);
    end
    
    % extract Y
    y = single(img_LR(:, :, 1));
    
    if( opts.gpu )
        y = gpuArray(y);
    end
    
    % bicubic upsample UV
    img_HR = imresize(img_LR, opts.scale);
    

    % forward
    inputs = {'LR', y};
    net.eval(inputs);
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