function net = init_MSLapSRN_model(opts, mode)
% -------------------------------------------------------------------------
%   Description:
%       initialize MS-LapSRN model
%
%   Input:
%       - opts  : options generated from init_MSLapSRN_opts()
%
%   Output:
%       - net   : dagnn model
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
    
    %% parameters
    rng('default');
    rng(0) ;
    
    f   = opts.conv_f;
    n   = opts.conv_n;
    pad = floor(f/2);
    
    if( f == 3 )
        crop = [0, 1, 0, 1];
    elseif( f == 5 )
        crop = [1, 2, 1, 2];
    else
        error('Need to specify crop in deconvolution for f = %d\n', f);
    end
    
    %% initialize model
    net = dagnn.DagNN;
    
    %% multiscale training
    for s = 1:length(opts.scales)

        scale = opts.scales(s);
        level = ceil(log(scale) / log(2));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Feature extraction branch
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %% input conv
        inputs  = { sprintf('x%dSR_LR', scale) };
        outputs = { sprintf('x%dSR_input_conv', scale) };
        params  = { 'input_conv_f', 'input_conv_b' };
        
        net.addLayer(outputs{1}, ...
                     dagnn.Conv('size', [f, f, 1, n], ...
                                'pad', pad, ...
                                'stride', 1), ...
                     inputs, outputs, params);
        
        level_input = outputs{1};
    
        %% feature embedding sub-network
        for l = 1 : level
            
            block_input = {};
            block_output = {};
            
            for r = 1:opts.recursive

                if r == 1
                    block_input{r} = level_input;
                else
                    block_input{r} = block_output{r-1};
                end

                %% recursive block
                for d = 1:opts.depth

                    if d == 1
                        next_input = block_input{r};
                    end

                    % ReLU
                    inputs  = { next_input };
                    outputs = { sprintf('x%dSR_level%d_R%d_relu%d', scale, l, r, d) };

                    net.addLayer(outputs{1}, ...
                                 dagnn.ReLU('leak', 0.2), ...
                             inputs, outputs);

                    next_input = outputs{1};
                    

                    % conv
                    inputs  = { next_input };
                    outputs = { sprintf('x%dSR_level%d_R%d_conv%d', scale, l, r, d) };
                    params  = { sprintf('conv%d_f', d), ...
                                sprintf('conv%d_b', d)};

                    net.addLayer(outputs{1}, ...
                                 dagnn.Conv('size', [f, f, n, n], ...
                                            'pad', pad, ...
                                            'stride', 1), ...
                                 inputs, outputs, params);

                    next_input = outputs{1};

                end %% end of recursive block

                %% local skip connection
                if strcmp(opts.LRL, 'NS')

                    % no skip connection
                    block_output{r} = next_input;

                elseif strcmp(opts.LRL, 'DS')

                    % next_input + block_input
                    inputs  = { next_input, block_input{r} };
                    outputs = { sprintf('x%dSR_level%d_R%d_output', scale, l, r) };
                    net.addLayer(outputs{1}, ...
                        dagnn.Sum(), ...
                        inputs, outputs);
                    
                    block_output{r} = outputs{1};

                elseif strcmp(opts.LRL, 'SS')

                    % next_input + level_input
                    inputs  = { next_input, level_input };
                    outputs = { sprintf('x%dSR_level%d_R%d_output', scale, l, r) };
                    net.addLayer(outputs{1}, ...
                        dagnn.Sum(), ...
                        inputs, outputs);
                    
                    block_output{r} = outputs{1};
                
                else

                    error('Unknown local skip connection %s.', opts.LRL);

                end %% end of local skip connection


            end %% end of recursive


            %% features upsample layers
            % ReLU
            inputs  = { block_output{opts.recursive} };
            outputs = { sprintf('x%dSR_level%d_uprelu', scale, l) };
     
            
            net.addLayer(outputs{1}, ...
                         dagnn.ReLU('leak', 0.2), ...
                         inputs, outputs);

            next_input = outputs{1};
                     

            % conv
            inputs  = { next_input };
            outputs = { sprintf('x%dSR_level%d_upconv', scale, l) };
            params  = { 'upconv_f', 'upconv_b' };
                        
            net.addLayer(outputs{1}, ...
                         dagnn.ConvTranspose(...
                             'size', [f, f, n, n], ...
                             'upsample', 2, ...
                             'crop', crop, ...
                             'numGroups', 1, ...
                             'hasBias', true), ...
                         inputs, outputs, params) ;
            
            
            level_input = outputs{1};
            
                    
        end %% end of level (feature extraction)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Image reconstruction branch
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
        for l = 1 : level

            if l == 1
                next_input = sprintf('x%dSR_LR', scale);
            else
                next_input = sprintf('x%dSR_%dx_output', scale, 2^(l-1));
            end
            
            %% image upsample layer
            inputs  = { next_input };
            outputs = { sprintf('x%dSR_level%d_img_up', scale, l) };
            params  = { 'img_up_f' };

            net.addLayer(outputs{1}, ...
                dagnn.ConvTranspose(...
                    'size', [4, 4, 1, 1], ...
                    'upsample', 2, ...
                    'crop', 1, ...
                    'numGroups', 1, ...
                    'hasBias', false), ...
                inputs, outputs, params) ;

            
            %% residual prediction layer (f x f x n x 1)
            inputs  = { sprintf('x%dSR_level%d_upconv', scale, l) };
            outputs = { sprintf('x%dSR_level%d_residual', scale, l) };
            params  = { 'residual_conv_f', 'residual_conv_b' };
            
            net.addLayer(outputs{1}, ...
                dagnn.Conv('size', [f, f, n, 1], ...
                           'pad', pad, ...
                           'stride', 1), ...
                inputs, outputs, params);

            %% addition layer
            inputs  = { sprintf('x%dSR_level%d_img_up', scale, l), ...
                        sprintf('x%dSR_level%d_residual', scale, l) };
            outputs = { sprintf('x%dSR_%dx_output', scale, 2^l) };
            net.addLayer(outputs{1}, ...
                dagnn.Sum(), ...
                inputs, outputs);
            
            next_input = outputs{1};
            
            %% Loss layer
            inputs  = { next_input, ...
                        sprintf('x%dSR_%dx_HR', scale, 2^l) };
            outputs = { sprintf('x%dSR_%dx_%s_loss', scale, 2^l, opts.loss) };
            
            net.addLayer(outputs{1}, ...
                     dagnn.vllab_dag_loss(...
                        'loss_type', opts.loss), ...
                     inputs, outputs);
                    
        end %% end of level (image reconstruction)
                 

    end %% end of multiscale

    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% initialize parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if strcmp(mode, 'train')

        %% input conv
        params  = { 'input_conv_f', 'input_conv_b' };

        sigma   = opts.init_sigma;
        filters = sigma * randn(f, f, 1, n, 'single');
        biases  = zeros(1, n, 'single');

        idx = net.getParamIndex(params{1});
        net.params(idx).value         = filters;
        net.params(idx).learningRate  = 1;
        net.params(idx).weightDecay   = 1;

        idx = net.getParamIndex(params{2});
        net.params(idx).value         = biases;
        net.params(idx).learningRate  = 0.1;
        net.params(idx).weightDecay   = 1;

        
        %% deep conv
        for d = 1:opts.depth

            params  = { sprintf('conv%d_f', d), ...
                        sprintf('conv%d_b', d)};

            sigma   = sqrt( 2 / (f * f * n) );
            filters = sigma * randn(f, f, n, n, 'single');
            biases  = zeros(1, n, 'single');

            idx = net.getParamIndex(params{1});
            net.params(idx).value         = filters;
            net.params(idx).learningRate  = 1;
            net.params(idx).weightDecay   = 1;

            idx = net.getParamIndex(params{2});
            net.params(idx).value         = biases;
            net.params(idx).learningRate  = 0.1;
            net.params(idx).weightDecay   = 1;

        end

        %% feature upsample
        params  = { 'upconv_f', 'upconv_b' };

        sigma   = sqrt( 2 / (f * f * n) );
        filters = sigma * randn(f, f, n, n, 'single');
        biases  = zeros(1, n, 'single');

        idx = net.getParamIndex(params{1});
        net.params(idx).value         = filters;
        net.params(idx).learningRate  = 1;
        net.params(idx).weightDecay   = 1;

        idx = net.getParamIndex(params{2});
        net.params(idx).value         = biases;
        net.params(idx).learningRate  = 0.1;
        net.params(idx).weightDecay   = 1;

        %% image upsample
        params  = { 'img_up_f' };

        filters = single(bilinear_kernel(4, 1, 1));

        idx = net.getParamIndex(params{1});
        net.params(idx).value         = filters;
        net.params(idx).learningRate  = 1;
        net.params(idx).weightDecay   = 1;

        %% residual prediction
        params  = { 'residual_conv_f', 'residual_conv_b' };

        sigma   = sqrt(2 / (f * f * n));
        filters = sigma * randn(f, f, n, 1, 'single');
        biases  = zeros(1, 1, 'single');

        idx = net.getParamIndex(params{1});
        net.params(idx).value         = filters;
        net.params(idx).learningRate  = 1;
        net.params(idx).weightDecay   = 1;

        idx = net.getParamIndex(params{2});
        net.params(idx).value         = biases;
        net.params(idx).learningRate  = 0.1;
        net.params(idx).weightDecay   = 1;

    end

end

