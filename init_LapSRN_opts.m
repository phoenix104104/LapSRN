function opts = init_LapSRN_opts(scale, depth, gpu)
% -------------------------------------------------------------------------
%   Description:
%       Generate model options for LapSRN
%
%   Input:
%       - scale : SR upsampling scale
%       - depth : number of conv layers in one pyramid level
%       - gpu   : GPU ID, 0 for CPU mode
%
%   Output:
%       - opts  : options for LapSRN
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

    %% network options
    opts.scale              = scale;
    opts.depth              = depth;
    opts.weight_decay       = 0.0001;
    opts.init_sigma         = 0.001;
    opts.conv_f             = 3;
    opts.conv_n             = 64;
    opts.loss               = 'L1';

    %% training options
    opts.gpu                = gpu;
    opts.batch_size         = 64;
    opts.num_train_batch    = 100;     % number of training batch in one epoch
    opts.num_valid_batch    = 100;      % number of validation batch in one epoch
    opts.lr                 = 1e-5;     % initial learning rate
    opts.lr_step            = 50;       % number of epochs to drop learning rate
    opts.lr_drop            = 0.5;      % learning rate drop ratio
    opts.lr_min             = 1e-6;     % minimum learning rate
    opts.patch_size         = 128;
    opts.data_augmentation  = 1;
    opts.scale_augmentation = 1;

    %% dataset options
    opts.train_dataset          = {};
    opts.train_dataset{end+1}   = 'T91';
    opts.train_dataset{end+1}   = 'BSDS200';
    %opts.train_dataset{end+1}   = 'General100';
    opts.valid_dataset          = {};
    %opts.valid_dataset{end+1}   = 'Set5';
    %opts.valid_dataset{end+1}   = 'Set14';
    opts.valid_dataset{end+1}   = 'BSDS100';


    %% setup model name
    opts.data_name = 'train';
    for i = 1:length(opts.train_dataset)
        opts.data_name = sprintf('%s_%s', opts.data_name, opts.train_dataset{i});
    end

    opts.net_name = sprintf('LapSRN_x%d_depth%d_%s', ...
                            opts.scale, opts.depth, opts.loss);

    opts.model_name = sprintf('%s_%s_pw%d_lr%s_step%d_drop%s_min%s_bs%d', ...
                            opts.net_name, ...
                            opts.data_name, opts.patch_size, ...
                            num2str(opts.lr), opts.lr_step, ...
                            num2str(opts.lr_drop), num2str(opts.lr_min), ...
                            opts.batch_size);


    %% setup dagnn training parameters
    if( opts.gpu == 0 )
        opts.train.gpus     = [];
    else
        opts.train.gpus     = [opts.gpu];
    end
    opts.train.batchSize    = opts.batch_size;
    opts.train.numEpochs    = 1000;
    opts.train.continue     = true;
    opts.train.learningRate = learning_rate_policy(opts.lr, opts.lr_step, opts.lr_drop, ...
                                                   opts.lr_min, opts.train.numEpochs);

    opts.train.expDir = fullfile('models', opts.model_name) ; % model output dir
    if( ~exist(opts.train.expDir, 'dir') )
        mkdir(opts.train.expDir);
    end

    opts.train.model_name       = opts.model_name;
    opts.train.num_train_batch  = opts.num_train_batch;
    opts.train.num_valid_batch  = opts.num_valid_batch;
    
    % setup loss
    opts.level = ceil(log(opts.scale) / log(2));
    opts.train.derOutputs = {};
    for s = opts.level : -1 : 1
        opts.train.derOutputs{end+1} = sprintf('level%d_%s_loss', s, opts.loss);
        opts.train.derOutputs{end+1} = 1;
    end


end