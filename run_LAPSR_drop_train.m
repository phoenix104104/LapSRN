function run_LAPSR_drop_train(scale, pw, depth, loss, lr, lr_step, lr_drop, gc, gpu)
    
    if( nargin < 4 )
        fprintf('Usage: run_holistic_residual_SR_L1_fixbr_train(scale, depth, lr, gpu)\n');
        return;
    end
    
    addpath('../utils');
    addpath(fullfile(pwd, '../../../deep_interpolation/code/matconvnet/matlab'));
    vl_setupnn;
    clear opts;

    if( gpu > 0 )
        opts.gpus = [gpu];
    else
        opts.gpus = [];
    end
    opts.train_dataset      = '291';
    opts.valid_dataset      = 'BSDS100';
    opts.scale              = scale;
    opts.patch_size         = pw;
    opts.depth              = depth;
    opts.loss               = loss;
    opts.num_patch_per_img  = 1;
    opts.num_img_per_batch  = 64;
    opts.batch_size         = opts.num_img_per_batch;
    opts.num_batch_in_epoch = 1000;  % 1 * 64 * 1000 patches / epoch
    opts.lr                 = lr;
    opts.lr_step            = lr_step;
    opts.lr_drop            = lr_drop;
    opts.gradient_clip      = gc;
    
    opts.data_augmentation  = 1;
    
    
    opts.data_name = sprintf('train_%s_valid_%s', opts.train_dataset, opts.valid_dataset);
    
	opts.net_name = sprintf('LAPSR_x%d_pw%d_depth%d_%s', ...
                    opts.scale, opts.patch_size, opts.depth, opts.loss);
                    
    
	opts.model_name = sprintf('%s_%s_lr%s_step%d_drop%s_gc%s_aug%d', ...
                        opts.net_name, ...
                        opts.data_name, ...
                        num2str(lr), opts.lr_step, num2str(opts.lr_drop), ...
                        num2str(opts.gradient_clip), ...
                        opts.data_augmentation);
                    

    %% setup network parameters
    opts.net.weight_decay   = 0.0001;
    opts.net.init_sigma     = 0.001;
    opts.net.conv_f         = 3;
    opts.net.conv_n         = 64;


    %% setup training parameters
    opts.train.gpus     	= opts.gpus;
    opts.train.batchSize    = opts.batch_size;
    opts.train.numEpochs    = 1000;
    opts.train.continue     = true;
    opts.train.learningRate = max(generate_lr(lr, opts.lr_step, opts.lr_drop, 1000), ...
                                  lr * 0.1);
    opts.train.gradient_clip = gc;
    opts.train.num_batch_in_epoch = opts.num_batch_in_epoch;
    
    opts.model_dir = fullfile('model', opts.train_dataset);
    opts.train.expDir = fullfile(opts.model_dir, opts.model_name) ; % model output dir

    opts.train.model_name = opts.model_name;
    
    opts.level = ceil(log(opts.scale) / log(2));
    opts.train.derOutputs = {};
    for s = opts.level : -1 : 1
        opts.train.derOutputs{end+1} = sprintf('level%d_%s_loss', s, opts.loss);
        opts.train.derOutputs{end+1} = 1;
    end
    
    
    if( ~exist(opts.train.expDir, 'dir') )
        mkdir(opts.train.expDir);
    end

    % save opts
    filename = fullfile(opts.train.expDir, 'opts.mat');
    fprintf('Save parameter %s\n', filename);
    save(filename, 'opts');

    %% initialize network
    fprintf('Initialize network...\n');
    model_filename = fullfile(opts.train.expDir, 'net-epoch-0.mat');

%     if( ~exist(model_filename, 'file') )
        model = init_LAPSR(opts);
        fprintf('Save %s\n', model_filename);
        net = model.saveobj();
        save(model_filename, 'net');
%     else
%         fprintf('Load %s\n', model_filename);
%         model = load(model_filename);
%         model = dagnn.DagNN.loadobj(model.net);
%     end

    
    %% load imdb
    filename = fullfile('imdb', sprintf('imdb_%s.mat', opts.data_name));
    fprintf('Load data %s\n', filename);
    imdb = load(filename);

    %% training

    get_batch = @(x,y,mode) getBatch_multiscale(opts,x,y,mode) ;
    
    [net, info] = train_dagnn(model, imdb, get_batch, opts.train, ...
                                'val', find(imdb.images.set == 3));


