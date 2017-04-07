% clear all;

%% main parameters
scale = 4;
depth = 5;


%% network options
opts.scale          = scale;
opts.depth          = depth;
opts.weight_decay   = 0.0001;
opts.init_sigma     = 0.001;
opts.conv_f         = 3;
opts.conv_n         = 64;
opts.loss           = 'L1';

%% training options
opts.gpus               = [1];
opts.batch_size         = 64;
opts.num_train_batch    = 10;
opts.num_valid_batch    = 10;
opts.lr                 = 1e-5;
opts.lr_step            = 50;
opts.lr_drop            = 0.5;
opts.lr_min             = 1e-6;
opts.patch_size         = 48;
opts.data_augmentation  = 1;

%% dataset options
opts.train_dataset      = {};
opts.train_dataset{end+1} = 'T91';
opts.train_dataset{end+1} = 'BSDS200';
%opts.train_dataset{end+1} = 'General100';
opts.valid_dataset      = {};
opts.valid_dataset{end+1} = 'Set5';
opts.valid_dataset{end+1} = 'Set14';
opts.valid_dataset{end+1} = 'BSDS100';

opts.data_name = 'train';
for i = 1:length(opts.train_dataset)
    opts.data_name = sprintf('%s_%s', opts.data_name, opts.train_dataset{i});
end
    
opts.net_name = sprintf('LapSRN_x%d_depth%d_%s', ...
                        opts.scale, opts.depth, opts.loss);
                    
    
opts.model_name = sprintf('%s_%s_pw%d_lr%s_step%d_drop%s_min%s', ...
                        opts.net_name, ...
                        opts.data_name, opts.patch_size, ...
                        num2str(opts.lr), opts.lr_step, ...
                        num2str(opts.lr_drop), num2str(opts.lr_min));
                    

%% setup training parameters
opts.train.gpus     	= opts.gpus;
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

opts.level = ceil(log(opts.scale) / log(2));
opts.train.derOutputs = {};
for s = opts.level : -1 : 1
    opts.train.derOutputs{end+1} = sprintf('level%d_%s_loss', s, opts.loss);
    opts.train.derOutputs{end+1} = 1;
end
    

%% save opts
filename = fullfile(opts.train.expDir, 'opts.mat');
fprintf('Save parameter %s\n', filename);
save(filename, 'opts');

%% setup paths
addpath(genpath('utils'));
addpath(fullfile(pwd, 'matconvnet-1.0-beta24/matlab'));
vl_setupnn;

%% initialize network
fprintf('Initialize network...\n');
model_filename = fullfile(opts.train.expDir, 'net-epoch-0.mat');

if( ~exist(model_filename, 'file') )
    model = init_LapSRN(opts);
    fprintf('Save %s\n', model_filename);
    net = model.saveobj();
    save(model_filename, 'net');
else
    fprintf('Load %s\n', model_filename);
    model = load(model_filename);
    model = dagnn.DagNN.loadobj(model.net);
end

    
%% load imdb
imdb_filename = fullfile('imdb', sprintf('imdb_%s.mat', opts.data_name));
if( ~exist(imdb_filename, 'file') )
    make_imdb(imdb_filename, opts);
end
fprintf('Load data %s\n', imdb_filename);
imdb = load(imdb_filename);

fprintf('Pre-load all images...\n');
imdb.images.img = batch_imread(imdb.images.filename);
    
%% training
get_batch = @(x,y,mode) getBatch_multiscale(opts,x,y,mode);

[net, info] = vllab_cnn_train_dag(model, imdb, get_batch, opts.train, ...
                                  'val', find(imdb.images.set == 2));

