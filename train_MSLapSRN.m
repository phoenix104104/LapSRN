function train_MSLapSRN(scales, depth, recursive, gpu)
% -------------------------------------------------------------------------
%   Description:
%       Script to train MS-LapSRN from scratch
%
%   Input:
%       - scales    : SR upsampling scales (use vector, e.g., [2, 4, 8])
%       - depth     : numbers of conv layers in recursive block
%       - recursive : numbers of recursive blocks
%       - gpu       : GPU ID, 0 for CPU mode
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


    %% initialize opts
    opts = init_MSLapSRN_opts(scales, depth, recursive, gpu);

    %% save opts
    filename = fullfile(opts.train.expDir, 'opts.mat');
    fprintf('Save parameter %s\n', filename);
    save(filename, 'opts');

    %% setup paths
    addpath(genpath('utils'));
    addpath(fullfile(pwd, 'matconvnet/matlab'));
    vl_setupnn;

    %% initialize network
    fprintf('Initialize network...\n');
    model_filename = fullfile(opts.train.expDir, 'net-epoch-0.mat');

    if( ~exist(model_filename, 'file') )
        model = init_MSLapSRN_model(opts, 'train');
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

    fprintf('Pre-load all images (%d training, %d validation)...\n', ...
        length(find(imdb.images.set == 1)), length(find(imdb.images.set == 2)));
    imdb.images.img = batch_imread(imdb.images.filename);

    %% training
    get_batch = @(x,y,mode) getBatch_MSLapSRN(opts,x,y,mode);

    [net, info] = vllab_cnn_train_dag(model, imdb, get_batch, opts.train, ...
                                    'val', find(imdb.images.set == 2));

