function train_LapSRN(scale, depth, gpu)
% -------------------------------------------------------------------------
%   Description:
%       Script to train LapSRN from scratch
%
%   Input:
%       - scale : SR upsampling scale
%       - depth : numbers of conv layers in each pyramid level
%       - gpu   : GPU ID, 0 for CPU mode
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


    %% initialize opts
    opts = init_opts(scale, depth, gpu);

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
    get_batch = @(x,y,mode) getBatch_LapSRN(opts,x,y,mode);

    [net, info] = vllab_cnn_train_dag(model, imdb, get_batch, opts.train, ...
                                      'val', find(imdb.images.set == 2));

