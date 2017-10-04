function make_imdb(imdb_filename, opts)
% -------------------------------------------------------------------------
%   Description:
%       generate imdb file for training LapSRN
%
%   Input:
%       - imdb_filename : imdb file name
%       - opts : options generated from init_opts()
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

    addpath('utils');

    %% settings
    img_ext = 'png';

    list_dir    = fullfile('lists');
    output_dir  = fullfile('imdb');
    if( ~exist(output_dir, 'dir') )
        mkdir(output_dir);
    end

    %% training data
    train_list = {};

    for d = 1:length(opts.train_dataset)

        image_dir   = fullfile('datasets', opts.train_dataset{d});
        list_filename = fullfile(list_dir, sprintf('%s.txt', opts.train_dataset{d}));
        image_list = load_list(list_filename);
        num_image = length(image_list);

        for i = 1:num_image

            filename = fullfile(image_dir, sprintf('%s.%s', image_list{i}, img_ext));
            fprintf('%s %d / %d: %s\n', opts.train_dataset{d}, i, num_image, filename);
            if( ~exist(filename, 'file') )
                error('%s does not exist!\n', filename);
            end
            train_list{end + 1} = filename;
        end

    end

    num_train = length(train_list);

    %% validation data
    valid_list = {};

    for d = 1:length(opts.valid_dataset)

        image_dir   = fullfile('datasets', opts.valid_dataset{d});
        list_filename = fullfile(list_dir, sprintf('%s.txt', opts.valid_dataset{d}));
        image_list = load_list(list_filename);
        num_image = length(image_list);

        for i = 1:num_image

            filename = fullfile(image_dir, sprintf('%s.%s', image_list{i}, img_ext));
            fprintf('%s %d / %d: %s\n', opts.valid_dataset{d}, i, num_image, image_list{i});
            if( ~exist(filename, 'file') )
                error('%s does not exist!\n', filename);
            end
            valid_list{end + 1} = filename;
        end

    end

    num_valid = length(valid_list);

    fprintf('Collect %d training images\n', num_train);
    fprintf('Collect %d validation images\n', num_valid);

    %% build imdb
    clear images;
    images.filename = [train_list'; valid_list'];

    images.set = 2 * ones(1, num_train + num_valid);    % set = 2 for validation data
    images.set(1:num_train) = 1;                        % set = 1 for training data

    %% save imdb
    fprintf('Save %s\n', imdb_filename);
    save(imdb_filename, 'images', '-v7.3');
    
end