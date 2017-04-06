addpath('utils');

%% settings
train_dataset{1} = 'T91';
train_dataset{2} = 'BSDS200';
%train_dataset{3} = 'General100';

valid_dataset{1} = 'Set5';
valid_dataset{2} = 'Set14';
valid_dataset{3} = 'BSDS100';


img_ext = 'png';

data_name = 'train';
for i = 1:length(train_dataset)
    data_name = sprintf('%s_%s', data_name, train_dataset{i});
end

list_dir    = fullfile('lists');
output_dir  = fullfile('imdb');
if( ~exist(output_dir, 'dir') )
    mkdir(output_dir);
end


%% training data
train_list = {};

for d = 1:length(train_dataset)
    
    image_dir   = fullfile('datasets', train_dataset{d});
    list_filename = fullfile(list_dir, sprintf('%s.txt', train_dataset{d}));
    image_list = load_list(list_filename);
    num_image = length(image_list);

    for i = 1:num_image
        
        filename = fullfile(image_dir, sprintf('%s.%s', image_list{i}, img_ext));
        fprintf('%s %d / %d: %s\n', train_dataset{d}, i, num_image, filename);
        if( ~exist(filename, 'file') )
            error('%s does not exist!\n', filename);
        end
        train_list{i} = filename;
    end

end

num_train = length(train_list);

%% validation data
valid_list = {};

for d = 1:length(valid_dataset)
    
    image_dir   = fullfile('datasets', valid_dataset{d}, 'GT');
    list_filename = fullfile(list_dir, sprintf('%s.txt', valid_dataset{d}));
    image_list = load_list(list_filename);
    num_image = length(image_list);

    for i = 1:num_image
        
        filename = fullfile(image_dir, sprintf('%s.%s', image_list{i}, img_ext));
        fprintf('%s %d / %d: %s\n', valid_dataset{d}, i, num_image, image_list{i});
        if( ~exist(filename, 'file') )
            error('%s does not exist!\n', filename);
        end
        valid_list{i} = filename;
    end

end

num_valid = length(valid_list);

fprintf('Collect %d training images\n', num_train);
fprintf('Collect %d validation images\n', num_valid);

% build imdb
clear images;
images.filename = [train_list'; valid_list'];

images.set = 2 * ones(1, num_train + num_valid);    % set = 2 for validation data
images.set(1:num_train) = 1;                        % set = 1 for training data


% save data

filename = sprintf('imdb_%s.mat', data_name);

filename = fullfile(output_dir, filename);
fprintf('Save %s\n', filename);
save(filename, 'images', '-v7.3');
