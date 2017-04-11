function inputs = getBatch_LapSRN(opts, imdb, batch, mode)
% -------------------------------------------------------------------------
%   Description:
%       get one batch for training LapSRN
%
%   Input:
%       - opts  : options generated from init_opts()
%       - imdb  : imdb file generated from make_imdb()
%       - batch : array of ID to fetch
%       - mode  : 'train' or 'val'
%
%   Output:
%       - inputs: input for dagnn (include LR and HR images)
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

    %% get images
    image_batch = imdb.images.img(batch);
    
    %% crop
    HR = zeros(opts.patch_size, opts.patch_size, 1, length(batch), 'single');
    
    for i = 1:length(batch)
        
        img = image_batch{i};
        
        if( opts.data_augmentation && strcmp(mode, 'train') )
            % randomly resize between 0.5 ~ 1.0
            ratio = randi([5, 10]) * 0.1;
            img = imresize(img, ratio);
        end
        
        % min width/height should be larger than patch size
        if( size(img, 1) < opts.patch_size || size(img, 2) < opts.patch_size )
            img = vllab_imresize(img, opts.patch_size);
        end
        
        H = size(img, 1);
        W = size(img, 2);
        
        % random crop
        r1 = floor(opts.patch_size / 2);
        r2 = opts.patch_size - r1 - 1;
        
        mask = zeros(H, W);
        mask(1 + r1 : end - r2, 1 + r1 : end - r2) = 1;
        
        [X, Y] = meshgrid(1:W, 1:H);
        X = X(mask == 1);
        Y = Y(mask == 1);
        
        select = randperm(length(X), 1);
        X = X(select);
        Y = Y(select);
        
        HR(:, :, :, i) = img(Y - r1 : Y + r2, X - r1 : X + r2, :);
    end
      
    
    %% data augmentation
    if( opts.data_augmentation && strcmp(mode, 'train') )
        
        % rotate
        rotate = rand;
        if( rotate < 0.25 )
            HR = rot90(HR, 1);
        elseif( rotate < 0.5 )
            HR = rot90(HR, 2);
        elseif( rotate < 0.75 )
            HR = rot90(HR, 3);
        end
        
        % horizontally flip
        if( rand > 0.5 )
            HR = fliplr(HR);
        end
        
        % vertically flip
        if( rand > 0.5 )
            HR = flipud(HR);
        end

    end % end of data augmentation
    
    %% make dagnn input
    inputs = {};
    inputs{end+1} = 'level1_HR';
	inputs{end+1} = HR;
    
    for i = 2 : opts.level
        ratio = 1 / 2^(i - 1);
        inputs{end+1} = sprintf('level%d_HR', i);
        inputs{end+1} = imresize(HR, ratio);
    end
    
    inputs{end+1} = 'LR';
	inputs{end+1} = imresize(HR, 1 / opts.scale);
    
    % convert to GPU array
    if( numel(opts.gpus) > 0 )
        for i = 2:2:length(inputs)
            inputs{i} = gpuArray(inputs{i});
        end
    end
    
end
