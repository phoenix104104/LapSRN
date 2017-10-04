function inputs = getBatch_MSLapSRN(opts, imdb, batch, mode)
% -------------------------------------------------------------------------
%   Description:
%       get one batch for training MS-LapSRN
%       We equally split a batch for multiple scales (opts.scales)
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
%       Fast and Accurate Image Super-Resolution with Deep Laplacian Pyramid Networks
%       Wei-Sheng Lai, Jia-Bin Huang, Narendra Ahuja, and Ming-Hsuan Yang
%       arXiv, 2017
%
%   Contact:
%       Wei-Sheng Lai
%       wlai24@ucmerced.edu
%       University of California, Merced
% -------------------------------------------------------------------------

    %% get images
    image_batch = imdb.images.img(batch);
    
    %% crop patches
    HR = zeros(opts.patch_size, opts.patch_size, 1, length(batch), 'single');
    
    for i = 1:length(batch)
        
        img = image_batch{i};
        
        ratio = 1;
        
        if( opts.scale_augmentation && strcmp(mode, 'train') )
            % randomly resize between 0.5 ~ 1.0
            ratio = randi([5, 10]) * 0.1;

        end
        
        eps = 1e-3;
        
        % min width/height should be larger than patch size
        if size(img, 1) < size(img, 2)
            if size(img, 1) * ratio < opts.patch_size
                ratio = opts.patch_size / size(img, 1) + eps;
            end
        else
            if size(img, 2) * ratio < opts.patch_size
                ratio = opts.patch_size / size(img, 2) + eps;
            end
        end
        
        img = imresize(img, ratio);
        
        % random crop
        H = size(img, 1);
        W = size(img, 2);
        
        y = randi(H - opts.patch_size + 1);
        x = randi(W - opts.patch_size + 1);
        HR(:, :, :, i) = img(y : y + opts.patch_size - 1, x : x + opts.patch_size - 1, :);

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
        
    end % end of data augmentation
    
    
    %% dagnn input
    inputs = {};

    split_size = ceil(length(batch) / length(opts.scales));
    st = (0 : length(opts.scales) - 1) * split_size + 1;
    ed = min((1 : length(opts.scales)) * split_size, length(batch));

    for s = 1:length(opts.scales)
        
        scale = opts.scales(s);
        level = ceil(log(scale) / log(2));

        HR_split = HR(:, :, :, st(s) : ed(s));
        
        % LR
        inputs{end + 1} = sprintf('x%dSR_LR', scale);
        inputs{end + 1} = imresize(HR_split, 1 / scale, 'bicubic');

        % intermediate HR
        for l = 1 : level - 1
            inputs{end + 1} = sprintf('x%dSR_%dx_HR', scale, 2^l);
            inputs{end + 1} = imresize(HR_split, 1 / 2^(level - l), 'bicubic');
        end

        % HR
        inputs{end + 1} = sprintf('x%dSR_%dx_HR', scale, scale);
        inputs{end + 1} = HR_split;

    end

    %% convert to GPU array
    if( opts.gpu > 0 )
        for i = 2:2:length(inputs)
            inputs{i} = gpuArray(inputs{i});
        end
    end
end
