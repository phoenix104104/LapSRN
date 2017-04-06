function inputs = getBatch_multiscale(opts, imdb, batch, mode)

    %% load images
    %image_batch  = batch_imread(imdb.images.filename(batch));
    image_batch = imdb.images.img(batch);
    
    %% crop
    B = opts.num_patch_per_img * opts.num_img_per_batch;
    HR = zeros(opts.patch_size, opts.patch_size, 1, B, 'single');

    
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
        
        r1 = floor(opts.patch_size / 2);
        r2 = opts.patch_size - r1 - 1;
        
        mask = zeros(H, W);
        mask(1 + r1 : end - r2, 1 + r1 : end - r2) = 1;
        
        [X, Y] = meshgrid(1:W, 1:H);
        X = X(mask == 1);
        Y = Y(mask == 1);
        
        select = randperm(length(X), opts.num_patch_per_img);
        X = X(select);
        Y = Y(select);
        
        for j = 1:length(X)
            idx = (i - 1) * opts.num_patch_per_img + j;
            x = X(j);
            y = Y(j);
            HR(:, :, :, idx) = img(y - r1 : y + r2, x - r1 : x + r2, :);
        end
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
    
    
    if( numel(opts.gpus) > 0 )
        for i = 2:2:length(inputs)
            inputs{i} = gpuArray(inputs{i});
        end
    end
    
end
