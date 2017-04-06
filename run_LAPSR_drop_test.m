function run_LAPSR_drop_test(scale, pw, depth, loss, lr, lr_step, lr_drop, gc, gpu, ...
                             dataset, test_scale, epoch)
    
% 	test_scale = scale;
    test_dataset = dataset;
    
    if( nargin < 7 )
        fprintf('Usage: run_holistic_residual_SR_test(pw, model_scale, depth, lr, gpu, testing_dataset, test_scale, epoch)\n');
        return;
    end
    
    img_ext = 'png';
    
    %% parameters
    opts.gpu                = gpu;
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

    %% setting
    bd = test_scale;
    
    addpath(genpath('../utils'));
    addpath(fullfile(pwd, '../../../deep_interpolation/code/matconvnet/matlab'));
    vl_setupnn;

    %% Load model
    opts.model_dir = fullfile('model', opts.train_dataset);
    opts.train.expDir = fullfile(opts.model_dir, opts.model_name) ; % model output dir
      
    model_filename = fullfile(opts.train.expDir, sprintf('net-epoch-%d.mat', epoch));

    fprintf('Load %s\n', model_filename);
    net = load(model_filename);
    net = dagnn.DagNN.loadobj(net.net);
    net.mode = 'test' ;

    output_var = 'level1_output';
    output_index = net.getVarIndex(output_var);
    net.vars(output_index).precious = 1;

    if( opts.gpu )
        gpuDevice(opts.gpu)
        net.move('gpu');
    end

    %% input/output path
    output_dir = fullfile(opts.train.expDir, ...
                          sprintf('epoch_%d', epoch), ...
                          test_dataset, sprintf('x%d', test_scale));

    if( ~exist(output_dir, 'dir') )
        mkdir(output_dir);
    end


    %% image list
    input_dir = fullfile('../../dataset', test_dataset, ...
                         'test', 'gt');
                     
    list_filename = sprintf('../../list/%s_test.txt', test_dataset);
    img_list = load_list(list_filename);
    num_img = length(img_list);
    

    PSNR = zeros(num_img, 1);
    SSIM = zeros(num_img, 1);
    IFC  = zeros(num_img, 1);
    time = zeros(num_img, 1);
    
    for i = 1:num_img

        % Load HR image
        img_name = sprintf('%s.%s', img_list{i}, img_ext);
        filename = fullfile(input_dir, img_name);
        fprintf('Load image %d/%d: %s\n', i, num_img, filename);

        img_GT = im2double(imread(filename));
        img_GT = mod_crop(img_GT, test_scale);
    
        % Generate LR image
        img = imresize(img_GT, 1/test_scale);
        
        % RGB to YUV
        if( size(img, 3) > 1 )
            img = rgb2ycbcr(img);
        end
            
        y = single(img(:, :, 1));
        
        % bicubic upsample UV
        img = imresize(img, test_scale);
        
        
        if( opts.gpu )
            y = gpuArray(y);
        end

        % SR
        inputs = {'LR', y};
        
        tic;
        net.eval(inputs);
        time(i) = toc;
        
        y = gather(net.vars(output_index).value);
        
        
        
        if( size(y, 1) ~= size(img, 1) )
            y = imresize(y, [size(img, 1), size(img, 2)]);
        end
    
        img(:, :, 1) = double(y);
        
        % YUV to RGB
        if( size(img, 3) > 1 )
            img = ycbcr2rgb(img);
        end
        
%         % save image
        img_name = sprintf('%s.png', img_list{i});
        output_filename = fullfile(output_dir, img_name);
% 
        fprintf('Save %s\n', output_filename);
        imwrite(img, output_filename);
%             
        %% evaluation
%         img = im2double(imread(output_filename));
        img = im2uint8(img);
        img_GT = im2uint8(img_GT);
        
        % convert to gray scale
        img_GT = rgb2ycbcr(img_GT); img_GT = img_GT(:, :, 1);
        img = rgb2ycbcr(img); img = img(:, :, 1);
        
        % shave boundary
        img_GT = shave_bd(img_GT, bd);
        img = shave_bd(img, bd);
        
        % evaluate
%         [PSNR(i), SSIM(i)] = ComputePSNR_SSIM(img_GT, img);
        PSNR(i) = psnr(img_GT, img);
        SSIM(i) = ssim(img_GT, img);
        

        
%         IFC(i) = ifcvec( img_GT, img );
%         if( ~isreal(IFC(i)) )
%             IFC(i) = 0;
%         end
    end
    
    PSNR(end+1) = mean(PSNR);
    SSIM(end+1) = mean(SSIM);
    IFC(end+1) = mean(IFC);
    time(end+1) = mean(time);
    
    fprintf('Average PSNR = %f\n', PSNR(end));
    fprintf('Average SSIM = %f\n', SSIM(end));
    fprintf('Average IFC = %f\n', IFC(end));
    fprintf('Average time = %f\n', time(end));
    
    filename = fullfile(output_dir, 'PSNR.txt');
    save_matrix(PSNR, filename);

    filename = fullfile(output_dir, 'SSIM.txt');
    save_matrix(SSIM, filename);
    
    filename = fullfile(output_dir, 'IFC.txt');
    save_matrix(IFC, filename);
