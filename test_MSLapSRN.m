function test_MSLapSRN(model_name, model_scale, epoch, dataset, test_scale, gpu, save_img, compute_ifc)
% -------------------------------------------------------------------------
%   Description:
%       Script to test MS-LapSRN on benchmark datasets
%       Compute PSNR, SSIM and IFC
%
%   Input:
%       - model_name    : model filename saved in 'models' folder
%       - model_scale   : model upsampling scale for constructing pyramid
%       - epoch         : model epoch to test
%       - dataset       : testing dataset (Set5, Set14, BSDS100, Urban100, Manga109)
%       - test_scale    : testing SR scale
%       - gpu           : GPU ID
%       - save_img      : Save output images or not [Default = 0]
%       - compute_ifc   : Compute IFC or not [Default = 0]
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
    
    if nargin < 5
        error('test_MSLapSRN(model_filename, epoch, dataset, test_scale, gpu, [save_img = 0]');
    end

    if ~exist('save_img', 'var')
        save_img = 0;
    end
    
    if ~exist('compute_ifc', 'var')
        compute_ifc = 0;
    end
    
    %% setup paths
    addpath(genpath('utils'));
    addpath(fullfile(pwd, 'matconvnet/matlab'));
    vl_setupnn;
    
    
    %% Load multi-scale model
    model_filename = fullfile('models', model_name, sprintf('net-epoch-%d.mat', epoch));
    fprintf('Load %s\n', model_filename);
    
    net = load(model_filename);
    net_trained = dagnn.DagNN.loadobj(net.net);

    opts_filename = fullfile('models', model_name, 'opts.mat');
    fprintf('Load %s\n', opts_filename);

    opts = load(opts_filename);
    opts = opts.opts;

    opts.scales = [model_scale];

    %% create single-scale model
    net = init_MSLapSRN_model(opts, 'test');

    %% copy pretrained weights
    fprintf('Copy weights to single scale model...\n');
    net = copy_model_weights(net, net_trained);

    num_params = count_network_parameters(net);
    fprintf('================================\n');
    fprintf('Total %d network parameters\n', num_params);
    fprintf('================================\n');

    if( gpu )
        gpuDevice(gpu)
        net.move('gpu');
    end

    %% image path
    input_dir = fullfile('datasets', dataset);
    output_dir = fullfile('models', model_name, sprintf('epoch_%d', epoch), ...
                          dataset, sprintf('x%d', test_scale));

    if( ~exist(output_dir, 'dir') )
        mkdir(output_dir);
    end

    %% load image list
    list_filename = sprintf('lists/%s.txt', dataset);
    img_list = load_list(list_filename);
    num_img = length(img_list);
    

    %% testing
    PSNR = zeros(num_img, 1);
    SSIM = zeros(num_img, 1);
    IFC  = zeros(num_img, 1);
    time = zeros(num_img, 1);
    
    for i = 1:num_img
        
        img_name = img_list{i};
        fprintf('Testing %s %dx: %d/%d: %s: epoch %d\n', dataset, test_scale, i, num_img, model_name, epoch);
        
        %% Load HR image
        input_filename = fullfile(input_dir, sprintf('%s.png', img_name));
        img_GT = im2double(imread(input_filename));
        img_GT = mod_crop(img_GT, test_scale);
    
        %% generate LR image
        img_LR = imresize(img_GT, 1/test_scale, 'bicubic');
        
        %% apply SR
        [img_HR, time(i)] = SR_MSLapSRN(img_LR, net, model_scale, test_scale, gpu);
        
        % if out of memory, use patch-based SR
        %[img_HR, time(i)] = SR_patch_MSLapSRN(img_LR, net, model_scale, test_scale, gpu);
        
        %% save result
        if( save_img )
            output_filename = fullfile(output_dir, sprintf('%s.png', img_name));
            fprintf('Save %s\n', output_filename);
            imwrite(img_HR, output_filename);
        end
        
        %% evaluate
        [PSNR(i), SSIM(i), IFC(i)] = evaluate_SR(img_GT, img_HR, test_scale, compute_ifc);

    end
    
    PSNR(end+1) = mean(PSNR);
    SSIM(end+1) = mean(SSIM);
    IFC(end+1)  = mean(IFC);
    time(end+1) = mean(time);
    
    
    filename = fullfile(output_dir, 'PSNR.txt');
    fprintf('Save %s\n', filename);
    save_matrix(PSNR, filename);

    filename = fullfile(output_dir, 'SSIM.txt');
    fprintf('Save %s\n', filename);
    save_matrix(SSIM, filename);
    
    filename = fullfile(output_dir, 'IFC.txt');
    fprintf('Save %s\n', filename);
    save_matrix(IFC, filename);

    
    fprintf('Average PSNR = %f\n', PSNR(end));
    fprintf('Average SSIM = %f\n', SSIM(end));
    fprintf('Average IFC = %f\n', IFC(end));
    fprintf('Average Time = %f\n', time(end));
