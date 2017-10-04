% -------------------------------------------------------------------------
%   Description:
%       Script to evaluate pretrained MS-LapSRN on benchmark datasets
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

%% testing options
% dataset     = 'Set5';
% dataset     = 'Set14';
dataset     = 'BSDS100';
% dataset     = 'Urban100';
% dataset     = 'Manga109';

% model_name  = 'MSLapSRN_D5R2';
% model_name  = 'MSLapSRN_D5R5';
model_name  = 'MSLapSRN_D5R8';

model_scale = 2;            % pretrained model upsampling scale
test_scale  = model_scale;  % testing scale can be different from model scale
gpu         = 1;            % GPU ID, gpu = 0 for CPU mode
compute_ifc = 0;            % IFC calculation is slow, enable when needed

%% setup paths
input_dir = fullfile('datasets', dataset);
output_dir = fullfile('results', dataset, sprintf('x%d', test_scale), model_name);

if( ~exist(output_dir, 'dir') )
    mkdir(output_dir);
end

addpath(genpath('utils'));
addpath(fullfile(pwd, 'matconvnet/matlab'));
vl_setupnn;

%% Load pretrained multi-scale model
model_filename = fullfile('pretrained_models', sprintf('%s.mat', model_name));
fprintf('Load %s\n', model_filename);

net = load(model_filename);
net_trained = dagnn.DagNN.loadobj(net.net);

opts_filename = fullfile('pretrained_models', sprintf('%s_opts.mat', model_name));
fprintf('Load %s\n', opts_filename);

opts = load(opts_filename);
opts = opts.opts;

opts.scales = [model_scale];

%% create single-scale model
net = init_MSLapSRN_model(opts, 'test');

%% copy pretrained weights
fprintf('Copy weights to single scale model...\n');
net = copy_model_weights(net, net_trained);

if( gpu ~= 0 )
    gpuDevice(gpu)
    net.move('gpu');
end
%% load image list
list_filename = fullfile('lists', sprintf('%s.txt', dataset));
img_list = load_list(list_filename);
num_img = length(img_list);


%% testing
PSNR = zeros(num_img, 1);
SSIM = zeros(num_img, 1);
IFC  = zeros(num_img, 1);

for i = 1:num_img
    
    img_name = img_list{i};
    fprintf('Testing %s on %s %dx: %d/%d: %s\n', model_name, dataset, test_scale, i, num_img, img_name);
    
    %% Load GT image
    input_filename = fullfile(input_dir, sprintf('%s.png', img_name));
    img_GT = im2double(imread(input_filename));
    img_GT = mod_crop(img_GT, test_scale);

    %% generate LR image
    img_LR = imresize(img_GT, 1/test_scale, 'bicubic');
    
    %% apply SR
    img_HR = SR_MSLapSRN(img_LR, net, model_scale, test_scale, gpu);
    
    %% save result
    output_filename = fullfile(output_dir, sprintf('%s.png', img_name));
    fprintf('Save %s\n', output_filename);
    imwrite(img_HR, output_filename);
    
    %% evaluate
    [PSNR(i), SSIM(i), IFC(i)] = evaluate_SR(img_GT, img_HR, test_scale, compute_ifc);
    
end

PSNR(end+1) = mean(PSNR);
SSIM(end+1) = mean(SSIM);
IFC(end+1)  = mean(IFC);

fprintf('Average PSNR = %f\n', PSNR(end));
fprintf('Average SSIM = %f\n', SSIM(end));
fprintf('Average IFC = %f\n', IFC(end));

filename = fullfile(output_dir, 'PSNR.txt');
save_matrix(PSNR, filename);

filename = fullfile(output_dir, 'SSIM.txt');
save_matrix(SSIM, filename);

filename = fullfile(output_dir, 'IFC.txt');
save_matrix(IFC, filename);

