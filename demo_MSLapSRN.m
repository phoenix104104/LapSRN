clear all;
% -------------------------------------------------------------------------
%   Description:
%       Script to demo MS-LapSRN for one image
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

img_filename = 'emma.jpg';

%% parameters
%model_name  = 'MSLapSRN_D5R2';
%model_name  = 'MSLapSRN_D5R5';
model_name  = 'MSLapSRN_D5R8';

model_scale = 4; % model SR scale
test_scale  = 4; % testing SR scale
gpu         = 1; % GPU ID

%% setup paths
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

%% Load GT image
fprintf('Load %s\n', img_filename);
img_GT = im2double(imread(img_filename));
img_GT = mod_crop(img_GT, test_scale);

%% Generate LR image
img_LR = imresize(img_GT, 1/test_scale);

%% apply LapSRN
fprintf('Apply MS-LapSRN for %dx SR\n', test_scale);
img_HR = SR_MSLapSRN(img_LR, net, model_scale, test_scale, gpu);

%% show results
img_LR = imresize(img_LR, test_scale);
figure, imshow(cat(2, img_LR, img_HR, img_GT));
title(sprintf('Bicubic %dx    |    MS-LapSRN %dx    |    Ground Truth', test_scale, test_scale));

