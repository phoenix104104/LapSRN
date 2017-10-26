# Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution (CVPR 2017)

[Wei-Sheng Lai](http://graduatestudents.ucmerced.edu/wlai24/), 
[Jia-Bin Huang](https://filebox.ece.vt.edu/~jbhuang/), 
[Narendra Ahuja](http://vision.ai.illinois.edu/ahuja.html), 
and [Ming-Hsuan Yang](http://faculty.ucmerced.edu/mhyang/)

IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2017

### Table of Contents
1. [Introduction](#introduction)
1. [Citation](#citation)
1. [Requirements and Dependencies](#requirements-and-dependencies)
1. [Installation](#installation)
1. [Test Pre-trained Models](#test-pre-trained-models)
1. [Training LapSRN](#training-lapsrn)
1. [Training MS-LapSRN](#training-ms-lapsrn)
1. [Third-Party Implementation](#third-party-implementation)

### Introduction
The Laplacian Pyramid Super-Resolution Network (LapSRN) is a progressive super-resolution model that super-resolves an low-resolution images in a coarse-to-fine Laplacian pyramid framework.
Our method is fast and achieves state-of-the-art performance on five benchmark datasets for 4x and 8x SR.
For more details and evaluation results, please check out our [project webpage](http://vllab.ucmerced.edu/wlai24/LapSRN/) and [paper](http://vllab.ucmerced.edu/wlai24/LapSRN/papers/cvpr17_LapSRN.pdf).

![teaser](http://vllab.ucmerced.edu/wlai24/LapSRN/images/emma_text.gif)



### Citation

If you find the code and datasets useful in your research, please cite:
    
    @inproceedings{LapSRN,
        author    = {Lai, Wei-Sheng and Huang, Jia-Bin and Ahuja, Narendra and Yang, Ming-Hsuan}, 
        title     = {Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution}, 
        booktitle = {IEEE Conferene on Computer Vision and Pattern Recognition},
        year      = {2017}
    }
    

### Requirements and Dependencies
- MATLAB (we test with MATLAB R2017a on Ubuntu 16.04 and Windows 7)
- Cuda & Cudnn (we test with Cuda 8.0 and Cudnn 5.1)

### Installation
Download repository:

    $ git clone https://github.com/phoenix104104/LapSRN.git

Run install.m in MATLAB to compile MatConvNet:

    # Start MATLAB
    $ matlab
    >> install
   
If you install MatConvNet in your own path, you need to change the corresponding path in `install.m`, `train_LapSRN.m` and `test_LapSRN.m`.

### Test Pre-trained Models

To test LapSRN / MS-LapSRN on a single-image:

    >> demo_LapSRN
    >> demo_MSLapSRN

This script will load the pretrained LapSRN / MS-LapSRN model and apply SR on emma.jpg.

To test LapSRN / MS-LapSRN on benchmark datasets, first download the testing datasets:

    $ cd datasets
    $ wget http://vllab1.ucmerced.edu/~wlai24/LapSRN/results/SR_testing_datasets.zip
    $ unzip SR_testing_datasets.zip
    $ cd ..

Then choose the evaluated dataset and upsampling scale in `evaluate_LapSRN_dataset.m` and `evaluate_MSLapSRN_dataset.m`, and run:

    >> evaluate_LapSRN_dataset
    >> evaluate_MSLapSRN_dataset

which can reproduce the results in our paper.


### Training LapSRN

To train LapSRN from scratch, first download the training datasets:

    $ cd datasets
    $ wget http://vllab1.ucmerced.edu/~wlai24/LapSRN/results/SR_training_datasets.zip
    $ unzip SR_train_datasets.zip
    $ cd ..

or use the provided bash script to download all datasets and unzip at once:

    $ cd datasets
    $ ./download_SR_datasets.sh
    $ cd ..

Then, setup training options in `init_LapSRN_opts.m`, and run `train_LapSRN(scale, depth, gpuID)`. For example, to train LapSRN with depth = 10 for 4x SR using GPU ID = 1:

    >> train_LapSRN(4, 10, 1)
    
Note that we only test our code on single-GPU mode. MatConvNet supports training with multiple GPUs but you may need to modify our script and options (e.g., `opts.gpu`).

To test your trained LapSRN model, use `test_LapSRN(model_name, epoch, dataset, test_scale, gpu)`. For example, test LapSRN with depth = 10, scale = 4, epoch = 10 on Set5:

    >> test_LapSRN('LapSRN_x4_depth10_L1_train_T91_BSDS200_pw128_lr1e-05_step50_drop0.5_min1e-06_bs64', 10, 'Set5', 4, 1)

which will report the PSNR and SSIM.


### Training MS-LapSRN

Setup training options in `init_MSLapSRN_opts.m`, and run `train_MSLapSRN(scales, depth, recursive, gpuID)`, where `scales` should be a vector, e.g., [2, 4, 8]. For example, to train MS-LapSRN with D = 5, R = 2 for 2x, 4x and 8x SR:

    >> train_MSLapSRN([2, 4, 8], 5, 2, 1)
    
To test your trained MS-LapSRN model, use `test_MS-LapSRN(model_name, model_scale, epoch, dataset, test_scale, gpu)`, where `model_scale` is used to define the number of pyramid levels. `test_scale` could be different from `model_scale`. For example, test MS-LapSRN-D5R2 with two pyramid levels (`model_scale = 4`), epoch = 10, on Set5 for 3x SR:

    >> test_MSLapSRN('MSLapSRN_x248_SS_D5_R2_fn64_L1_train_T91_BSDS200_pw128_lr5e-06_step100_drop0.5_min1e-06_bs64', 4, 10, 'Set5', 3, 1)

which will report the PSNR and SSIM.

### Third-Party Implementation

- [Pytorch](https://github.com/twtygqyy/pytorch-LapSRN)
- [TensorFlow](https://github.com/zjuela/LapSRN-tensorflow)
