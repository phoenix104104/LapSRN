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
1. [Demo](#demo)
1. [Training](#training)
1. [Testing](#testing)

### Introduction
The Laplacian Pyramid Super-Resolution Network (LapSRN) is a progressive super-resolution model that super-resolves an low-resolution images in a coarse-to-fine Laplacian pyramid framework.
Our method is fast and achieves state-of-the-art performance on five benchmark datasets for 4x and 8x SR.
For more details and evaluation results, please check out our [project webpage](http://vllab1.ucmerced.edu/~wlai24/LapSRN/) and [paper](http://vllab1.ucmerced.edu/~wlai24/LapSRN/papers/cvpr17_LapSRN.pdf).

![teaser](http://vllab1.ucmerced.edu/~wlai24/LapSRN/images/emma_text.gif)



### Citation

If you find the code and datasets useful in your research, please cite:
    
    @inproceedings{LapSRN,
        author    = {Lai, Wei-Sheng and Huang, Jia-Bin and Ahuja, Narendra and Yang, Ming-Hsuan}, 
        title     = {Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution}, 
        booktitle = {IEEE Conferene on Computer Vision and Pattern Recognition},
        year      = {2017}
    }
    

### Requirements and Dependencies
- MATLAB (we test with MATLAB R2015a on Ubuntu 14.04 and Windows 7)
- [MatConvNet](http://www.vlfeat.org/matconvnet/)

### Installation

    # Start MATLAB
    $ matlab
    >> install
   
This script will copy `vllab_dag_loss.m` to `matconvnet/matlab/+dagnn` and run `vl_compilenn` to setup matconvnet.

**Note**: `vllab_dag_loss.m` may not be properly copied on some systems (e.g., Windows 10). You need to manually copy it to `matconvnet/matlab/+dagnn`.

### Demo

To test LapSRN on a single-image:

    >> demo_LapSRN

This script will load the pretrained LapSRN model and apply SR on emma.jpg.

To test LapSRN on benchmark datasets, first download the testing datasets:

    $ cd datasets
    $ wget http://vllab1.ucmerced.edu/~wlai24/LapSRN/results/SR_testing_datasets.zip
    $ unzip SR_testing_datasets.zip
    $ cd ..

Then choose the evaluated dataset and upsampling scale in `evaluate_LapSRN_dataset.m` and run:

    >> evaluate_LapSRN_dataset

which can reproduce the results in our paper Table 4.


### Training

To train LapSRN from scratch, first download the training datasets:

    $ cd datasets
    $ wget http://vllab1.ucmerced.edu/~wlai24/LapSRN/results/SR_training_datasets.zip
    $ unzip SR_train_datasets.zip
    $ cd ..

or use the provided bash script to download all datasets and unzip at once:

    $ cd datasets
    $ ./download_SR_datasets.sh
    $ cd ..

Then, setup training options in `init_opts.m`, and run `train_LapSRN(scale, depth, gpuID)`. For example, to train LapSRN with depth = 10 for 4x SR using GPU ID = 1:

    >> train_LapSRN(4, 10, 1)
    
Note that we only test our code on single-GPU mode. MatConvNet supports training with multiple GPUs but you may need to modify our script and options (e.g., `opts.gpu`).

    
### Testing

Use `test_LapSRN(model_scale, depth, gpu, dataset, test_scale, epoch)` to test your own LapSRN model. For example, test LapSRN with depth = 10, scale = 4, epoch = 10 on Set5:

    >> test_LapSRN(4, 10, 1, 'Set5', 4, 10)

which will report the PSNR, SSIM and IFC.
