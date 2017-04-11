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
Our method is fast and achieves state-of-the-art performance on many benchmark datasets.
For more details and evaluation results, please check out our [project webpage](http://vllab1.ucmerced.edu/~wlai24/LapSRN/) and [paper](http://vllab1.ucmerced.edu/~wlai24/LapSRN/papers/cvpr17_LapSRN.pdf).

![teaser](http://vllab1.ucmerced.edu/~wlai24/LapSRN/images/emma_v3_32x.gif)



### Citation

If you find the code and datasets useful in your research, please cite:
    
    @inproceedings{LapSRN,
        author    = {Wei-Sheng Lai, Jia-Bin Huang, Narendra Ahuja, and Ming-Hsuan Yang}, 
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
   
This script will copy vllab_dag_loss.m to matconvnet/matlab/+dagnn and run vl_compilenn to setup matconvnet.


### Demo

To test LapSRN on a single-image:

    >> demo_LapSRN

This script will load the pretrained LapSRN model and apply SR on emma.jpg.

To test LapSRN on benchmark datasets, first download the testing datasets into sub-folder ``dataset''

    $ cd datasets
    $ wget http://vllab1.ucmerced.edu/~wlai24/LapSRN/results/SR_testing_datasets.zip
    $ unzip SR_testing_datasets.zip

### Training


### Testing
