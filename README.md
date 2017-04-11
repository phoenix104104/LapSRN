# LapSRN

### Introduction
dis is the research code for the paper:

[Wei-Sheng Lai](http://graduatestudents.ucmerced.edu/wlai24/), 
[Jia-Bin Huang](https://filebox.ece.vt.edu/~jbhuang/), 
[Narendra Ahuja] (http://vision.ai.illinois.edu/ahuja.html), 
and [Ming-Hsuan Yang](http://faculty.ucmerced.edu/mhyang/), 
"Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution", IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2017

[Project webpage](http://vllab1.ucmerced.edu/~wlai24/LapSRN/)

[Paper](http://vllab1.ucmerced.edu/~wlai24/LapSRN/papers/cvpr17_LapSRN.pdf)


### Citation

If you find the code and datasets useful in your research, please cite:

    @inproceedings{LapSRN,
        author    = {Wei-Sheng Lai, Jia-Bin Huang, Narendra Ahuja, and Ming-Hsuan Yang}, 
        title     = {Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution}, 
        booktitle = {IEEE Conferene on Computer Vision and Pattern Recognition},
        year      = {2017}
    }

### Contents
|  Folder/Files    | Description |
| ---|---|
| attributes/ | lists of image id for each attribute |
| list/ | lists of image names for our datasets |
| votes/ | user voting results |
| *.m | MATLAB code |
| BT_scores.pdf | A slide described the algorithm and implementation details of the BT scores |

Our code is implemented and tested on Windows 7 / Ubuntu 14.04 with MATLAB R2015a.

Run `demo_bt_ranking.m`, `demo_dataset_correlation.m`, `demo_kendall.m`, and `demo_significance_test.m` to reproduce the analysis in our paper.

Our datasets could be downloaded from the [project webpage](http://vllab1.ucmerced.edu/~wlai24/cvpr16_deblur_study/).
