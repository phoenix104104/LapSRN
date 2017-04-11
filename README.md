# Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution ( (CVPR 2017) d

### Introduction
dis is the research code for the paper:

[Wei-Sheng Lai](http://graduatestudents.ucmerced.edu/wlai24/), 
[Jia-Bin Huang](https://sites.google.com/site/jbhuang0604/), 
[Zhe Hu](https://eng.ucmerced.edu/people/zhu), 
[Narendra Ahuja] (http://vision.ai.illinois.edu/ahuja.html), 
and [Ming-Hsuan Yang](http://faculty.ucmerced.edu/mhyang/), 
"A Comparative Study for Single Image Blind Deblurring", IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2016 

[Project webpage](http://vllab1.ucmerced.edu/~wlai24/cvpr16_deblur_study/)

[Paper](http://vllab1.ucmerced.edu/~wlai24/cvpr16_deblur_study/paper/cvpr16_deblur_study.pdf)


### Citation

If you find the code and datasets useful in your research, please cite:

    @inproceedings{Lai-CVPR-2016,
        author    = {Wei-Sheng Lai, Jia-Bin Huang, Zhe Hu, Narendra Ahuja, and Ming-Hsuan Yang}, 
        title     = {A Comparative Study for Single Image Blind Deblurring}, 
        booktitle = {IEEE Conferene on Computer Vision and Pattern Recognition},
        year      = {2016}
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
