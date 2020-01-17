Towards Interpretable and Robust Hand Detection via Pixel-wise Prediction
=====
Introduction
-----
This is a tensorflow implementation of Towards Interpretable and Robust Hand Detection via Pixel-wise Prediction.  
A Pixel-wise Hand Detection Network (PHDN) trained in an end-to-end fashion is proposed to detect hands efficiently. We design the Highlight Feature Fusion (HFF) block to highlight the distinctive features among multiple layers and learn more discriminative features to make predictions. The PHDN architecture with VGG16 backbone: ![arch](images/arch.png)

### Citation
If you use our code or models, please cite our paper.  

    @article{LIU2020107202,
        author = "Dan Liu and Libo Zhang and Tiejian Luo and Lili Tao and Yanjun Wu",
        title = "Towards Interpretable and Robust Hand Detection via Pixel-wise Prediction",
        journal = "Pattern Recognition",
        pages = "107202",
        year = "2020",
        issn = "0031-3203",
        doi = "https://doi.org/10.1016/j.patcog.2020.107202"
        }

Decription of files
-----
|file/directory|discription|
|--------|--------|
|lanms/                |A C++ version of NMS|
|nets/                 |Contains Resnet V1 model and VGG16 model|
|iou-tracker/          |IOU Tracker proposed in the AVSS 2017 paper [High-Speed Tracking-by-Detection Without Using Image Information](http://elvera.nue.tu-berlin.de/files/1517Bochinski2017.pdf) [code](https://github.com/bochinski/iou-tracker/)|
|sort-tracker/         |A multiple object tracker [code](https://github.com/abewley/sort)|
|deep_sort/            |An extension the original SORT algorithm [code](https://github.com/nwojke/deep_sort)|
|data_util.py          |A base data generator|
|oxford.py　　　　　    |Data processor for Oxford dataset|
|image_augmentation.py |Various image augmentation methods|
|resnet_v1_model_dice_multi.py                        |ResNet50+BFF+Multi-Scale Model|
|resnet_v1_model_dice_multi_weighted_fusion.py        |ResNet50+HFF+Multi-Scale Model|
|vgg16_model_dice_multi.py                            |VGG16+BFF+Multi-Scale Model|
|vgg16_model_dice_multi_weighted_fusion.py            |VGG16+HFF+Multi-Scale Model|
|multigpu_train_dice_multi.py                         |Train ResNet50+BFF+Multi-Scale Loss|
|multigpu_train_dice_multi_weighted_fusion.py         |Train ResNet50+HFF+Multi-Scale Loss|
|multigpu_train_vgg16_dice_multi.py                   |Train VGG16+BFF+Multi-Scale Loss|
|multigpu_train_vgg16_dice_multi_weighted_fusion.py   |Train VGG16+HFF+Multi-Scale Loss|
|eval_all_ckpt_dice_multi.py                          |Evaluate ResNet50+BFF+Multi-Scale Loss|
|eval_all_ckpt_dice_multi_weighted_fusion.py          |Evaluate ResNet50+HFF+Multi-Scale Loss|
|eval_all_ckpt_vgg16_dice_multi.py                    |Evaluate VGG16+BFF+Multi-Scale Loss|
|eval_all_ckpt_vgg16_dice_multi_weighted_fusion.py    |Evaluate VGG16+HFF+Multi-Scale Loss|

Installation
------
Tensorflow > 1.0  
python3  
python packages:  
　　numpy  
　　shapely  
　　opencv-python   

Download
-----
### Data preparing  

[Oxford Hand Dataset](http://www.robots.ox.ac.uk/~vgg/data/hands)  
[VIVA Hand Dataset](http://cvrr.ucsd.edu/vivachallenge/index.php/hands/hand-detection)  
[VIVA Hand Tracking Dataset](http://cvrr.ucsd.edu/vivachallenge/index.php/hands/hand-tracking)

The ground truths should be named exactly the same as the corresponding image such as *VOC2010_1323.jpg* and *VOC2010_1323.txt*. And you are supposed to put the images and ground truths in the same directory. An example image and coresponding gound truth:  

![examples/VOC2010_1323.jpg](examples/VOC2010_1323.jpg)  

    60,269,78,257,87,270,69,283,hand
    118,244,130,245,128,260,116,259,hand
Each line in the ground truth file indicates a hand bounding box. The eight numbers seperated by "," represent the *x* and *y* coordinates of the bounding box in clockwise starting from the upper left. The text "hand" represents the catagory, which is pointless when there is only one catagory.  

### Pre-trained model  

You can download the pre-trained Resnet V1 50 and VGG16 models from the [slim](https://github.com/tensorflow/models/tree/master/research/slim) page.  

Train your detector
-----
You can train the detector with the following command:  

    python multigpu_train_dice_multi.py --input_size=512 --batch_size_per_gpu=12 --checkpoint_path=../model/hand_dice_multi_oxford_aug/ --training_data_path=../data/oxford/train/  --learning_rate=0.0001 --num_readers=16 --gpu_list=0 --restore=False --pretrained_model_path=../model/resnet50/resnet_v1_50.ckpt

Test your detector
-----
You can evaluate the model with the following command:  

    python eval_all_dice_multi.py --test_data_path=../data/Oxford/test/ --gpu_list=0 --checkpoint_path=../model/hand_dice_multi_oxford_aug/ --output_dir=../result/oxford-test-result/pos/ heatmap_output_dir=../result/oxford-test-result/heatmaps_resenet_dice_multi_oxford/ --no_write_images=True

Build your tracker
-----
You can build IOU tracker, SORT tracker, deep SORT tracker with the detection results generated by the PHDN model.

### IOU tracker

    cd iou-tracker
    python mydemo.py --detection_dir='your_path_to_detection_results' output_dir='track_results/model_'

### SORT tracker

    cd deep_sort
    python deep_sort_app.py --sequence_dir='your_path_to_video_sequence' output_dir='track_results/model_' -detection_dir='your_path_to_detection_results' output_dir='track_results/model_'

### deep SORT tracker

    cd sort_tracker
    python sort.py --display

Troubleshooting
-----
* 
    * 