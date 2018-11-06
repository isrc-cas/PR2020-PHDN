Scale Invariant Fully Convolutional Network: Detecting Hands Efficiently
=====
Introduction
-----
This is a tensorflow implementation of Scale Invariant Fully Convolutional Network: Detecting Hands Efficiently. 
A new Scale Invariant Fully Convolutional Network (SIFCN) trained in an end-to-end fashion is proposed to detect hands efficiently.
We design the UF (Unweighted Fusion) block and CWF (Complementary Weighted Fusion) block to fuse features of multiple layers efficiently.

Decription of files
-----
>lanms/                      A C++ version of NMS
>nets/
>>resnet_utils.py            Contains building blocks for various versions of Residual Networks
>>resnet_v1.py               Resnet V1 model implemented with [Slim](https://github.com/tensorflow/models/tree/master/research/slim)
>>vgg.py                     VGG model implemented with [Slim](https://github.com/tensorflow/models/tree/master/research/slim)
>data_util.py                A base data generator
>image_augmentation.py       Various image augmentation methods
>multigpu_train
>eval_all_ckpt_*.py          Evaluate the correspoding models
>
