Scale Invariant Fully Convolutional Network: Detecting Hands Efficiently
=====
Introduction
-----
This is a tensorflow implementation of Scale Invariant Fully Convolutional Network: Detecting Hands Efficiently. 
A new Scale Invariant Fully Convolutional Network (SIFCN) trained in an end-to-end fashion is proposed to detect hands efficiently.
We design the UF (Unweighted Fusion) block and CWF (Complementary Weighted Fusion) block to fuse features of multiple layers efficiently.

Decription of files
-----
>lanms/
　　　　　　　　　　　　　　　A C++ version of NMS <br>
>nets/<br>
　resnet_utils.py
　　　　　　　　　　　　　　　Contains building blocks for various versions of Residual Networks<br>
　resnet_v1.py
　　　　　　　　　　　　　　　Resnet V1 model implemented with [Slim](https://github.com/tensorflow/models/tree/master/research/slim)<br>
　vgg.py　　　　　　　　　　　　　　VGG model implemented with [Slim](https://github.com/tensorflow/models/tree/master/research/slim)<br>
>data_util.py　　　　　　　　　　　　　A base data generator<br>
>oxford_R01.py　　　　　　　　　　　　Data processor for Oxford dataset<br>
>VIVA_R01.py　　　　　　　　　　　　　Data processor for VIVA dataset<br>
>image_augmentation.py　　　　　　　　Various image augmentation methods<br>
>multigpu_train_*.py　　　　　　　　　Train models<br>
>eval_all_ckpt_*.py				Evaluate the correspoding models<br>
>resnet_v1_model_*.py			SIFCN with Resnet V1 50 as the backbone network<br>
>vgg16_model_*.py				SIFCN with VGG16 as the backbone network<br>
>*_multi*.py					The multi-scale loss discribed in the paper is used<br>
>*_weighted_fusion*.py			The CWF block is used<br>, if not marked, the UF block is used as default<br>