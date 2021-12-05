# CS299-Project---Sign-Language-Translator

## Model-Preparation

This Folder contains all the files I created for testing and preparing my ML model for the App. <br>
It has some files having OPENCV python scripts for trying out masking for better recognition. <br>
_sep_train_val_ python notebook was used for separating my training and validation data. <br>
_new_model-Copy1_ python notebook contains the script for training my model. <br>
Also trained on Kaggle's GPU for better performance of models.

The Base model is keras' MobileNetV2 with 'ImageNet' weights. The final model is trained on the hand dataset applying the concept of Transfer learning. <br>
The Model is saved in "save" folder and also as different h5 files. Note: mobilenetv2_192_kaggle_bm142.h5 is the latest and recommended one.<br> 

You can try out the model using opcam.py script.

## Android Application

An App created using TFLite module of mobnet_kaggle_192.tflite and CameraX API, allowing real-time sign recognition. Got to learn about handling callback events and handling events in threads for latency reduction.



