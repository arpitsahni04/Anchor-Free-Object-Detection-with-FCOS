# Anchor-Free-Object-Detection-with-FCOS
 ## Introduction

 I have implemented a former State of Art Anchor Free single-stage object detector based on **FCOS (Fully-Convolutional One-Stage Object Detection)** https://arxiv.org/abs/1904.01355v5. The goal of this project is to detect specific object classes. My detector closely follows the FCOS design, but I made some adjustments to create a smaller model with different hyperparameters. I had to do this because I'm working with limited AWS resources. We train our network on **Pascal VOC dataset ** http://host.robots.ox.ac.uk/pascal/VOC/

![Alt text](assets/FCOS.png)
We Implement **Feature Pyramid Networks for Object Detection** (https://arxiv.org/abs/1612.03144) from scratch and use that as the Neck for our Object Detector. We also implement a custom Prediction Head to work with this Network, along with Various Losses detailed as follows:


![Alt text](assets/FPN.png)

**Object classification**: FCOS uses Sigmoid Focal Loss, an extension of cross-entropy loss that deals with class-imbalance. FCOS faces a class imbalance issue because a majority of locations would be assigned “background”. If not handled properly, the model will simply learn to predict “background” for every location. We Implement this loss from scratch



**Box regression**: The FCOS paper uses Generalized Intersection-over-Union loss to minimize the difference between predicted and GT LTRB deltas. We  implement this loss from scratch Again.




**Centerness regression**: Centerness predictions and GT targets are real-valued numbers in [0, 1], so FCOS uses binary cross-entropy (BCE) loss to optimize it. One may use an L1 loss, but BCE empirically works slightly better.

## Enviornment Setup
```
$ conda create --name myenv python=3.7
$ pip install -r requirements.txt
```
---

## Training 
```
$ python3 train.py --overfit=False
```

## Inference
```
python3 train.py --inference 
```

##  Visualizations
### Feature Centers 
 ![Alt text](assets/image-1.png)
 ![Alt text](assets/image-2.png)
 ![Alt text](assets/image-3.png)

 ## Results and Inference 
 ![Alt text](assets/image-4.png)
 ![Alt text](assets/image-5.png)
 ![Alt text](assets/image-6.png)
 ![Alt text](assets/image-7.png)
 ![Alt text](assets/image-8.png)
 ![Alt text](assets/image-9.png)