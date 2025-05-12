# Face-Mask-Detection-System

## Team Members

* Muhammad Qasim (22I-1994)
* Ayaan Khan (22I-2066)
* Abubakar Nadeem (22I-2003)
* Ahmed Mehmood (22I-1915)

## Overview

This project presents a real-time face mask detection system using computer vision and deep learning. The system can identify whether individuals are wearing masks from image and video inputs. It is designed for use in public safety applications such as compliance monitoring during pandemics.

## Dataset

* *Source*: [Kaggle Face Mask Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)
* *Structure*:

  * with_mask/: \~3,725 images
  * without_mask/: \~3,725 images
* *Split*: 80% training, 20% testing (stratified)

## Image Preprocessing Pipeline

1. *Resize*: 224x224 pixels
2. *Color Conversion*: BGR to RGB and grayscale
3. *Histogram Equalization*: Improves contrast
4. *Gaussian Blur*: Noise reduction
5. *Otsu Thresholding*: Image binarization
6. *Morphological Operations*: Noise removal and component connection
7. *Masking*: Binary mask applied to RGB image
8. *Model Preprocessing*: MobileNetV2 normalization

## Model Architecture

### MobileNetV2 (Transfer Learning)

* Lightweight CNN optimized for edge devices
* Key Features:

  * Inverted Residual Blocks
  * Linear Bottlenecks
  * Depthwise Separable Convolutions

### Fine-Tuning Strategy

* Learning Rate Scheduling
* Early Stopping
* Dropout (20%)
* Fine-tuning last 30% of layers
* Data Augmentation: flips, rotations, zooms, contrast

## Face Detection

* *Model*: SSD with ResNet-10 Backbone
* *Files*:

  * deploy.prototxt: Network config
  * res10_300x300_ssd_iter_140000_fp16.caffemodel: Weights
* *Framework*: OpenCV DNN module

## Web Application

* *Framework*: Flask
* *Features*:

  * Real-time video mask detection
  * Image upload and processing
  * Statistics (mask detection counts and confidence)
  * Visualization with bounding boxes and labels

## Results

* *Accuracy*: High accuracy in detecting face masks
* *Performance*: Real-time detection on standard hardware
* *Robustness*: Works under varying lighting conditions

### Visual Output Examples

* Flask interface with real-time detection
* Correct detection of masks (green boxes) and absence (red boxes)

## Limitations

* Reduced performance with occlusions or extreme head poses
* Computationally intensive for high-resolution streams

## Future Enhancements

* Multi-class mask type detection
* Edge device optimization
* Compliance statistics over time
* Integration with access control systems or crowd monitoring
* Mobile app version

## Conclusion

The system combines deep learning, image processing, and web deployment to deliver a practical face mask detection solution. Using MobileNetV2 and OpenCV’s SSD face detector, the model delivers real-time performance and high accuracy. The Flask app provides a user-friendly interface for practical deployment.
