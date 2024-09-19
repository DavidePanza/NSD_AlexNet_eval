***Project Overview:***

This repository contains the scripts used for my project, which evaluates the performance of AlexNet on recognizing a subset of images from the NSD (Natural Scenes Dataset). The dataset is referenced from the following publication:
Allen, E.J., St-Yves, G., Wu, Y., et al. A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial intelligence. Nature Neuroscience (2021).

The project focuses on evaluating AlexNet’s capabilities in image recognition through several key steps, from image selection and processing to model training and performance assessment.

<br>

***Project Workflow:***


***Image Selection and Processing:***

NSD images are derived from the MS COCO dataset (Lin et a., 2014. Microsoft COCO: Common Objects in Context. In European Conference on Computer Vision (ECCV) (pp. 740-755). Springer.)  
Three different subsets of images are created:

- Single-labeled images.
- Multi-labeled images including the category "person."
- Multi-labeled images excluding the category "person."
  
Image cropping and storage steps are also performed to prepare the data for model training.

<br>

***AlexNet Training and Testing:***

AlexNet is trained and tested on the selected image subsets.  
Features are extracted from AlexNet’s fc_3 layer.  
Performance is evaluated using the d-prime (d') metric, with outputs softmax-transformed.

<br>

***SVM Training and Testing:***

A linear classifier (SVM) is trained and tested on the features extracted from AlexNet.  
Performance is assessed using d-prime (d'), with SVM outputs either:

- Sigmoid-transformed probabilities.
- Calibrated probabilities using Platt scaling.

----------
Relevant scripts to run the different processing stages can be found in the ipynb folder
