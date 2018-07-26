---
layout: post
title: Probabilistic Graphical Model - Intro
category: 
- blog
- pgm
elements:
- GraphicalModel
---

### Intro

Deep learning has attracted lot of traction from various researchers coming from different fields from manufacturing, biology, image processing and language processing.  In the series of the blog post, I would be discussing how Uncertainty is modelled in Deeplearning. Before we jump into what it means to model uncertainty, we should know why it is important to have Deep Neural Nets to further its own impact. 
	
### Motivation

The recent techniques of Recurrent Neural Nets,  Convolution Nets , Dropout and Regularization often lead to networks predicting deterministic functions. Models such as Gaussian Processes, often define probability distributions over functions with a confidence bounds for the Machine Learning system to do inference. The autonomous car would need to decide if it needs to be really careful about making its own decision by using other probabilistic estimates. Deep learning models predictions for such scenarios often leadto potential questions if the network is throwing random guesses or making sensible predictions.

 Uncertainty information is crucial for medical diagnosis practitioner to know, if a model is detecting lot of false positives but quantifies its high level uncertainty (low belief) for a given set of input data.


Here are the other scenarios where uncertainty information would be crucial :

Model Uncertainty : 
- Model Parameters : Large number of models paraters to choose from. 
- Structure Uncertainty : Whats the right model structure ?

Data  Uncertainty :
- Data out of Distribution: Does the data show lies in the data distribution used for training the systems.
- Noisy Data Distribution: Does the data used contains an inherent in measurement  error leading to inappropriate learning.


### Applications need to model Uncertainty

High frequency trading systems take decision, as control is handed over the machine learning systems to take economic decisions. Can systems detect they are in uncertain state of making decisions (by evaluating asset risk ) and let humans take control over it. Similar is the need with Medical Diagnosis and Autonomous vechicles.

References: 
- [Gal, Y. (2016). Uncertainty in deep learning (Doctoral dissertation, PhD thesis, University of Cambridge).](http://mlg.eng.cam.ac.uk/yarin/thesis/thesis.pdf)