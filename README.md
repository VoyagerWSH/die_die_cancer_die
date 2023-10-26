
# Predicting Lung Cancer from Low-Dose CT Scans
#### CPH 200A Project 2
#### Due Date: 5PM PST Nov 16, 2023

## Introduction
Building on the skills you developed in Project 1, in this project you will develop deep learning tools to perform lung cancer detection, localization and risk estimation using low-dose CT scans from the National Lung Screening Trial (NLST).The goal of this project is to give you hands-on experience developing state-of-the-art neural networks and to analyze the clinical opportunities they enable. At the end of this project, you will write a short project report, describing your model and your analyses.  Submit your **project code** and a **project report** by the due date, **5PM pst on Nov 16th, 2023**. 

## Part 0: Setup

For a refresher on how to access the CPH App nodes, setting up your development environment or using SGE, please refer to the Project 1 `README.md`]. In addition to the package requirement for project 1, make sure to install the packages listed in project 2s `requirements.txt` file.

As before, you can check your installation was succesful by running `python check_installation.py` from the project 2 directory. 

The NLST dataset metadata is availale at:
`/wynton/protected/group/cph/cornerstone/nlst-metadata/`

Preprocessed NLST scans (compatabile with the included data loaders) are included at:
`/scratch/datasets/nlst/preprocessed/`

Note, the scan's themselves are saved on local NVMe storage to accelerate your experiments IO. 

## Part 1: Build toy-cancer models with PathMNIST 
In this part of the project, we'll leverage a toy dataset [PathMNIST](https://medmnist.com/) to introduce [PyTorch](https://pytorch.org/), [PyTorchLightning](https://lightning.ai/docs/pytorch/stable/) and [Wandb](https://wandb.ai/). With these tools, you'll train a series of increasingly complex models to tiny (`28x28 px`) pathology images and study the impact of various design choices on model performance.

Note, there is a **huge** design space in neural network design, and so you may find extending your `dispatcher.py` from Project 1 to be a useful tool for managing your experiments. You may also find the starter code in `main.py`, `lightning.py` and `dataset.py` to be useful starting points for your experiments. 

### 1.1: Training Simple Neural Networks with PyTorch Lightning (20 pts)

In this exercise, develop a simple neural network to classify pathology images from the PathMNIST dataset. Develop the following models:

- Linear Model
- MLP Model 
- Simple CNN Model
- ResNet-18 Model (with and without ImageNet pretraining)

In doing so, explore the impact of model depth (i.e num layers), batch normalization, data augmentation and hidden dimensions on model performance. In the context of ResNet models, explore the impact of pretraining.

Your best model should be able to reach a validation accuracy of at least 99%. In your project report, include plots comparing the model variants and the impact of the design choices you explored.

## Part 2: Build a cancer detection model with NLST

Now that you have experience developing deep learning models on toy datasets, it's time to apply these skills to a real world problem. In this part of the project, you will develop a deep learning model to predict lung cancer from low-dose CT scans from the National Lung Screening Trial (NLST). As before, you may find the project2 starter code helpful in getting started. Note, these experiments will be much more computationally intensive than the toy experiments in part 1, so you may find it useful to use the SGE cluster to run your experiments and to use multiple GPUs per experiment. You may also find it useful to use the `torchio` library for data augmentations.

### 2.1: Building cancer detection classifier (25 pts)

In this exercise, develop classifiers to predict if a patient will be diagnosed with cancer within 1 year of their CT scan. In `src/dataset.py`, you'll find the `NLST` LightningDataModule which will load a preprocessed version of the dataset where CT scans are downsampled to a resolution of `256x256x200` and stored on the fast NVME local storage.

Develop a lung cancer binary classifer and explore the impact of pretraining and model architectures on model performance. In your project report, please include experiments with the following models:

- A simple 3D CNN model (extending your toy experiment)
- A ResNet-18 model (with and without ImageNet pretraining) (adapted to 3D CT scans)
- ResNet-3D models (with and without video pretrainig)
- (Optional) Swin-T models (with Video pretraining)

In addition to these experiments, please also include an exploration of why pretraining helps model performance. To what extent is the performance boost driven by feature transfer as opposed a form of optimization preconditioning?  Please design experiments to address this question and include your results in your project report.

By the end of this part of the project, your validation 1-Year AUC should be at least 0.80. 

### 2.2: Building a better model with localization (25 pts)
In addition to cancer labels, our dataset also contains region annotations for each cancer CT scan. In this exercise, you will leverage this information to improve your cancer detection model.  The bounding box data and the equivalent segementation masks are loaded for you in `src/dataset.py`. 

In your project report, please:
- Introduce your method to incorporate localization information into your model
  -  Note, there are many valid options here!
- Add metrics to quantify the quality of your localizations (e.g. IoU, or a likelihood metric)
- Add vizualizations of your generated localizations against the  ground truth

By the end of this part of the project, your validation 1-Year AUC should be at least 0.87.

### 2.3: Compare to LungRads criteria and simulate possible improvements to clinical workflow (10 pts)

Now that you've developed a lung cancer detection model, it's time to analyze its clinical implications. In this exercise, you will compare your model to the LungRads criteria (which are loaded in `dataset.py`) and simulate the impact of your model on the clinical workflow. In your report, please introduce a workflow of how your model could be used to ameleorate screening and provide quantitative estimates of your workflow's impact. Be sure to study the impact of your model across various subgroups, as you did in Project 1. Finally, please include a discussion of the limitations of your analyses and subsequent studies are needed to drive these tools to impact.

## Part 3: Extending your LDCT model to predict cancer risk

In this part of the project, you will extend your cancer detection model to predict cancer risk and compare your results to your best model from project 1. 

### Comparing to your best model from Project 1 (note not 100% overlapping questionares) (10pts)
Questionares from NLST are available in `src/dataset.py`. In your project report, please validate your PLCO model (from project 1) on the NLST dataset. Note, some of the information available in PLCO is not available in NLST, so you may need to simplify your project 1 model. Is there a meanigful performance difference between your PLCO model across the PLCO and NLST datasets? If so, why?

### Extend risk model to predict cancer risk (20pts)
In this exercise, you will extend your cancer detection model to predict cancer risk. Specifically, you will predict the probability of a patient being diagnosed with cancer within `[1,2,3,4,5,6]` years of their CT scan. Note, there are many multiple ways to achieve this goal.

In your project report, please include the following:
- Your approach to extending your classifier to predict risk over time
- Detailed performance evaluation of your risk model across multiple time horizons (e.g. 1,3 and 6 years)
- Comparison of your imaging-based risk model against your PLCO model
- An approach for combining your imaging-based risk model with the clinical information in your PLCO model
  - Note, there are many valid options here!
  - Performance evaluation of this combined model across multiple time horizons.

Your image-based 6-year validation AUC should be at least 0.76 and your 1-year Validation should at least as good as your best detection model.

### Explore clinical implications of this model (10pts)

Now that you've developed your risk model, it's time to analyze the clinical opportunities in enables. Please propose a workflow for how your model could be used to improve screening and quantify the potential impact of your workflow. Be sure to study the impact of your model across various subgroups, as you did in Project 1.  Finally, please include a discussion of the limitations of your analyses and subsequent studies are needed to drive these tools to impact.
