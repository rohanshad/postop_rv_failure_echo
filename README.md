Python scripts used to train, run and visualize the ensembled models for prediction of postop RV failure in LVAD patients from pre-op echocardiograms. See paper for more details. Created by Nicolas Quach and Rohan Shad, MD Dec 2020. The model consists of a two stream late-fusion 3D residual neural network. This implementation in Python is intended for use on GPU, and certain parts of the script can only be run on Nvidia GPU containing devices. Multithreading is implemented in several scripts for acceleration. Currently tested on Python 3.7.7, Tensorflow-gpu 2.1.0, Keras 2.3.1.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5161867.svg)](https://doi.org/10.5281/zenodo.5161867)

## Installation

The requirements of the various scripts is very complex and is managed using 3 different Anaconda virtual environments. These instructions assume you have Git and Anaconda installed. If not, install them before proceeding.

### PART I: tf_gpu

The `tf_gpu` environment runs most of the scripts. This bulk of this virtual environment contains Tensorflow, Keras, and densetrack (the package for improved dense trajectory calculation).

Create a new Conda env that has Tensorflow GPU (run and tested on v2.1.0):
```bash
$ conda create --name tf_gpu tensorflow-gpu
```

Accept all packages and install. This will take some time. Activate the environment with:
```bash
$ source activate tf_gpu
```
Now, install the rest of the package dependencies for the environment:
```bash
$ pip install -r tf_gpu_requirements.txt
```
The `tf_gpu` virtual environment is now setup. Deactivate the environment to proceed to the next step:
```bash
$ source deactivate tf_gpu
```
### PART II: innvestigate

This environment is to run the [innvestigate](https://github.com/albermax/innvestigate) package to create Layer-wise Relevance Propagation heatmaps. A separate environment is required since the package only runs on Tensorflow 1.12.0. 

Create a new Conda env with the appropriate tensorflow-gpu version:
```bash
$ conda create --name innvestigate tensorflow-gpu==1.12.0
```

Accept all packages and install. Activate the environment:
```bash
$ source activate innvestigate
```

Install the rest of the dependencies:
```bash
$ pip install -r innvestigate_requirements.txt
```

All the environments are now setup.

## Workflow

The following section provides guidance for the workflow of training and running models. 

### Directory setup

Within a single directory, place all DICOM files for a particular dataset. These files should be anonymized, and their DICOM metadata PatientID tag should be converted to REDcap ID number or some other identifying number. Within this same directory, place a REDcap_class.csv file, which has one column with REDcap ID number and another column with corresponding RV failure class, written as lowercase strings. You will also want to have a csv file containing REDcap ID and a set allocation ('train', 'val', 'test') -> ie one column with REDcap ID and another column with its set allocation.

### Prepare data

1. Preprocess: First step is to sort the echo files by class, preprocess them by normalizing (and potentially resizing). 
```
$ source activate tf_gpu
$ python preprocess.py --root=<full path to direc with DICOMs> --csv=<csv file mapping REDCap to RVF class> {-m optional flag to enable multithreading, -r flag for forced recalculation}
```

2. Next calculate optical flow. This does not utilize GPU acceleration, but is written for multithreading if enabled.
```
$ python optical_flow_calculator.py --root=<full path to current working direc> {-m optional flag to enable multithreading}
```

3. Next, split the data into train, validation, and test sets based off a csv in the current working directory 
```
$ python split_data.py --root=<full path to direc with DICOMs> --splitnum=<give your split a number in case you have multiple splits> {-m optional flag to enable multithreading}
```

### Pretraining Resnet152 on Echonet
The Echonet dataset was downloaded and optical flow was calculated using the above scripts. We pre-trained on kinetics-600 using a similar method. You will need to pretrain a greyscale and an optical flow Resnet152:
```
$ source activate tf_gpu
$ python pretrain_greyscale.py --root=<full path to direc w/ Echonet data> --ckptdir=<fullpath to checkpoint direc> {-r optional flag to for training from scratch}
$ python pretrain_OF.py --root=<full path to direc with DICOMs> --ckptdir=<fullpath to checkpoint direc> {-r optional flag to for training from scratch}
```

### Training a two stream model
Train a two-stream model from the pretrained Resnet152 models is accomplished with `train_twostream.py`. Change the training parameters by changing the globals within the script.
```
$ source activate tf_gpu
$ python train_twostream.py --root=<full path to current working direc> --grey_weights=<full path to pretrained greyscale weights> --OF_weights=<full path to pretrained OF weights> --ckptdir=<path to direc where ckpt is saved> --splitnum=<the split number> --model_prefix=<give the model a name> {-r optional flag to for training from scratch}
```

### Predicting RV Failure class on test set ECHOs using a fully trained model
Once a model has been trained, load the weights back in and use it to predict on videos. Predicted RV failure probabilities are estimated by averaging the probabilities of several randomly sampled 32 frame clips.
```
$ source activate tf_gpu
$ python predict.py --root=<full path to current working direc> --weights=<full path to model weights> --splitnum=<split number for the test set you want>
```

### Producing Guided Backpropagation visualizations
Because models utilize residual blocks in its architecture which contain skip connections, Layerwise Relevance Propagation cannot be used for CNN visualization, and instead we must rely on gradient based methods like guided backpropagation.
```
$ source activate innvestigate
$ python visualize_GBP.py --root=<full path to current working direc> --weights=<full path to model weights> --splitnum=<split number for the test set you're pulling from> --nsamples=<number of videos to examine>
```

