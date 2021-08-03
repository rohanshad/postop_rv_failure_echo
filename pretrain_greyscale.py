'''
Script: pretrain_greyscale.py

Script to pretrain a ResNet152 network on greyscale videos derived from the echo dataset from Oayang et al.
Trains a network to predict ejection fraction based off of a video clip.

Assumes a filestructure:
Echonet_dataset/
├── filename2class.pkl (generated dict of patient ID->ejection fraction)
├── train 
│   └── file1.avi etc...
├── val
│   └── file2.avi etc ... 
└── test
	└── file3.avi etc ...
	
'''

import numpy as np
import os
import pickle as pkl
import random
from shutil import rmtree
from model_zoo import *
from keras.callbacks.callbacks import CSVLogger
import argparse as ap
from multiprocessing import cpu_count
import random
from keras.utils import Sequence
from vidaug import augmentors as va

#######GLOBALS########
#THESE PARAMETERS ARE FOR THE NETWORK ARCHITECTURE AND TRAINING LOOP
NFRAMES = 32 #number of frames per clip
NROWS = 112 #pixels in row of frame
NCOLS = 112 #pixels in col of frame
NCHANNEL = 1 #number of channels (should be 1, hold over from the optical flow days)
NCLASS = 1 #specific 2 class or 4 class prediction. 
BATCH_SIZE = 8 #specify batch size. 8 seems to be the biggest you can get without OOM errors
EPOCH = 100 #number of epochs to train on
THRESHOLD = 2 #Threshold for binarizing class (1=normal/mild:moderate/severe, 2=normal/mild/moderate:severe)

#THESE PARAMETERS ARE FOR TRAINING AND CHECKPOINTING
MODEL_PREFIX = 'Resnet152_Echonet_norm_greyscale_Oct15_' #this will determine checkpoint file name
CKPT_MONITOR = 'val_loss' #tell train loop what to monitor at the end of each epoch
CKPT_MODE = 'min' #minimize or maximize the ckpt objective
COSINE_ANNEAL = True #Toggles cosine annealing on
LR = 1e-3 #Learning rate. 1e-5 seems to work well
LR_WEIGHT = {'conv3d':1, 'fc_6':1, 'fc_7':1} #LR weights for particular layers. Currently set to no weight bias

'''
Class DataGeneratorEchonet
Inherits keras Sequence object
Creates a custom generator that pulls random echo files, 
subsamples a 32 frame clip and assembles a batch for training.
'''
class DataGeneratorEchonet(Sequence):
	'''
	Constructor
	Inputs:
	* train_direc=full path to direc with training movie pkl files
	* classmap_path=full path to the stored REDCap-to-RVF class dict.
	* batch_size = size of batches to generate
	'''
	def __init__(self, train_direc,classmap_path, batch_size=2):
		'Initialization'
		self.batch_size = batch_size
		class_dict = pkl.load(open(classmap_path, 'rb'))
		#print(class_dict)
		movie_list = []
		filenames = os.listdir(train_direc)
		class_list = []
		self.filenames = []
		for filename in filenames:
			if filename == 'filename2class.pkl':
				continue
			if filename[-3:] == 'pkl':
				redcap = filename[:-4]
				if redcap not in class_dict.keys():
					continue
				self.filenames.append(filename)
				movie_list.append(os.path.join(train_direc, filename))
				class_list.append(class_dict[redcap])

		self.labels = class_list
		self.movie_list = movie_list
		self.n_channels = NCHANNEL
		self.n_classes = NCLASS
		self.n = 0
		self.max = self.__len__()

	'''
	Helper function __len__
	Calculates maximum number of batches that can be generated given the training set
	'''
	def __len__(self):
		return int(np.ceil(len(self.movie_list)/float(self.batch_size)))

	'''
	Helper function __getitem___
	Input:
	* index=batch index
	Output:
	* X = array of size (BATCH_SIZE, NFRAMES, NROWS, NCOLS, NCHANNELS)
	* y = array of labels for each movie clip, turned into one hot vectors
	'''
	def __getitem__(self, index, debug=False):
		'Generate one batch of data'
		X = []
		y = []
		movies = self.movie_list[index*self.batch_size:(index+1)*self.batch_size]
		labels = self.labels[index*self.batch_size:(index+1)*self.batch_size]
		

		for movie, label in zip(movies,labels):
			movie = pkl.load(open(movie, 'rb'))
			nframes = movie.shape[0]
			assert(nframes >= NFRAMES)
			start_frame = random.randint(0, nframes - NFRAMES)
			subset = movie[start_frame:start_frame+NFRAMES,:,:]
			subset = np.reshape(subset, (NFRAMES, NROWS, NCOLS, NCHANNEL))
			X.append(subset)
			y.append(label)

		if debug:
			return np.asarray(X), np.asarray(y), movies, labels
		return np.asarray(X), np.asarray(y)
	'''
	Generator function __next__
	Actual function that you call to get another batch of data.
	Once the index self.n exceeds self.max, restarts at the start of the training set.
	'''
	def __next__(self):
		if self.n >= self.max:
			self.n = 0
		result = self.__getitem__(self.n)
		self.n += 1
		return result


if __name__ == '__main__':
	#Read in script arguments
	parser = ap.ArgumentParser()
	parser.add_argument("-d", "--root", required=True, help='full path to directory where DICOMS are initially stored')
	parser.add_argument('-c', '--ckptdir', required=True, help='full path to directory where checkpoint will be stored')
	parser.add_argument('-r', '--recalculate', action='store_true', help='Flag if you want to train from scratch')
	args = vars(parser.parse_args())

	print(args)
	root_direc = args['root']
	ckpt_direc = args['ckptdir']
	recalculate = args['recalculate']

	#Define path to pickled map of filename to ejection fraction
	class_path = os.path.join(root_direc, 'filename2class.pkl') #path to classmap.pkl, which contains dict that maps REDCap to RVF class

	model_name = MODEL_PREFIX + '.hdf5'
	train_direc = os.path.join(root_direc, 'train_norm1') #define where training movies are
	test_direc = os.path.join(root_direc, 'val_norm1') #define where testing movies are
	
	ckpt_name = model_name
	ckpt_path = os.path.join(ckpt_direc, ckpt_name)

	#Make a folder to store the training history in the folder where checkpoint files are stored
	csv_direc = os.path.join(ckpt_direc, 'model_csv') 
	safe_makedir(csv_direc)
	csv_name = model_name + '.csv'
	csv_path = os.path.join(csv_direc, csv_name)
	
	#Define the pickle file that will save which epoch the model has just completed
	init_epoch_path = os.path.join(ckpt_direc, model_name + '_epoch.pkl') #this file stores the current epoch of the model. Helpful if training is interrupted

	num_movies = num_movies(train_direc)
	batch_size = BATCH_SIZE
	num_epochs = EPOCH
	lr = LR

	#Figures out the starting epoch. Default is 0
	init_epoch = 0

	if not recalculate:
		#if init_epoch_path pkl file exists, loads it so model can start training at this epoch number
		if os.path.exists(init_epoch_path):
			init_epoch = int(pkl.load(open(init_epoch_path, 'rb'))) 

	print('Initial epoch: ', init_epoch)

	if init_epoch == (num_epochs-1):
		print('Already trained to max epochs of ' + str(num_epochs) + '!')
		exit()

	#Need to figure out how many training iterations is equivalent to 1 epoch
	steps_per_epoch = num_movies//batch_size

	#Create generators
	print('Creating generators...')
	gen = DataGeneratorEchonet(train_direc=train_direc, batch_size=batch_size,classmap_path=class_path) #create training data generator
	test_gen = DataGeneratorEchonet(train_direc=test_direc, batch_size=batch_size, classmap_path=class_path) #create testing data generator for validation
	print('Done!')
	print('Steps per epoch: ', steps_per_epoch)

	print('Initializing network...')
	input_size = (NFRAMES, NROWS, NCOLS, NCHANNEL)

	#load any existing checkpoint
	weights = None
	if not recalculate:
		if os.path.exists(ckpt_path):
			print('Loaded checkpoint ' + ckpt_path + '!')
			weights = ckpt_path
		else:
			print('No checkpoint found, training from scratch!')
	else:
		print('RECALCULATE flag is TRUE, training from scratch!')

	#Make a model
	model = Resnet152_AdamW_regression(pretrained_weights = weights, input_size=input_size, lr=LR, cosine_anneal=COSINE_ANNEAL, tot_iter=steps_per_epoch*num_epochs, lr_weight=LR_WEIGHT)
	
	print('Making callbacks...')
	#make checkpoint logger. Monitors training loss and saves checkpoint if loss improves at end of epoch. 
	model_checkpoint = ModelCheckpoint(ckpt_path, monitor=CKPT_MONITOR,verbose=1, save_best_only=True, mode=CKPT_MODE) 
	#Keras builtin logger that outputs model training parameters to csv file 
	csv_logger = CSVLogger(csv_path, append=True)
	print('Done!')

	#Determine number of CPUs available for input/output operations
	num_cpu = cpu_count()
	print('Number of CPUs available for data processing: ', num_cpu)
	print('Fitting model...')
	
	#Fit model, using all available CPUs to power the data generator.
	#Custom training loop
	best_loss = 1000000
	val_losses = []
	for i in range(init_epoch, num_epochs):
		print('Epoch: ' + str(i) + '/' + str(num_epochs))
		history = model.fit_generator(generator=gen, epochs=1, callbacks=[csv_logger], use_multiprocessing=True, workers=num_cpu)
		val_loss = model.evaluate(test_gen, use_multiprocessing=True, workers=num_cpu)
		val_losses.append(val_loss)
		if val_loss < best_loss:
			print('Val loss decreased! Best val loss now:', val_loss)
			model.save_weights(ckpt_path)
			best_loss = val_loss
		else:
			print('Val loss did not decrease: ', val_loss)
		pkl.dump(i, open(init_epoch_path, 'wb'))

	np.savetxt(os.path.join(csv_direc, model_name + '_val.csv'), np.asarray(val_losses))
	



