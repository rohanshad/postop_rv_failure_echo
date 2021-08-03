'''
Script: train_twostream.py
Dependency environment: tf_gpu

Script to train a TwoStream model using two pretrained Resnet152 models trained on the EchoNet dataset
Has video augmentation on the fly.

Assumes a filestructure:
dataset/
├── greyscale 
│   ├── classmap.pkl
│	├── split0_train
│  	│		└── stanford0002a.pkl etc ...
│	├── split0_val
│	│  		└── houston0110k.pkl etc ...
│	└── split0_test
│	  		└── spectrum0234a.pkl etc ...
└── OF 
    ├── classmap.pkl
	├── split0_train
  	│		└── stanford0005a.pkl etc ...
	├── split0_val
	│  		└── houston0116k.pkl etc ...
	└── split0_test
	  		└── spectrum0274a.pkl etc ...			
.
'''
import numpy as np
import os
import pickle as pkl
import random
from shutil import rmtree
from model_zoo_twostream import *
from keras.callbacks.callbacks import EarlyStopping, LambdaCallback, ReduceLROnPlateau
import argparse as ap
from multiprocessing import cpu_count
import random
from CLR_callback import CyclicLR
from vidaug import augmentors as va
from keras.utils import Sequence

#Set up video augmentation pipeline
sometimes = lambda aug: va.Sometimes(0.5, aug) # Used to apply augmentor with 50% probability
seq = va.Sequential([sometimes(va.RandomRotate(degrees=30)), sometimes(va.RandomTranslate(x=10, y=10)), sometimes(va.Sequential([va.RandomResize(rate=0.2), va.RandomCrop(size=(112,112))]))])

#######GLOBALS########
#THESE PARAMETERS ARE FOR THE NETWORK ARCHITECTURE AND TRAINING LOOP
NFRAMES = 32 #number of frames per clip
NROWS = 112 #pixels in row of frame
NCOLS = 112 #pixels in col of frame
NCHANNEL = 2 #number of channels (should be 1, hold over from the optical flow days)
NCLASS = 2 #specific 2 class or 4 class prediction. 
BATCH_SIZE = 4 #specify batch size. 8 seems to be the biggest you can get without OOM errors
EPOCH = 75 #number of epochs to train on
THRESHOLD = 2 #Threshold for binarizing class (1=normal/mild:moderate/severe, 2=normal/mild/moderate:severe)

#THESE PARAMETERS ARE FOR TRAINING AND CHECKPOINTING
MODEL_PREFIX = '_Transfer_TwoStream_AdamW_aug_Dec13_classratio18_split'
CKPT_MONITOR = 'val_loss' #determines the ckpt objective
CKPT_MODE = 'min' #determines whether to minimize or maximize the ckpt objective
USE_AUG = True #Toggles use of on-the-fly video augmentation
COSINE_ANNEAL = True #Toggles use of cosine annealing of learning rate
LR = 1e-4 #Learning rate. 1e-5 seems to work well
USE_CLR = False #Toggles use of Cyclic learning rate
MIN_LR = 1e-5 #min LR for cyclic LR
MAX_LR = 1e-3 #max LR for cyclic LR
CLR_MODE = 'triangular' #mode for cyclic LR
LR_WEIGHT = {'conv3d':1} #LR weights for AdamW
REDUCE_LR = False #Toggles use of ReduceLRonPlateau callback
MONITOR_PARAM = 'val_auc_1' #determines objective for ReduceLRonPlateau callback
PATIENCE = 2 #determines patience parameter of ReduceLRonPlateau callback

print('SETTINGS:')
print('CKPT_MONITOR=', CKPT_MONITOR)
print('USE_AUG=', USE_AUG)
print('COSINE_ANNEAL=', COSINE_ANNEAL)
print('LR=', LR)
print('USE_CLR=', USE_CLR)
print('MIN_LR=', MIN_LR)
print('MAX_LR=', MAX_LR)
print('CLR_MODE=', CLR_MODE)
print('LR_WEIGHT=', LR_WEIGHT)
print('REDUCE_LR=', REDUCE_LR)
print('MONITOR_PARAM=', MONITOR_PARAM)
print('PATIENCE=', PATIENCE)

'''
Helper function: safe_makedir
Given path, will make all necessary folders to create path to folder

Input:
path = full path to direc you wish to create

Output:
None
'''
def safe_makedir(path):
	if not os.path.exists(path):
		os.makedirs(path)

'''
Helper function: num_movies
Given a full path to directory, count the number of pickle files in direc

Input:
train_direc = full path to direc of interest

Output:
int = number of pickle files in direc of interest
'''
def num_movies(train_direc):
	movie_list = []
	filenames = os.listdir(train_direc)

	for filename in filenames:
		if filename[-3:] == 'pkl':
			movie_list.append(os.path.join(train_direc, filename))
	return len(movie_list)

'''
Class DataGeneratorMerged
Inherits keras Sequence object
Creates a custom generator that pulls random greyscale and OF files 
subsamples the same 32 frame clip in both greyscale and OF channels, and assembles a batch for training.
-> NO VIDEO AUGMENTATION
'''
class DataGeneratorMerged(Sequence):
	'''
	Constructor
	Inputs:
	* grey_train_direc=full path to direc with greyscale training movie pkl files
	* grey_train_direc=full path to direc with OF training movie pkl files
	* classmap_path=full path to the stored REDCap-to-RVF class dict.
	* batch_size = size of batches to generate
	* threshold = int for thresholding what counts as RVF (assumes 4 class. default cutoff at severe ie 2)
	'''
	def __init__(self, grey_train_direc, OF_train_direc, classmap_path, batch_size=2, threshold=2):
		'Initialization'
		self.batch_size = batch_size
		class_dict = pkl.load(open(classmap_path, 'rb'))
		#print(class_dict)
		grey_movie_list = []
		OF_movie_list = []
		filenames = os.listdir(grey_train_direc)
		class_list = []
		self.filenames = []
		for filename in filenames:
			if filename[-3:] == 'pkl':
				redcap = filename[:-4]
				if redcap not in class_dict.keys():
					continue
				self.filenames.append(filename)
				grey_movie_list.append(os.path.join(grey_train_direc, filename))
				OF_movie_list.append(os.path.join(OF_train_direc, filename))
				class_list.append(class_dict[redcap])

		self.labels = class_list
		self.grey_movie_list = grey_movie_list
		self.OF_movie_list = OF_movie_list
		self.n = 0
		self.max = self.__len__()
		self.threshold=threshold

	'''
	Helper function __len__
	Calculates maximum number of batches that can be generated given the training set
	'''
	def __len__(self):
		return int(np.ceil(len(self.grey_movie_list)/float(self.batch_size)))

	'''
	Helper function __getitem___
	Input:
	* index=batch index
	Output:
	* X = array of size (BATCH_SIZE, NFRAMES, NROWS, NCOLS, NCHANNELS)
	* y = array of labels for each movie clip, turned into one hot vectors
	'''
	def __getitem__(self, index):
		'Generate one batch of data'
		X_grey = []
		X_OF = []
		y = []
		grey_movies = self.grey_movie_list[index*self.batch_size:(index+1)*self.batch_size]
		OF_movies = self.OF_movie_list[index*self.batch_size:(index+1)*self.batch_size]
		
		labels = self.labels[index*self.batch_size:(index+1)*self.batch_size]
		labels = [(x > self.threshold) for x in labels]
		

		for grey_movie, OF_movie, label in zip(grey_movies, OF_movies, labels):
			grey_movie = pkl.load(open(grey_movie, 'rb'))
			OF_movie = pkl.load(open(OF_movie, 'rb'))
			nframes = grey_movie.shape[0]
			assert(nframes >= NFRAMES)
			start_frame = random.randint(0, nframes - NFRAMES)

			grey_subset = grey_movie[start_frame:start_frame+NFRAMES,:,:]
			OF_subset = OF_movie[start_frame:start_frame+NFRAMES,:,:,:]
			grey_subset = np.reshape(grey_subset, (NFRAMES, NROWS, NCOLS, 1))
			OF_subset = np.reshape(OF_subset, (NFRAMES, NROWS, NCOLS, 2))
			
			X_grey.append(grey_subset)
			X_OF.append(OF_subset)
			
			y.append(label)

		return [np.asarray(X_grey), np.asarray(X_OF)], np.asarray(y)
	'''
	Generator function __next__
	Actual function that you call to get another batch of data.
	Once the index self.n exceeds self.max, restarts at the start of the training set.
	'''
	def __next__(self):
		if self.n >= self.max:
			self.n = 0
		else:
			result = self.__getitem__(self.n)
		self.n += 1
		return result

'''
Class DataGeneratorMerged_aug
Inherits keras Sequence object
Creates a custom generator that pulls random greyscale and OF files 
subsamples the same 32 frame clip in both greyscale and OF channels, and assembles a batch for training.
-> WITH VIDEO AUGMENTATION
'''
class DataGeneratorMerged_aug(Sequence):
	'''
	Constructor
	Inputs:
	* train_direc=full path to direc with training movie pkl files
	* classmap_path=full path to the stored REDCap-to-RVF class dict.
	* batch_size = size of batches to generate
	'''
	def __init__(self, grey_train_direc, OF_train_direc, classmap_path, augmentor, batch_size=2, threshold=2):
		'Initialization'
		self.batch_size = batch_size
		class_dict = pkl.load(open(classmap_path, 'rb'))
		#print(class_dict)
		grey_movie_list = []
		OF_movie_list = []
		filenames = os.listdir(grey_train_direc)
		class_list = []
		self.filenames = []
		for filename in filenames:
			if filename[-3:] == 'pkl':
				redcap = filename[:-4]
				if redcap not in class_dict.keys():
					continue
				self.filenames.append(filename)
				grey_movie_list.append(os.path.join(grey_train_direc, filename))
				OF_movie_list.append(os.path.join(OF_train_direc, filename))
				class_list.append(class_dict[redcap])

		self.labels = class_list
		self.grey_movie_list = grey_movie_list
		self.OF_movie_list = OF_movie_list
		self.n = 0
		self.max = self.__len__()
		self.threshold=threshold
		self.seq = augmentor

	'''
	Helper function __len__
	Calculates maximum number of batches that can be generated given the training set
	'''
	def __len__(self):
		return int(np.ceil(len(self.grey_movie_list)/float(self.batch_size)))

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
		X_grey = []
		X_OF = []
		y = []
		grey_movies = self.grey_movie_list[index*self.batch_size:(index+1)*self.batch_size]
		OF_movies = self.OF_movie_list[index*self.batch_size:(index+1)*self.batch_size]
		
		labels = self.labels[index*self.batch_size:(index+1)*self.batch_size]
		labels = [(x > self.threshold) for x in labels]

		for grey_movie, OF_movie,  label in zip(grey_movies, OF_movies, labels):
			grey_movie = pkl.load(open(grey_movie, 'rb'))
			OF_movie = pkl.load(open(OF_movie, 'rb'))
			nframes = grey_movie.shape[0]
			assert(nframes >= NFRAMES)
			start_frame = random.randint(0, nframes - NFRAMES)

			grey_subset = grey_movie[start_frame:start_frame+NFRAMES,:,:]
			OF_subset = OF_movie[start_frame:start_frame+NFRAMES,:,:,:]
			#AUGMENT
			stacked = np.stack((grey_subset[:,:,:,0], OF_subset[:,:,:,0], OF_subset[:,:,:,1]), axis=-1)
			stacked = np.asarray(self.seq(stacked))
			#DECOUPLE
			aug_grey_subset = np.reshape(stacked[:,:,:,0], (NFRAMES, NROWS, NCOLS, 1))
			aug_OF_subset = stacked[:,:,:,1:]

			X_grey.append(aug_grey_subset)
			X_OF.append(aug_OF_subset)
			
			y.append(label)

		return [np.asarray(X_grey), np.asarray(X_OF)], np.asarray(y)
	'''
	Generator function __next__
	Actual function that you call to get another batch of data.
	Once the index self.n exceeds self.max, restarts at the start of the training set.
	'''
	def __next__(self):
		if self.n >= self.max:
			self.n = 0
		else:
			result = self.__getitem__(self.n)
		self.n += 1
		return result



if __name__ == '__main__':
	parser = ap.ArgumentParser()
	parser.add_argument("-d", "--root", required=True, help='full path to directory where DICOMS are initially stored')
	parser.add_argument("-g", "--grey_weights", required=True, help='full path to pretrained Resnet152 weights for greyscale video')
	parser.add_argument("-o", "--OF_weights", required=True, help='full path to pretrained Resnet152 weights for OF video')
	parser.add_argument('-c', '--ckptdir', required=True, help='full path to directory where checkpoint will be stored')
	parser.add_argument('-s', '--splitnum', type=int, required=True, help='integer for which split you want to predict on.')
	parser.add_argument("-n", "--model_prefix", required=True, help='Name of model that will be incorporated into ckpt filename')
	parser.add_argument('-r', '--recalculate', action='store_true', help='Flag if you want to train from scratch')
	args = vars(parser.parse_args())

	print(args)
	root_direc = args['root']
	ckpt_direc = args['ckptdir']
	recalculate = args['recalculate']
	split_num = args['splitnum']
	grey_transfer_path = args['grey_weights']
	OF_transfer_path = args['OF_weights']
	model_prefix = args['model_prefix']

	grey_direc = os.path.join(root_direc, 'greyscale')
	OF_direc = os.path.join(root_direc, 'OF')

	class_path = os.path.join(root_direc, 'classmap.pkl') #path to classmap.pkl, which contains dict that maps REDCap to RVF class

	model_name = model_prefix + str(split_num) + '.hdf5'

	grey_train_direc = os.path.join(grey_direc, 'split' + str(split_num) + '_train') #define where training movies are
	grey_test_direc = os.path.join(grey_direc, 'split' + str(split_num) + '_test' ) #define where testing movies are
	grey_val_direc = os.path.join(grey_direc, 'split' + str(split_num) + '_val' ) 

	OF_train_direc = os.path.join(OF_direc, 'split' + str(split_num) + '_train') #define where training movies are
	OF_test_direc = os.path.join(OF_direc, 'split' + str(split_num) + '_test' ) #define where testing movies are
	OF_val_direc = os.path.join(OF_direc, 'split' + str(split_num) + '_val' ) 

	ckpt_name = model_name
	ckpt_path = os.path.join(ckpt_direc, ckpt_name)
	
	init_epoch_path = os.path.join(ckpt_direc, model_name + '_epoch.pkl') #this file stores the current epoch of the model. Helpful if training is interrupted

	num_movies = num_movies(grey_train_direc)
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

	steps_per_epoch = num_movies//batch_size

	print('Creating generators...')

	gen = DataGeneratorMerged_aug(grey_train_direc=grey_train_direc, OF_train_direc=OF_train_direc, augmentor=seq, batch_size=batch_size,classmap_path=class_path) #create training data generator
	val_gen = DataGeneratorMerged(grey_train_direc=grey_val_direc, OF_train_direc=OF_val_direc, batch_size=batch_size, classmap_path=class_path) #create testing data generator for validation
	test_gen = DataGeneratorMerged(grey_train_direc=grey_test_direc, OF_train_direc=OF_test_direc, batch_size=batch_size, classmap_path=class_path) #create testing data generator for validation
	print('Done!')
	print('Steps per epoch: ', steps_per_epoch)

	print('Initializing network...')

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
	
	if USE_CLR:
		model = two_stream(grey_weights=grey_transfer_path, OF_weights=OF_transfer_path, saved_weights=weights, nframes=NFRAMES, nrows=NROWS, ncols=NCOLS, lr=MIN_LR, cosine_anneal=COSINE_ANNEAL, tot_iter=steps_per_epoch*num_epochs, lr_weight=LR_WEIGHT)
	else:
		model = two_stream(grey_weights=grey_transfer_path, OF_weights=OF_transfer_path, saved_weights=weights, nframes=NFRAMES, nrows=NROWS, ncols=NCOLS, lr=LR, cosine_anneal=COSINE_ANNEAL, tot_iter=steps_per_epoch*num_epochs, lr_weight=LR_WEIGHT)
	print('Making callbacks...')
	#make checkpoint logger. Monitors training loss and saves checkpoint if loss improves at end of epoch. 
	model_checkpoint = ModelCheckpoint(ckpt_path, monitor=CKPT_MONITOR,verbose=1, save_best_only=True, mode=CKPT_MODE) 

	csv_direc = os.path.join(ckpt_direc, 'two_stream_csv') #define where to store csv file with model training history will be stored
	safe_makedir(csv_direc)
	csv_name = model_name + '.csv'
	csv_path = os.path.join(csv_direc, csv_name)

	#Create callback functions
	epoch_logger = LambdaCallback(on_epoch_end=lambda epoch, logs: pkl.dump(epoch, open(init_epoch_path, 'wb')))
	reduce_lr = ReduceLROnPlateau(monitor=MONITOR_PARAM, patience=PATIENCE)
	clr = CyclicLR(base_lr=MIN_LR, max_lr=MAX_LR, step_size=4*steps_per_epoch, mode=CLR_MODE)
	print('Done!')
	num_cpu = cpu_count()
	print('Number of CPUs available for data processing: ', num_cpu)
	print('Fitting model...')
	
	#Fit model, using all available CPUs to power the data generator.
	#Custom training loop. Evaluates val and test at every epoch
	history_vector = []
	best_auc = 0
	saved_test_auc = 0
	for i in range(init_epoch, num_epochs):
		print('Epoch: ' + str(i) + '/' + str(num_epochs))
		if REDUCE_LR:
			if USE_CLR:
				history = model.fit_generator(generator=gen, epochs=1, class_weight={0:0.2, 1:1.8}, callbacks=[epoch_logger, clr, reduce_lr], use_multiprocessing=True, workers=num_cpu)
				train_loss, train_acc, train_auc, train_TP, train_TN, train_FP, train_FN = model.evaluate(gen, use_multiprocessing=True, workers=num_cpu)
				val_loss, val_acc, val_auc, val_TP, val_TN, val_FP, val_FN = model.evaluate(val_gen, use_multiprocessing=True, workers=num_cpu)
				test_loss, test_acc, test_auc, test_TP, test_TN, test_FP, test_FN = model.evaluate(test_gen, use_multiprocessing=True, workers=num_cpu)
				print('Val loss:', val_loss, '\tVal acc:', val_acc, '\tVal AUC:', val_auc, '\tVal TP:', val_TP, '\tVal TN:', val_TN, '\tVal FP:', val_FP, '\tVal FN:', val_FN)
				print('Test loss:', test_loss, '\tTest acc:', test_acc, '\tTest AUC:', test_auc, '\tTest TP:', test_TP, '\tTest TN:', test_TN, '\tTest FP:', test_FP, '\tTest FN:', test_FN)
				print('Best val AUC=', best_auc, '\tCorresponding test AUC=', saved_test_auc)
				history_vector.append([train_loss, val_loss, test_loss, train_acc, val_acc, test_acc, train_auc, val_auc, test_auc, train_TP, val_TP, test_TP, train_TN, val_TN, test_TN, train_FP, val_FP, test_FP, train_FN, val_FN, test_FN])
				if val_auc > best_auc:
					print('Val AUC increased! Best val AUC now:', val_auc)
					model.save_weights(ckpt_path)
					print('Saved weights to', ckpt_path)
					best_auc = val_auc
					saved_test_auc = test_auc
					pkl.dump(i, open(init_epoch_path, 'wb'))
				else:
					pkl.dump(i, open(init_epoch_path, 'wb'))

			else:
				history = model.fit_generator(generator=gen, epochs=1, class_weight={0:0.2, 1:1.8}, callbacks=[epoch_logger, reduce_lr], use_multiprocessing=True, workers=num_cpu)
				train_loss, train_acc, train_auc, train_TP, train_TN, train_FP, train_FN = model.evaluate(gen, use_multiprocessing=True, workers=num_cpu)
				val_loss, val_acc, val_auc, val_TP, val_TN, val_FP, val_FN = model.evaluate(val_gen, use_multiprocessing=True, workers=num_cpu)
				test_loss, test_acc, test_auc, test_TP, test_TN, test_FP, test_FN = model.evaluate(test_gen, use_multiprocessing=True, workers=num_cpu)
				print('Val loss:', val_loss, '\tVal acc:', val_acc, '\tVal AUC:', val_auc, '\tVal TP:', val_TP, '\tVal TN:', val_TN, '\tVal FP:', val_FP, '\tVal FN:', val_FN)
				print('Test loss:', test_loss, '\tTest acc:', test_acc, '\tTest AUC:', test_auc, '\tTest TP:', test_TP, '\tTest TN:', test_TN, '\tTest FP:', test_FP, '\tTest FN:', test_FN)
				print('Best val AUC=', best_auc, '\tCorresponding test AUC=', saved_test_auc)
				history_vector.append([train_loss, val_loss, test_loss, train_acc, val_acc, test_acc, train_auc, val_auc, test_auc, train_TP, val_TP, test_TP, train_TN, val_TN, test_TN, train_FP, val_FP, test_FP, train_FN, val_FN, test_FN])
				if val_auc > best_auc:
					print('Val AUC increased! Best val AUC now:', val_auc)
					model.save_weights(ckpt_path)
					print('Saved weights to', ckpt_path)
					best_auc = val_auc
					saved_test_auc = test_auc
					pkl.dump(i, open(init_epoch_path, 'wb'))
				else:
					pkl.dump(i, open(init_epoch_path, 'wb'))
		else:
			if USE_CLR:
				history = model.fit_generator(generator=gen, epochs=1, class_weight={0:0.2, 1:1.8}, callbacks=[epoch_logger, clr], use_multiprocessing=True, workers=num_cpu)
				train_loss, train_acc, train_auc, train_TP, train_TN, train_FP, train_FN = model.evaluate(gen, use_multiprocessing=True, workers=num_cpu)
				val_loss, val_acc, val_auc, val_TP, val_TN, val_FP, val_FN = model.evaluate(val_gen, use_multiprocessing=True, workers=num_cpu)
				test_loss, test_acc, test_auc, test_TP, test_TN, test_FP, test_FN = model.evaluate(test_gen, use_multiprocessing=True, workers=num_cpu)
				print('Val loss:', val_loss, '\tVal acc:', val_acc, '\tVal AUC:', val_auc, '\tVal TP:', val_TP, '\tVal TN:', val_TN, '\tVal FP:', val_FP, '\tVal FN:', val_FN)
				print('Test loss:', test_loss, '\tTest acc:', test_acc, '\tTest AUC:', test_auc, '\tTest TP:', test_TP, '\tTest TN:', test_TN, '\tTest FP:', test_FP, '\tTest FN:', test_FN)
				print('Best val AUC=', best_auc, '\tCorresponding test AUC=', saved_test_auc)
				history_vector.append([train_loss, val_loss, test_loss, train_acc, val_acc, test_acc, train_auc, val_auc, test_auc, train_TP, val_TP, test_TP, train_TN, val_TN, test_TN, train_FP, val_FP, test_FP, train_FN, val_FN, test_FN])
				if val_auc > best_auc:
					print('Val AUC increased! Best val AUC now:', val_auc)
					model.save_weights(ckpt_path)
					print('Saved weights to', ckpt_path)
					best_auc = val_auc
					saved_test_auc = test_auc
					pkl.dump(i, open(init_epoch_path, 'wb'))
				else:
					pkl.dump(i, open(init_epoch_path, 'wb'))
			else:
				history = model.fit_generator(generator=gen, epochs=1, class_weight={0:0.2, 1:1.8}, callbacks=[epoch_logger], use_multiprocessing=True, workers=num_cpu)
				train_loss, train_acc, train_auc, train_TP, train_TN, train_FP, train_FN = model.evaluate(gen, use_multiprocessing=True, workers=num_cpu)
				val_loss, val_acc, val_auc, val_TP, val_TN, val_FP, val_FN = model.evaluate(val_gen, use_multiprocessing=True, workers=num_cpu)
				test_loss, test_acc, test_auc, test_TP, test_TN, test_FP, test_FN = model.evaluate(test_gen, use_multiprocessing=True, workers=num_cpu)
				print('Val loss:', val_loss, '\tVal acc:', val_acc, '\tVal AUC:', val_auc, '\tVal TP:', val_TP, '\tVal TN:', val_TN, '\tVal FP:', val_FP, '\tVal FN:', val_FN)
				print('Test loss:', test_loss, '\tTest acc:', test_acc, '\tTest AUC:', test_auc, '\tTest TP:', test_TP, '\tTest TN:', test_TN, '\tTest FP:', test_FP, '\tTest FN:', test_FN)
				print('Best val AUC=', best_auc, '\tCorresponding test AUC=', saved_test_auc)
				history_vector.append([train_loss, val_loss, test_loss, train_acc, val_acc, test_acc, train_auc, val_auc, test_auc, train_TP, val_TP, test_TP, train_TN, val_TN, test_TN, train_FP, val_FP, test_FP, train_FN, val_FN, test_FN])
				if val_auc > best_auc:
					print('Val AUC increased! Best val AUC now:', val_auc)
					model.save_weights(ckpt_path)
					print('Saved weights to', ckpt_path)
					best_auc = val_auc
					saved_test_auc = test_auc
					pkl.dump(i, open(init_epoch_path, 'wb'))
				else:
					pkl.dump(i, open(init_epoch_path, 'wb'))
		
	np.savetxt(csv_path, np.asarray(history_vector), delimiter=',')
		




	