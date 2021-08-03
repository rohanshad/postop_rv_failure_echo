'''
visualize_guidedbackprop.py
Stand alone script for producing Guided Backpropagation visualizations of 
the 3D conv-net
Dependency environment: innvestigate

Directory tree structure:
root--GBP_visualizations--GBP_movies--RVF1--GBP_frame1.png etc...
						|
						--GBP_proj--RVF1.png, normal1.png etc...
'''
import numpy as np
import os
import pydicom as dcm
import pickle as pkl
from tqdm import tqdm
import random
import skimage.transform as trans
from shutil import rmtree
import datetime
import sys
import innvestigate as inv 
import innvestigate.utils
from skimage.io import imsave
import matplotlib.pyplot as plt
from model_zoo import *
from utils import *
from resnet_builder_keras import Resnet3DBuilder
import argparse as ap

####GLOBALS####
NFRAMES = 32
NROWS = 112
NCOLS = 112
NCHANNEL = 1
NCLASS = 2
THRESHOLD = 2
'''
Class DataGenerator
Defines a custom generator Class to read in processed movies (pkl files) in the training 
set (stored in train_direc), and output batches of size batch_size for the neural net to train.
Attributes:
* self.batch_size = batch size 
* self.filenames = list of processed movie files in the training direc
* self.labels = RVF classes for each movie in the training set
* self.n_channels = number of channels per frame (should be 1 if greyscale)
* self.n_classes = specify number of output classes (should be 2 if binary classification)
* self.n = internal counter variable to keep track of how many batches have been generated in the current epoch
* self.max = maximum number of batches that can be generated given the training set. Automatically calculated
'''
class DataGeneratorMerged(Sequence):
	'''
	Constructor
	Inputs:
	* train_direc=full path to direc with training movie pkl files
	* classmap_path=full path to the stored REDCap-to-RVF class dict.
	* batch_size = size of batches to generate
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
			if filename == 'filename2class.pkl':
				continue
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
	def __getitem__(self, index, debug=False):
		'Generate one batch of data'
		X_grey = []
		X_OF = []
		y = []
		grey_movies = self.grey_movie_list[index*self.batch_size:(index+1)*self.batch_size]
		OF_movies = self.OF_movie_list[index*self.batch_size:(index+1)*self.batch_size]
		
		labels = self.labels[index*self.batch_size:(index+1)*self.batch_size]
		labels = [(x > self.threshold) for x in labels]
		

		for grey_movie, OF_movie, label in zip(grey_movies, OF_movies, labels):
			print(grey_movie)
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



if __name__ == '__main__':
	parser = ap.ArgumentParser()
	parser.add_argument("-d", "--root", required=True, help='full path to directory where DICOMS are initially stored')
	parser.add_argument("-w", "--weights", required=True, help='full path to TwoStream model saved weights')
	parser.add_argument('-s', '--splitnum', type=int, required=True, help='integer for which split you want to predict on.')
	parser.add_argument('-n', '--nsamples', type=int, required=True, help='number of videos to produce')

	args = vars(parser.parse_args())

	print(args)
	root_direc = args['root']
	ckpt_path = args['weights']
	split_num = args['splitnum']
	num_samples = args['nsamples']

	grey_direc = os.path.join(root_direc, 'greyscale')
	OF_direc = os.path.join(root_direc, 'OF')

	grey_test_direc = os.path.join(grey_direc, 'split' + str(split_num) + '_test')
	OF_test_direc = os.path.join(OF_direc, 'split' + str(split_num) + '_test')
	
	class_path = os.path.join(grey_direc, 'classmap.pkl') #path to where classmap.pkl is sorted (the file with REDCap to RVF class pickled map)
	
	batch_size = 1
	num_epochs = 16
	lr = 1e-5

	print('Creating generators...')
	gen = DataGeneratorMerged(grey_train_direc=grey_test_direc, OF_train_direc=OF_test_direc, batch_size=batch_size,classmap_path=class_path)
	
	print('Initializing network...')
	
	weights = None
	if os.path.exists(ckpt_path):
		print('Loaded checkpoint ' + ckpt_path + '!')
		weights = ckpt_path
	else:
		print('No checkpoint found, training from scratch!')

	#Just to keep track of how many RVF echos and how many normal echos have been seen
	true_counter = 0
	false_counter = 0

	heatmap_direc = os.path.join(root_direc, 'guidedbackprop_visualization') #this is where the GBP visualizations will go
	if os.path.exists(heatmap_direc):
		rmtree(heatmap_direc)
	safe_makedir(heatmap_direc)

	premodel = two_stream(saved_weights=weights, nframes=32, nrows=112, ncols=112, lr=1e-4)

	fc = premodel.layers[-2].output 
	final_out = Dense(2, activation='softmax', name='preds')(fc)

	model = Model(inputs=premodel.input, outputs=final_out)
	model = inv.utils.model_wo_softmax(model) #remove classification layer (needed for GBP)

	analyzer = inv.analyzer.gradient_based.GuidedBackprop(model) #Create analyzer object with GBP preset for pretty pictures
	GBP_movie_direc = os.path.join(heatmap_direc, 'movies')
	GBP_proj_direc = os.path.join(heatmap_direc, 'proj')
	if os.path.exists(GBP_movie_direc):
		rmtree(GBP_movie_direc)
	if os.path.exists(GBP_proj_direc):
		rmtree(GBP_proj_direc)
	safe_makedir(GBP_movie_direc)
	safe_makedir(GBP_proj_direc)

	for s in range(num_samples):
		X_batch, y_batch = next(gen) #generate a batch of data
		
		movie = X_batch

		label = None
		counter = None

		#this part keeps track of how many RVF and how many Normal movies have been seen, to help with file numbering
		if y_batch:
			label = 'RVF'
			counter = true_counter
			true_counter += 1
		else:
			label = 'normal'
			counter = false_counter
			false_counter += 1

		print(label + '_' + str(counter))

		a = analyzer.analyze(movie)
		
		print(a[0].shape, a[1].shape)
		concat = np.hstack((np.squeeze(a[0]), np.squeeze(a[1][:,:,:,:,0]), np.squeeze(a[1][:,:,:,:,1])))
		print('output shape:', concat.shape)
		
		arr = concat

		#Resize the GBP images here
		resized_arr = []
		for i in range(arr.shape[0]):
			image = arr[i,:,:]
			image = trans.resize(np.squeeze(image), (384*3,384))
			resized_arr.append(image)

		arr = np.stack(resized_arr)

		#calculate the Min-Max projection
		arr /= np.max(np.abs(arr))
		arr -= np.mean(arr)
		neg_vals = np.where(arr < 0, arr, 0)
		pos_vals = np.where(arr >= 0, arr, 0)

		minIP = neg_vals.min(axis=0)
		maxIP = pos_vals.max(axis=0)
		final_proj = minIP + maxIP 
		final_proj = np.squeeze(final_proj)

		avg_proj = np.mean(arr, axis=0)
		avg_proj = np.squeeze(avg_proj)
		
		filename = label + '_' + str(counter) + '.png'
		filepath = os.path.join(GBP_proj_direc, filename)
		plt.imsave(filepath, final_proj, cmap='seismic', vmin=-1, vmax=1)

		movie_direc = os.path.join(GBP_movie_direc, label + '_' + str(counter))
		safe_makedir(movie_direc)

		#save the individual frames of the GBP movie
		for i in range(arr.shape[0]):
			image = arr[i,:,:]
			image = trans.resize(np.squeeze(image), (384*3,384))
			filename = 'GBP_frame' + str(i) + '.png'
			filepath = os.path.join(movie_direc, filename)
			plt.imsave(filepath, image, cmap='seismic', vmin=-1, vmax=1)
		
		
	





		

