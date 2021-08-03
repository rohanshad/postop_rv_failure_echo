'''
Script: predict.py
Dependency environment: tf_gpu

Given trained TwoStream weights, will predict probability of severe RV failure for echos for a particular data split
Outputs data within new folder within the currect working directory. Saves RVF probabilities, filenames, patient identifier, ground truth 
as csv, and pickle.

Note: Script left compatible w/ a 4 class RVF system, but binarized to severe, which is consistent w/ 2020
INTERMACS definition of RV failure.
'''
import os 
import numpy as np
from model_zoo import *
import pickle as pkl
from keras import Model
import random
import time
import argparse as ap

#Threshold for RVF class. Not relevant with new INTERMACS definitions,
#but scripts left compatible w/ a 4 class system (normal, mild moderate severe) 
#binarized using the THRESHOLD global
#threshold=2 means [normal,mild,moderate=class 0, severe=class 1] --> corresponds to 2020 INTERMACS definition
THRESHOLD = 2

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
Function: predict_RVF
Given TwoStream model object, file paths to the video's greyscale and OF normalized files, and input size parameters
will output probability of RV failure, as binarized by THRESHOLD. Output prediction probabilities are the mean of 
raw output probabilities of n_sample number of clips

Input:
model = TwoStream model object
grey_filepath = full path to greyscale normalized video arr pickle file
OF_filepath = full path to OF normalized video arr pickle file
clip_frames = number of frames in clip (default 32)
nrows = pixels in X dim (default 112)
ncols = pixels in Y dim (default 112)
n_sample = number of clips to sample from video to create RVF probability estimate
'''
def predict_RVF(model, grey_filepath, OF_filepath, clip_frames=32, nrows=112, ncols=112, n_sample=5):
	print('Predicting ' + grey_filepath + '...')
	grey_movie = pkl.load(open(grey_filepath, 'rb'))
	OF_movie = pkl.load(open(OF_filepath, 'rb'))
	nframes = grey_movie.shape[0]
	if nframes < clip_frames:
		print('NOT ENOUGH FRAMES')
		return None
	fc6_list = []
	#Subsample the movies, calculate the RVF probability, save in list
	for i in range(n_sample):
		start_frame = random.randint(0, nframes - clip_frames)
		#Subsample movies
		grey_subset = grey_movie[start_frame:start_frame+clip_frames,:,:,:]
		OF_subset = OF_movie[start_frame:start_frame+clip_frames,:,:,:]
		#convert to correct input shape for neural net
		grey_subset = np.reshape(grey_subset, (1, clip_frames, nrows, ncols, 1))
		OF_subset = np.reshape(OF_subset, (1, clip_frames, nrows, ncols, 2))
		#predict RVF for the subsample
		fc6 = model.predict([grey_subset, OF_subset])
		fc6_list.append(fc6)
	#calculate the mean RVF probability of the samples to produce final estimate
	fc6_mean = np.mean(fc6_list)
	return fc6_mean

'''
Function: write_csv
Given file pointer and a 1D list of data, writes a csv file containing the list (vertically formatted)

Input:
fp = file pointer to csv file to write
datalist = 1D python list containing data

Output:
None
'''
def write_csv(fp, datalist):
	for data in datalist:
		fp.write(str(data) + ',\n')

if __name__ == '__main__':
	parser = ap.ArgumentParser()
	parser.add_argument("-r", "--root", required=True, help='full path to directory where DICOMS are initially stored')
	parser.add_argument("-w", "--weights", required=True, help='full path to weights file of trained TwoStream model')
	parser.add_argument('-s', '--splitnum', type=int, required=True, help='integer for which split you want to predict on.')

	args = vars(parser.parse_args())

	root = args['root'] #This is the root directory
	twostream_weights = args['weights'] 
	splitnum = args['splitnum']

	grey_direc = os.path.join(root, 'greyscale')
	OF_direc = os.path.join(root, 'OF')

	classmap_path = os.path.join(root, 'classmap.pkl') #this pickle file contains map of filename to RVF class
	filename2class = pkl.load(open(classmap_path, 'rb'))

	#define input sizes for greyscale and OF arms
	grey_input_size = (32, 112, 112, 1)
	OF_input_size = (32, 112, 112, 2)

	#Create TwoStream model object
	model = two_stream(saved_weights=twostream_weights, nframes=32, nrows=112, ncols=112, lr=1e-4)
	
	#Define the folders where the data is stored 
	grey_test_dir = os.path.join(grey_direc, 'split' + str(splitnum) + '_test')
	OF_test_dir = os.path.join(OF_direc, 'split' + str(splitnum) + '_test')

	grey_train_dir = os.path.join(grey_direc, 'split' + str(splitnum) + '_train')
	OF_train_dir = os.path.join(OF_direc, 'split' + str(splitnum) + '_train')

	grey_val_dir = os.path.join(grey_direc, 'split' + str(splitnum) + '_val')
	OF_val_dir = os.path.join(OF_direc, 'split' + str(splitnum) + '_val')

	test_file_list = os.listdir(grey_test_dir)
	val_file_list = os.listdir(grey_val_dir)

	X_val = []
	y_val = []

	X_test = []
	y_test = []

	val_filelist = []
	test_filelist = []
	val_redcaps = []
	test_redcaps = []

	#Predict validation set first
	val_start = time.time()
	counter = 0
	for f in val_file_list:
		if f[-3:] == 'pkl':
			if f[:-4] not in filename2class.keys():
				print(f, 'not in classmap keys!')
				continue
			RVF = (filename2class[f[:-4]] > THRESHOLD)
			grey_path = os.path.join(grey_val_dir, f)
			OF_path = os.path.join(OF_val_dir, f)

			redcap = f[:-5]
			val_redcaps.append(redcap)
			filename = f
			val_filelist.append(filename)

			fc = predict_RVF(model=model, grey_filepath=grey_path, OF_filepath=OF_path)
			
			X_val.append(fc)
			y_val.append(RVF)
			counter += 1

	val_end = time.time()
	val_time_per_echo = (val_end - val_start)/float(counter)

	#Predict test set
	test_start = time.time()
	counter = 0
	for f in test_file_list:
		if f[-3:] == 'pkl':
			if f[:-4] not in filename2class.keys():
				print(f, 'not in classmap keys!')
				continue
			RVF = (filename2class[f[:-4]] > THRESHOLD)
			grey_path = os.path.join(grey_test_dir, f)
			OF_path = os.path.join(OF_test_dir, f)

			redcap = f[:-5]
			test_redcaps.append(redcap)
			filename = f
			test_filelist.append(filename)
			
			fc = predict_fc6(model=model, grey_filepath=grey_path, OF_filepath=OF_path)
			
			X_test.append(fc)
			y_test.append(RVF)
			counter += 1
	test_end = time.time()
	test_time_per_echo = (test_end - test_start)/float(counter)

	print('Val time per echo:', val_time_per_echo, 'Test time per echo:', test_time_per_echo)
	
	#Will save predictions within a new folder created within the current working directory
	save_direc = os.path.join(root, 'predictions')
	#Make a subdirec specific for this split
	split_direc = os.path.join(save_direc, 'output_split' + str(splitnum))
	safe_makedir(save_direc)
	safe_makedir(split_direc)

	#Save predictions as csv and as pickle files
	write_csv(open(os.path.join(split_direc, 'X_val.csv'), 'w+'), X_val)
	write_csv(open(os.path.join(split_direc, 'X_test.csv'), 'w+'), X_test)
	
	pkl.dump(X_val, open(os.path.join(split_direc, 'X_val.pkl'), 'wb'))
	pkl.dump(X_test, open(os.path.join(split_direc, 'X_test.pkl'), 'wb'))
	
	write_csv(open(os.path.join(split_direc, 'val_filenames.csv'), 'w+'), val_filelist)
	write_csv(open(os.path.join(split_direc, 'test_filenames.csv'), 'w+'), test_filelist)
	write_csv(open(os.path.join(split_direc, 'val_redcaps.csv'), 'w+'), val_redcaps)
	write_csv(open(os.path.join(split_direc, 'test_redcaps.csv'), 'w+'), test_redcaps)

	pkl.dump(val_filelist, open(os.path.join(split_direc, 'val_filenames.pkl'), 'wb'))
	pkl.dump(test_filelist, open(os.path.join(split_direc, 'test_filenames.pkl'), 'wb'))
	pkl.dump(val_redcaps, open(os.path.join(split_direc, 'val_redcaps.pkl'), 'wb'))
	pkl.dump(test_redcaps, open(os.path.join(split_direc, 'test_redcaps.pkl'), 'wb'))

	write_csv(open(os.path.join(split_direc, 'y_val.csv'), 'w+'), y_val)
	write_csv(open(os.path.join(split_direc, 'y_test.csv'), 'w+'), y_test)
	
	pkl.dump(y_val, open(os.path.join(split_direc, 'y_val.pkl'), 'wb'))
	pkl.dump(y_test, open(os.path.join(split_direc, 'y_test.pkl'), 'wb'))
	







