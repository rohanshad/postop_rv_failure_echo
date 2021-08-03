'''
Script: optical_flow_calculator.py
Dependency environment: tf_gpu

Script to be run after dicoms have been sorted and preprocessed. 
Given a folders w/ dicoms sorted by RVF class, will calculate optical flow
and output the optical flow vector field to another folder. These arrays are then
normalized over the entire dataset and the normalized arrays saved in a separate folder.

Note that script accomodates 4 RVF classes, but for the purposes of the data analysis in paper
echos were sorted using only two classes (normal and severe), as per the 2020 INTERMACS 
definition of RV failure 
'''
import os
import numpy as np
import pickle as pkl
import cv2 as cv
import multiprocessing as mp
from utils import *
import time
import pydicom as dcm
import time
import argparse as ap 

#GLOBALS
NFRAMES = 32
NROWS = 112
NCOLS = 112

'''
Helper function: normalize
Given the path to direc w/ raw optical flow files and filename, as well as 
mean and std dev of the entire raw optical flow dataset, normalizes the specified raw optical flow file.
Saves file within save_direc.

Input:
filename = name of file to be processed, must be pickled raw optical flow Numpy array
processed_direc = full path to directory containing raw optical flow vector fields, saved as pickle files
save_direc = full path to directory where normalized vector field will be saved as pickle file. MUST exist at runtime
data_mean = float, mean of entire raw optical flow set
data_std = float, std dev of entire raw optical flow set

Output:
None
'''
def normalize(filename, processed_direc, save_direc, data_mean, data_std):
	movie = pkl.load(open(os.path.join(processed_direc, filename), 'rb'))
	movie = movie - data_mean
	movie = movie / data_std
	print('Normalizing file:', filename)
	pkl.dump(movie, open(os.path.join(save_direc, filename), 'wb'))

'''
Function: normalize_pkl
Given full path of current working direc (root_direc), full path of direc w/ raw OF files, and full path of directory
where normalized OF files will be saved, will normalize all OF files, using multithreading.

Input:
root_direc = full path to current working direc
processed_direc = full path to direc w/ raw OF pickle files
save_direc = full path to direc where normalized OF pickle files will be saved. Does not have to exist at runtime, will be created

Output:
None
'''
def normalize_pkl(root_direc, processed_direc, save_direc, multiprocess=True):
	#First need to calculate the mean and std of the raw OF dataset
	save_path = os.path.join(root_direc, 'data_mean_std_OF.pkl') #we'll save the mean and std dev in a pickle file in case we need it again
	filenames = os.listdir(processed_direc)
	avg_list = []
	variance_list = []
	#Iterate over all files, calculate mean and variance. 
	#Can be slow, not multithreaded
	for f in filenames:
		if f[-3:] == 'pkl':
			movie = pkl.load(open(os.path.join(processed_direc, f), 'rb'))
			avg_list.append(np.mean(movie))
			variance_list.append(np.var(movie))

	#overall dataset mean is the mean of means
	#overall dataset std dev is sqrt(mean of variances)
	data_mean = np.mean(avg_list)
	data_std = np.sqrt(np.mean(variance_list))

	pkl.dump([data_mean, data_std], open(save_path, 'wb')) #save mean and std dev

	safe_makedir(save_direc)

	if multiprocess:
		p = mp.Pool()

	for f in filenames:
		if f[-3:] == 'pkl':
			#Use multiprocessing to normalize all raw OF files.
			if multiprocess:
				p.apply_async(normalize, [f, processed_direc, save_direc, data_mean, data_std])
			else:
				normalize(f, processed_direc, save_direc, data_mean, data_std) 

	if multiprocess:
		p.close()
		p.join()

'''
Helper Function: video2OF
Given full path of DICOM files to process and full path of resulting file, will calculate the optical flow and save vector field arr as pickle
Note that optical flow is calculated on FULL SIZED echos, then resized to desired frame size for neural nets (112x112)
Input:
dicom_path = full path to DICOM file to process
save_path = full path of resulting pickle file

Output:
None
'''
def video2OF(dicom_path, save_path):
	#Read in DICOM file
	ds = dcm.dcmread(dicom_path)
	#Convert data object to list of frames
	frame_list = dicom2imglist(ds)
	#Convert list of frames into Numpy array NFRAMESxNROWSxNCOLS
	img_arr = np.stack(frame_list, axis=0)
	print('Processing file', dicom_path)
	nrows = img_arr.shape[1]
	ncols = img_arr.shape[2]
	nframes = img_arr.shape[0]
	concat_list = []
	for i in range(nframes - 1):
		frame = img_arr[i, :, :]
		next_frame = img_arr[i + 1, :, :]
		im1 = np.reshape(frame, (nrows, ncols, 1))
		im2 = np.reshape(next_frame, (nrows, ncols, 1))
		#Calculate optical flow on full sized frames
		flow = cv.calcOpticalFlowFarneback(frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
		stack = np.stack([flow[:, :, 0], flow[:, :, 1]])
		stack = np.transpose(stack, (1, 2, 0))
		concat_list.append(stack)
	#In order to keep optical flow array the same size as input, will copy the last optical flow frame.
	concat_list.append(concat_list[-1])
	#resize optical flow vector field to desired frame size
	new_size = (NROWS, NCOLS, 2)
	resized_list = resize_img_list(concat_list, new_size)
	nframes = len(resized_list)
	#Double check that OF array has enough frames. This should NEVER execute if sorting in preprocessing worked.
	#^prob can change into an assert statement
	if nframes < NFRAMES:
		print('NOT ENOUGH FRAMES! SKIPPING...', nframes)
		return
	OF_stack = np.stack(resized_list, axis=0)
	print('Saving stack of size ', OF_stack.shape)
	pkl.dump(OF_stack, open(save_path, 'wb'))

'''
Function: convert2OF
Given full path of current working directory full path of directory where OF files will be saved, 
will calculate the optical flow and save vector field arr as pickle. Multiprocessing enabled by default
Note that optical flow is calculated on FULL SIZED echos, then resized to desired frame size for neural nets (112x112)
Input:
root_direc = full path to current working directory
save_direc = full path to directory where raw optical flow files will be saved
multiprocess = bool that toggles use of multiprocessing, default to True

Output:
None
'''
def convert2OF(root_direc, save_direc, multiprocess=True):
	safe_makedir(save_direc)
	classes = ['normal', 'mild', 'moderate', 'severe']
	dicom_paths = []
	dicom_names = []

	for c in classes:
		direc_path = os.path.join(root_direc, c)
		filenames = os.listdir(direc_path)
		for f in filenames:
			if f[-3:] == 'dcm':
				dicom_paths.append(os.path.join(direc_path, f))
				dicom_names.append(f[:-4])

	start = time.time()
	if multiprocess:
		p = mp.Pool()
	for i in range(len(dicom_paths)):
		dicom_path = dicom_paths[i]
		dicom_name = dicom_names[i]
		if os.path.exists(os.path.join(save_direc, dicom_name + '.pkl')):
			print(os.path.join(save_direc, dicom_name + '.pkl'), ' already computed! skipping...')
		else:
			save_path = os.path.join(save_direc, dicom_name + '.pkl')
			if multiprocess:
				p.apply_async(video2OF, [dicom_path, save_path])
			else:
				video2OF(dicom_path, save_path)
	if multiprocess:
		p.close()
		p.join()
	end = time.time()
	time_per_echo = (end - start)/float(len(dicom_paths))
	print('Time per echo', time_per_echo)

if __name__ == '__main__':
	parser = ap.ArgumentParser()
	parser.add_argument('-r', '--root', required=True, help='full path to current working directory')
	parser.add_argument('-m', '--multiprocess', action='store_true', help='toggle multiprocess on or off, defaults to on')
	
	args = vars(parser.parse_args())

	root_direc = args['root'] #This is the root directory
	multiprocess = args['multiprocess']
	processed_direc = os.path.join(root_direc, 'processed_OF') #store raw resized OF files here
	normalized_direc = os.path.join(root_direc, 'normalized_OF') #store normalized OF files here
	convert2OF(root_direc, processed_direc, multiprocess=multiprocess)
	normalize_pkl(root_direc, processed_direc, normalized_direc, multiprocess=multiprocess)
