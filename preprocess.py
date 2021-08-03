'''
Script:  preprocess.py
Dependency environment: tf_gpu

Script to sort, preprocess, normalize echo DICOM files.
'''
import os
import numpy as np
import pickle as pkl
import random
from sklearn.model_selection import StratifiedShuffleSplit
import argparse as ap
import multiprocessing as mp 
import time
from shutil import copyfile, move, rmtree
import pydicom as dcm
import skimage.transform as trans
import tensorflow as tf
from utils import *

####GLOBALS#####
#SET THESE IF YOU WANT TO CHANGE FROM DEFAULT ARCHITECTURE
NFRAMES = 32
NROWS = 112
NCOLS = 112

'''
Function: preprocess_dicom
Given filename and what class it is (as string), will preprocess an individual echo DICOM file.

Input: 
filename = dicom file name (eg 'stanford0002a.dcm')
label = RVF class as str (eg 'normal')
dicom_direc = full path to currect working directory
processed_direc = full path to directory where preprocessed files will be saved

Output:
None
'''
def preprocess_dicom(filename, label, dicom_direc, processed_direc):
	save_name = filename[:-4] + '.pkl'
	save_path = os.path.join(processed_direc, save_name)

	if os.path.exists(save_path): #dont bother with computation if file already exists
		print(save_path, 'exists already! skipping...')
		return 

	class_direc = os.path.join(dicom_direc, label)
	ds = dcm.dcmread(os.path.join(class_direc, filename))
	frame_list = dicom2imglist(ds)
	if frame_list[0].shape[0:2] != (NROWS, NCOLS):
		print('Found dcm images of size ', frame_list[0].shape, 'RESIZING!')
		new_frame_list = resize_img_list(frame_list, (NROWS, NCOLS))

	stack = np.stack(new_frame_list, axis=0)
	movie = stack/np.amax(stack) 
	movie = np.reshape(movie, (movie.shape[0], NROWS, NCOLS, 1))

	print('Saving stack of size ' + str(movie.shape) + ' as ' + save_name)
	pkl.dump(movie, open(save_path, 'wb'))

'''
Helper function: normalize
Given the path to direc w/ raw greyscale files and filename, as well as 
mean and std dev of the entire greyscale dataset, normalizes the specified greyscale movie file.
Saves file within save_direc.

Input:
filename = name of file to be processed, must be pickled raw greyscale pixel Numpy array
processed_direc = full path to directory containing raw greyscale pixels, saved as pickle files
save_direc = full path to directory where normalized greyscale pixel arr will be saved as pickle file. MUST exist at runtime
data_mean = float, mean of entire raw greyscale set
data_std = float, std dev of entire raw greyscale set

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
Given full path of current working direc (root_direc), full path of direc w/ raw greyscale files, and full path of directory
where normalized greyscale files will be saved, will normalize all greyscale files, using multithreading.

Input:
root_direc = full path to current working direc
processed_direc = full path to direc w/ raw greyscale pickle files
save_direc = full path to direc where normalized greyscale pickle files will be saved. Does not have to exist at runtime, will be created

Output:
None
'''
def normalize_pkl(root_direc, processed_direc):
	save_path = os.path.join(root_direc, 'data_mean_std.pkl')
	filenames = os.listdir(processed_direc)
	avg_list = []
	variance_list = []
	for f in filenames:
		if f[-3:] == 'pkl':
			movie = pkl.load(open(os.path.join(processed_direc, f), 'rb'))
			avg_list.append(np.mean(movie))
			variance_list.append(np.var(movie))

	data_mean = np.mean(avg_list)
	data_std = np.sqrt(np.mean(variance_list))
	print(data_mean, data_std)

	
	pkl.dump([data_mean, data_std], open(save_path, 'wb'))

	save_direc = os.path.join(root_direc, 'normalized_dicoms')
	safe_makedir(save_direc)

	p = mp.Pool()
	for f in filenames:
		if f[-3:] == 'pkl':
			p.apply_async(normalize, [f, processed_direc, save_direc, data_mean, data_std])
			normalize(f, processed_direc, save_direc, data_mean, data_std)

	p.close()
	p.join()


if __name__ == '__main__':
	#Parse arguments
	parser = ap.ArgumentParser()
	parser.add_argument("-d", "--root", required=True, help='full path to directory where DICOMS are initially stored')
	parser.add_argument("-c", "--csv", required=True, default='REDcap_class.csv', help='csv file name with REDCap to RVF class mapping')
	parser.add_argument('-m', '--multiprocess', action='store_true', help='Toggles multiprocessing on/off')
	parser.add_argument('-r', '--recalculate', action='store_true', help='Will force recalculate')
	args = vars(parser.parse_args())

	print(args)
	dicom_direc = args['root'] #This is the root directory
	multiprocess_bool = args['multiprocess']
	csv_name = args['csv']
	recalculate = args['recalculate']

	print("Root directory: ", dicom_direc)
	if multiprocess_bool:
		print('Multiprocessing: ON')
		num_cpu = mp.cpu_count()
		print('Number of CPUs available: ', num_cpu)
	else:
		print('Multiprocessing: OFF')

	global_start = time.time()

	redcap_dict = make_redcapmap(dicom_direc, csv_name)
	redcap_list = list(redcap_dict.keys())
	
	
	###PART I###
	#Sort DICOMs
	print('Sorting dicoms...')
	sort_dicoms(dicom_direc) #Sort dicoms into different classes #UNCOMMENT ME IF STARTING ON A FRESH DATASET (FIRST TIME PROCESSING DICOMS)
	
	classes = ['normal','mild','moderate','severe']
	
	file_list = [] #holds filenames
	class_list = [] #holds classes tied to the filenames
	index = 0
	for c in classes:
		class_path = os.path.join(dicom_direc, c)
		filenames = os.listdir(class_path)
		for filename in filenames:
			if filename[-3:] == 'dcm':
				file_list.append(filename)
				class_list.append(index)
		index += 1
	
	processed_direc = os.path.join(dicom_direc, 'processed_dicoms') #we'll store all the processed video files here
	safe_makedir(processed_direc)
	
	
	if multiprocess_bool:
		p = mp.Pool()
	
	###PART II### PREPROCESS DATA
	#Here preprocess will resize and rescale pixels to 0-1 in case pixels are still in 8-bit 0-255 range
	print('Starting data preprocessing...')
	start = time.time()
	for i in range(len(file_list)):
		filename = file_list[i]
		label = classes[class_list[i]]
		redcap = filename[:-5]
		if redcap not in redcap_dict.keys():
			print('REDCAP', redcap, 'is not in csv file!')
			continue
		if multiprocess_bool:
			p.apply_async(preprocess_dicom, [filename, label, dicom_direc, processed_direc])
		else:
			preprocess_dicom(filename, label, dicom_direc, processed_direc)
	elapsed = time.time()-start
	time_per_file = 0
	if len(file_list) != 0:	
		time_per_file = elapsed/float(len(file_list))
	print('Done! Took ' + str(elapsed) + ' sec to run! (' + str(time_per_file) + ' sec per file)')
	
	if multiprocess_bool:
		p.close()
		p.join()
	
	#normalize the greyscale pixel arrays
	normalize_pkl(dicom_direc, processed_direc)
	
	#Make a map of filenames and RVF class, save as pickle
	#map should look like this: {'stanford0002a':0, 'houston0021b':3, ...}
	processed_direc = os.path.join(dicom_direc, 'normalized_dicoms')

	pickle_file_list = os.listdir(processed_direc)
	class_dict = {} 
	
	for f in pickle_file_list:
		redcap = f[:-5]
		name = f[:-4]
		if redcap not in redcap_dict.keys():
			print(redcap, 'is not a key in REDCap Map!')
			#os.remove(os.path.join(processed_direc, f))
			continue
		if redcap_dict[redcap] == 'normal':
			class_dict[name] = 0
		elif redcap_dict[redcap] == 'mild':
			class_dict[name] = 1
		elif redcap_dict[redcap] == 'moderate':
			class_dict[name] = 2
		elif redcap_dict[redcap] == 'severe':
			class_dict[name] = 3
		else:
			continue

	dict_name = os.path.join(dicom_direc, 'classmap.pkl') #Save filename:RVF dictionary here
	pkl.dump(class_dict, open(dict_name, 'wb')) 
	
