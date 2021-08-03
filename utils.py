'''
Script: utils.py
Dependency environment: tf_gpu

Script containing all commonly used helper functions
'''
from __future__ import print_function
import numpy as np
import os
import skimage.transform as trans
import pydicom as dcm
import pickle as pkl
from shutil import move
from sklearn.preprocessing import normalize
import random
import multiprocessing as mp
from keras.utils import Sequence

###GLOBALS####
NFRAMES = 32  # number of frames in video clip
NROWS = 112  # pixels in row of frame
NCOLS = 112  # pixels in col of frame
NCHANNEL = 1  # number of channels (hold over from when we incorporated optical flow)

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
Helper function: resize_img_list
Given list of images, resize all images in list to dim specified by new_size

Input:
img_list = 1D list of numpy arr
new_size = tuple specifying new image size

Output:
list = list of resized images
'''
def resize_img_list(img_list, new_size):
	new_list = []
	for i in range(len(img_list)):
		new_img = trans.resize(img_list[i], new_size)
		new_list.append(new_img)
	return new_list

'''
Function: make_redcapmap
Takes a csv file with REDcap and RVF class (written as a string) and turns it into a dict
Input:
* dicompath=full path to direc where csv lies
* csv_name=csv file name
Output:
* class_dict=dict with property {key=redcap ID, value=RVF class}
'''
def make_redcapmap(dicompath, csv_name):
	class_dict = {}
	with open(os.path.join(dicompath, csv_name)) as fp:
		line = fp.readline()
		counter = 0
		while line:
			if counter == 0:
				counter += 1
				line = fp.readline()
				continue
			else:
				line = line.strip('\n')
				items = line.split(',')
				redcap = items[0]
				RVF = items[1]
				if redcap not in class_dict.keys():
					class_dict[redcap] = RVF
				line = fp.readline()
				counter += 1
	return class_dict

'''
Function: dicom2imgdict
Converts PyDICOM dataset into a list of frames
Input: 
* imagefile=PyDICOM dataset class
Output: list of pixel arrays in chronological order
'''
def dicom2imglist(imagefile):
	try:
		ds = imagefile
		nrow = int(ds.Rows)
		ncol = int(ds.Columns)
		ArrayDicom = np.zeros((nrow, ncol), dtype=ds.pixel_array.dtype)
		img_list = []
		if len(ds.pixel_array.shape) == 4:  # format is (nframes, nrow, ncol, 3)
			nframes = ds.pixel_array.shape[0]
			R = ds.pixel_array[:, :, :, 0]
			B = ds.pixel_array[:, :, :, 1]
			G = ds.pixel_array[:, :, :, 2]
			gray = (0.2989 * R + 0.5870 * G + 0.1140 * B)
			for i in range(nframes):
				img_list.append(gray[i, :, :])
			return img_list
		elif len(ds.pixel_array.shape) == 3:  # format (nframes, nrow, ncol) (ie in grayscale already)
			nframes = ds.pixel_array.shape[0]
			for i in range(nframes):
				img_list.append(ds.pixel_array[i, :, :])
			return img_list
	except:
		return None

'''
Function: sort_dicoms
Sorts/moves the DICOM files within the specified directory by RV failure class,
as specified by REDcap_class.csv which MUST be present in the specified directory.
REDcap_class.csv contains two columns, first column is REDcap number and second column contains 
RV failure class as specified by on of the following strings ['normal', 'mild', 'moderate', 'severe'].
Moved DICOMs are renamed according to REDCap ID stored in the DICOM "Patient ID" metadata.
Input:
* dicompath=full path to directory to be sorted
Output:
None
'''
def sort_dicoms(dicompath):
	normal_dir = os.path.join(dicompath, 'normal')
	mild_dir = os.path.join(dicompath, 'mild')
	moderate_dir = os.path.join(dicompath, 'moderate')
	severe_dir = os.path.join(dicompath, 'severe')

	classes = ['normal', 'mild', 'moderate', 'severe']

	safe_makedir(normal_dir)
	safe_makedir(mild_dir)
	safe_makedir(moderate_dir)
	safe_makedir(severe_dir)

	redcap_RVF_dict = make_redcapmap(dicompath, 'REDcap_class.csv')
	redcap_counter = {}
	filenames = os.listdir(dicompath)

	for filename in filenames:
		if filename[-3:] != 'dcm':
			continue
		else:
			try:
				ds = dcm.dcmread(os.path.join(dicompath, filename))
			except:
				print('BAD DICOM FILE SKIPPING')
				pass
				continue
			redcap = ds.PatientID
			nframes = int(ds.NumberOfFrames)
			if nframes < NFRAMES:
				print('Skipping dicom! Not enough frames!')
				continue
			if redcap not in redcap_RVF_dict.keys():
				print('Skipping dicom! No RVF label found.', redcap)
				continue
			else:
				RVF = redcap_RVF_dict[redcap]
			if redcap not in redcap_counter.keys():
				redcap_counter[redcap] = 0
			else:
				redcap_counter[redcap] += 1

			new_filename = redcap + chr(redcap_counter[redcap] + 97) + '.dcm'
			os.rename(os.path.join(dicompath, filename), os.path.join(dicompath, new_filename))
			class_dir = os.path.join(dicompath, RVF)
			print('moving ' + os.path.join(dicompath, filename) + ' to ' + os.path.join(class_dir, new_filename))
			move(os.path.join(dicompath, new_filename), os.path.join(class_dir, new_filename))

'''
Helper function: extract_mrn
Given full path to current working direc and dicom filename, will return
the associated patient ID field of that DICOM file

Input:
root_direc = full path to current working direc (where dicom is presumably)
filename = filename of DICOM file

Output:
str = patient identifier listed in DICOM metadata
'''
def extract_mrn(root_direc, filename):
	if filename[-3:] != 'dcm':
		print(filename + ' IS NOT A DICOM!')
		return None
	else:
		try:
			filepath = os.path.join(root_direc, filename)
			ds = dcm.dcmread(filepath)
		except:
			print('BAD DICOM FILE SKIPPING')
			return None
		return ds.PatientID

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

