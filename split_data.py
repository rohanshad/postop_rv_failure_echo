'''
Script: split_data.py
Dependency environment: tf_gpu

Given a csv file with patient REDCaps and split allocation ('train', 'val', 'test'), will copy greyscale and 
OF normalized files into new folders.

The csv file must be in the current working directory. Csv file must not have a header, and only has two columns, 
first one is REDCap ID, second column is allocation

'''
import os
import numpy as np
from shutil import copyfile
import multiprocessing as mp 
from utils import *
import argparse as ap 

'''
Function: make_sortmap
Given full path to current working directory and name of csv file, will
return map of patient REDCap and its corresponding allocation
eg. map = {'stanford002': 'train', 'houston0102':'val', etc...}
Input:
dicompath = full path to current working directory
csv_name = filename of csv file with allocations

Output:
group_dict = map of REDCap ID and its corresponding allocation
'''
def make_sortmap(dicompath, csv_name):
	group_dict = {}
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
				group = items[1]
				if redcap not in group_dict.keys():
					group_dict[redcap] = group
				line = fp.readline()
				counter += 1
	return group_dict

'''
Function: copy_sort
Given a full path to direc w/ normalized echos, full path to destination direc
and the dict mapping REDCap ID to allocation, copies files to the appropriate folder.
Uses multiprocessing
Input:
source_direc = full path to direc w/ normalized echos (either greyscale or OF)
dest_direc = full path to destination directory. It and its subdirectories (train, val, and test) 
must exist at runtime
sort_map = dict mapping REDCap ID to allocation

Output:
None
'''
def copy_sort(source_direc, dest_direc, sort_map):
	filenames = os.listdir(source_direc)
	
	p = mp.Pool()
	
	for f in filenames:
		if f[-3:] == 'pkl':
			redcap = f[:-5]
			if redcap not in sort_map.keys():
				print(redcap, ' not in map! ERROR')
				continue
			group = sort_map[redcap]
			source = os.path.join(source_direc, f)
			if group == 'train':
				dest = os.path.join(os.path.join(dest_direc, 'split' + str(splitnum) + '_train'), f)
			elif group == 'val':
				dest = os.path.join(os.path.join(dest_direc, 'split' + str(splitnum) + '_val'), f)
			elif group == 'test':
				dest = os.path.join(os.path.join(dest_direc, 'split' + str(splitnum) + '_test'), f)
			else:
				print('BADNESS')
				exit(1)

			print('Copying', source, 'to', dest, '...')
			p.apply_async(copyfile, [source, dest])
	
	p.close()
	p.join()


if __name__ == '__main__':
	parser = ap.ArgumentParser()
	parser.add_argument('-r', '--root', required=True, help='full path to current working directory')
	parser.add_argument('-s', '--splitnum', type=int, required=True, help='integer for which split you want to predict on.')

	args = vars(parser.parse_args())

	root = args['root'] #This is the root directory
	splitnum = args['splitnum']

	grey_direc = os.path.join(root, 'greyscale')

	#Make necessary subdirectories for both greyscale and OF videos
	grey_train = os.path.join(grey_direc, 'split' + str(splitnum) + '_train')
	grey_val = os.path.join(grey_direc, 'split' + str(splitnum) + '_val')
	grey_test = os.path.join(grey_direc, 'split' + str(splitnum) + '_test')
	safe_makedir(grey_train)
	safe_makedir(grey_val)
	safe_makedir(grey_test)

	OF_direc = os.path.join(root, 'OF')
	OF_train = os.path.join(OF_direc, 'split' + str(splitnum) + '_train')
	OF_val = os.path.join(OF_direc, 'split' + str(splitnum) + '_val')
	OF_test = os.path.join(OF_direc, 'split' + str(splitnum) + '_test')
	safe_makedir(OF_train)
	safe_makedir(OF_val)
	safe_makedir(OF_test)

	#Make the dict mapping REDCap ID to allocation
	sort_map = make_sortmap(root, 'split_allocation_' + str(splitnum) + '.csv')

	#Copy files to the right place
	copy_sort(os.path.join(root, 'normalized_dicoms'), grey_direc, sort_map)
	copy_sort(os.path.join(root, 'normalized_OF'), OF_direc, sort_map)


