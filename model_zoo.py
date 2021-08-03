'''
Script: model_zoo.py
Dependency environment: tf_gpu

Contains functions that define the TwoStream model, and Resnet152 regression model
'''
import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras_adamw import AdamW
from resnet_builder_keras import Resnet3DBuilder
import tensorflow as tf
from keras.metrics import TopKCategoricalAccuracy, AUC, FalseNegatives, FalsePositives, TruePositives, TrueNegatives
import keras.backend as K 

'''
Function: two_stream
Given parameters of the TwoStream model, creates Keras Model object representing the model of interest
Inputs:
grey_weights = full path to pretrained Resnet152 greyscale weights
OF_weights = full path to pretrained Resnet152 optical flow weights
saved_weights = optional, full path to last saved TwoStream model weights
nframes = number of frames in video clip
nrows = pixels in X dim
ncols = pixels in Y dim
lr = learning rate, default 1e-4
cosine_anneal = bool for AdamW, toggles cosine annealing of learning rate
lr_weight = optional map for AdamW, can apply weights to learning rates of every layer, default None

Output:
Model object
'''
def two_stream(grey_weights = None, OF_weights = None, saved_weights = None, nframes = 32, nrows = 112, ncols = 112, lr = 1e-4, cosine_anneal=False, tot_iter=None, lr_weight=None):
	if (grey_weights == None) and (saved_weights == None):
		print('Pretrained greyscale weights not defined!')
		exit()
	if (OF_weights == None) and (saved_weights == None):
		print('Pretrained OF weights not defined!')
		exit()

	#This code defines a new TwoStream model, as if training from epoch 0
	if saved_weights == None:
		grey_input_size = (nframes, nrows, ncols, 1)
		OF_input_size = (nframes, nrows, ncols, 2)

		#Load Greyscale pretrained model on Echonet
		grey_tmp_model = Resnet3DBuilder.build_resnet_152(grey_input_size, 2)
		grey_tmp_model.layers.pop()
		grey_tmp_out = grey_tmp_model.layers[-1].output 
		grey_transfer_out = Dense(1, activation='linear', name='grey_preds')(grey_tmp_out)
		grey_transfer_model = Model(grey_tmp_model.input, grey_transfer_out)
		print('Loading weights for greyscale transfer model:', grey_weights)
		grey_transfer_model.load_weights(grey_weights)
		grey_transfer_model.layers.pop()
		grey_fc = grey_transfer_model.layers[-1].output #-> flattened output of greyscale transfer model

		#Load Greyscale pretrained model on Echonet
		OF_tmp_model = Resnet3DBuilder.build_resnet_152(OF_input_size, 2)
		OF_tmp_model.layers.pop()
		OF_tmp_out = OF_tmp_model.layers[-1].output 
		OF_transfer_out = Dense(1, activation='linear', name='OF_preds')(OF_tmp_out)
		OF_transfer_model = Model(OF_tmp_model.input, OF_transfer_out)
		print('Loading weights for greyscale transfer model:', OF_weights)
		OF_transfer_model.load_weights(OF_weights)
		OF_transfer_model.layers.pop()
		OF_fc = OF_transfer_model.layers[-1].output #-> flattened output of OF transfer model

		#Merging the 2 streams
		concat = Concatenate()([grey_fc, OF_fc])
		out = Dense(1, activation='sigmoid', name='preds')(concat)

		model = Model([grey_transfer_model.input, OF_transfer_model.input], out) #define model

		#define optimizer options
		opt = AdamW(lr=lr, model=model)
		if lr_weight != None:
			opt = AdamW(lr=lr, model=model, lr_multipliers=lr_weight)
			if cosine_anneal:
				opt = AdamW(lr=lr, model=model, lr_multipliers=lr_weight, use_cosine_annealing=True, total_iterations=tot_iter)
		else:
			if cosine_anneal:
				opt = AdamW(lr=lr, model=model, use_cosine_annealing=True, total_iterations=tot_iter)
		
		model.compile(optimizer=opt, loss='binary_crossentropy', metrics = ['accuracy', AUC(), TruePositives(), TrueNegatives(), FalsePositives(), FalseNegatives()])
		#model.summary(). #Uncomment to print Keras summary of model
		return model

	#This code will build a TwoStream model with weights from last saved model
	else:
		grey_input_size = (nframes, nrows, ncols, 1)
		OF_input_size = (nframes, nrows, ncols, 2)

		#Build greyscale arm
		grey_model = Resnet3DBuilder.build_resnet_152(grey_input_size, 2)
		grey_model.layers.pop()
		grey_fc = grey_model.layers[-1].output 

		#Build OF arm
		OF_model = Resnet3DBuilder.build_resnet_152(OF_input_size, 2)
		OF_model.layers.pop()
		OF_fc = OF_model.layers[-1].output 

		#Merge the 2 streams
		concat = Concatenate()([grey_fc, OF_fc])
		out = Dense(1, activation='sigmoid', name='preds')(concat)

		#Define model
		model = Model([grey_model.input, OF_model.input], out)
		model.load_weights(saved_weights)

		#Define optimizer settings
		opt = AdamW(lr=lr, model=model)
		if lr_weight != None:
			opt = AdamW(lr=lr, model=model, lr_multipliers=lr_weight)
			if cosine_anneal:
				opt = AdamW(lr=lr, model=model, lr_multipliers=lr_weight, use_cosine_annealing=True, total_iterations=tot_iter)
		else:
			if cosine_anneal:
				opt = AdamW(lr=lr, model=model, use_cosine_annealing=True, total_iterations=tot_iter)
		
		model.compile(optimizer=opt, loss='binary_crossentropy', metrics = ['accuracy', AUC(), TruePositives(), TrueNegatives(), FalsePositives(), FalseNegatives()])
		#model.summary()
		return model

'''
Function: Resnet152_AdamW_regression
Given parameters of Resnet152, creates Keras Model object representing the model of interest, uses AdamW optimizer
Inputs:
pretrained_weights = full path to Resnet152 model weights if already partially trained
input size = tuple of neural net input size (defaults to (32,112,112,1)). format is (NFRAMES, NROWS, NCOLS, NCHANNELS)
lr = learning rate, default 1e-3
cosine_anneal = bool for AdamW, toggles cosine annealing of learning rate
lr_weight = optional map for AdamW, can apply weights to learning rates of every layer, default None

Output:
Model object
'''
def Resnet152_AdamW_regression(pretrained_weights = None, input_size = (64,384,384,1), lr=1e-3, cosine_anneal=False, tot_iter=None, lr_weight=None):
	model = Resnet3DBuilder.build_resnet_152(input_size, classes)
	model.layers.pop()
	last_layer = model.layers[-1].output 
	out = Dense(1, activation='linear', name='preds')(last_layer)

	final_model = Model(model.input, out)

	if (pretrained_weights):
		try:
			final_model.load_weights(pretrained_weights)
		except:
			print('ERROR IN MODEL WEIGHTS, training from scratch...')
			pass

	#Set optimizer settings
	opt = AdamW(lr=lr, model=final_model)
	if lr_weight != None:
		opt = AdamW(lr=lr, model=final_model, lr_multipliers=lr_weight)
		if cosine_anneal:
			opt = AdamW(lr=lr, model=final_model, lr_multipliers=lr_weight, use_cosine_annealing=True, total_iterations=tot_iter)
	else:
		if cosine_anneal:
			opt = AdamW(lr=lr, model=final_model, use_cosine_annealing=True, total_iterations=tot_iter)
	final_model.compile(optimizer=opt, loss='mean_squared_error', metrics = [])

	#final_model.summary() #Uncomment to print Keras summary of model architecture
	return final_model


