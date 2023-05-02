#!/usr/bin/env python
# coding: utf-8

import sys
import importlib

from dataio.datahandler import datahandler
from dataio.datareader import datareader
from models.TDCNNv1 import TDCNNv1
from config import *
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np
import sklearn.preprocessing

dataset_root_path = '/home/veronica/USANNIOMIT/sen2dwater_combined' # adjust to the correct one

BATCH_SIZE = 1
# BATCH_SIZE = 70
SHAPE      = (64,64)
NORMALIZE  = True
SCALE = False

NDMI_MODEL_PATH = '/home/veronica/USANNIOMIT/ndwi_model2.h5'
NDVI_MODEL_PATH = '/home/veronica/USANNIOMIT/ndvi_model2.h5'
NDDI_MODEL_PATH = '/home/veronica/USANNIOMIT/nddi_model2.h5'


'''
alternative scaling function
'''
def scaler(arr):
	res = np.zeros_like(arr) 

	min_arr = np.nanmin(arr, axis = 0)
	max_arr = np.nanmax(arr, axis = 0)

	print(min_arr)
	print(max_arr) 
	for i in range(arr.shape[0]):
		for j in range(arr.shape[1]):
				min_arr = np.min(arr[i][j])
				max_arr = np.max(arr[i][j])

				# print(min_arr)
				# print(max_arr)
				res[i][j] = (arr[i][j] - min_arr) / (max_arr - min_arr)
				# print(res.shape)
	return res



'''
get theoretic NDDI value 
'''
def getNDDI(NDVI,NDWI):
	return (NDVI-NDWI)/(NDVI+NDWI+0.0001)


'''
plotting
'''

def plot_results(NDVI_gt, ndvi_prediction, NDWI_gt, ndwi_prediction, NDDI_gt, nddi_prediction, nddi_from_ndmi_pred_and_ndvi_pred, NDDItheoreticGT):
	fig, axes = plt.subplots(nrows = 2, ncols = BATCH_SIZE, figsize = (BATCH_SIZE*5,3*5))
	for i in range(BATCH_SIZE):
		axes[0,i].imshow(ndvi_prediction[i,:,:,0])
		axes[1,i].imshow(NDVI_gt[i,:,:,0])

		axes[0,i].set_title('NDVI Prediction')
		axes[1,i].set_title('NDVI Ground Truth')

		axes[0,i].axis(False)
		axes[1,i].axis(False)

	fig.tight_layout()
	plt.show()
	plt.close()

	fig, axes = plt.subplots(nrows = 2, ncols = BATCH_SIZE, figsize = (BATCH_SIZE*5,3*5))
	for i in range(BATCH_SIZE):
		axes[0,i].imshow(ndwi_prediction[i,:,:,0])
		axes[1,i].imshow(NDWI_gt[i,:,:,0])

		axes[0,i].set_title('NDWI Prediction')
		axes[1,i].set_title('NDWI Ground Truth')

		axes[0,i].axis(False)
		axes[1,i].axis(False)

	fig.tight_layout()
	plt.show()
	plt.close()

	fig, axes = plt.subplots(nrows = 3, ncols = BATCH_SIZE, figsize = (BATCH_SIZE*5,3*5))
	for i in range(BATCH_SIZE):
		axes[0,i].imshow(nddi_prediction[i,:,:,0])
		axes[1,i].imshow(NDDI_gt[i,:,:,0])
		axes[2,i].imshow(NDDItheoreticGT[i,:,:,0])

		axes[0,i].set_title('NDDI Prediction')
		axes[1,i].set_title('NDDI Ground Truth')
		axes[2,i].set_title('NDDI Theoretic GT')

		axes[0,i].axis(False)
		axes[1,i].axis(False)
		axes[2,i].axis(False)

	fig.tight_layout()
	plt.show()
	plt.close()

	fig, axes = plt.subplots(nrows = 2, ncols = BATCH_SIZE, figsize = (BATCH_SIZE*5,3*5))
	for i in range(BATCH_SIZE):
		axes[0,i].imshow(nddi_from_ndmi_pred_and_ndvi_pred[i,:,:,0])
		axes[1,i].imshow(NDDI_gt[i,:,:,0])

		axes[0,i].set_title('NDDI Prediction from NDVI and NDWI')
		axes[1,i].set_title('NDDI Ground Truth')

		axes[0,i].axis(False)
		axes[1,i].axis(False)

	fig.tight_layout()
	plt.show()
	plt.close()


def main():
	dh = datahandler(dataset_root_path)
	train_set, val_set = dh.split(SPLIT_FACTOR)

	NDVIval2 = datareader.generatorv2('NDVI', val_set, BATCH_SIZE, T_LEN, SHAPE, normalize=NORMALIZE)
	NDWIval2 = datareader.generatorv2('NDWI',val_set, BATCH_SIZE, T_LEN, SHAPE, normalize=NORMALIZE)
	NDDIval2 = datareader.generatorv2('NDDI',val_set, BATCH_SIZE, T_LEN, SHAPE, normalize=NORMALIZE)

	NDVI_input, NDVI_gt =  NDVIval2
	NDWI_input, NDWI_gt =  NDWIval2
	NDDI_input, NDDI_gt =  NDDIval2

	if SCALE:
		NDDI_gt = scaler(NDDI_gt)

	ndwi_model = load_model(NDMI_MODEL_PATH)
	ndvi_model = load_model(NDVI_MODEL_PATH)
	nddi_model = load_model(NDDI_MODEL_PATH)

	'''
	prediction
	'''
	ndvi_prediction = ndvi_model.predict(NDVI_input)
	ndwi_prediction = ndwi_model.predict(NDWI_input)
	nddi_prediction = nddi_model.predict(NDDI_input)

	shape = nddi_prediction.shape # Batch (), Width, Height, (NDWI, NNDI or NDVI)

	nddi_from_ndmi_pred_and_ndvi_pred = []
	for i in range(shape[0]): # get NDDI from NDWI and NDVI
		ndvi = ndvi_prediction[i]
		ndwi = ndwi_prediction[i]
		nddi_from_ndmi_pred_and_ndvi_pred.append(getNDDI(ndvi,ndwi))

	NDDItheoreticGT = getNDDI(NDVI_gt,NDWI_gt)

	plot_results(NDVI_gt, ndvi_prediction, NDWI_gt, ndwi_prediction, NDDI_gt, nddi_prediction, nddi_from_ndmi_pred_and_ndvi_pred, NDDItheoreticGT)

main()