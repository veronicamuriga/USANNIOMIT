#!/usr/bin/env python
# coding: utf-8

from dataio.datahandler import datahandler
from dataio.datareader import datareader
from models.TDCNNv1 import TDCNNv1
from config import *
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.models import load_model
import sklearn.preprocessing
# %matplotlib qt 
import cv2
import math
import tensorflow as tf
from tensorflow.keras.callbacks import Callback



dataset_root_path = '/home/veronica/USANNIOMIT/sen2dwater_combined' # TODO: adjust path
dh       = datahandler(dataset_root_path)
train_set, val_set = dh.split(SPLIT_FACTOR)


BATCH_SIZE = 100
SHAPE      = (64,64)
NORMALIZE  = True


NDVIval2 = datareader.generatorv2('NDVI',val_set,
                                        BATCH_SIZE,
                                        T_LEN,
                                        SHAPE,
                                        normalize=NORMALIZE)
NDWIval2 = datareader.generatorv2('NDWI',val_set,
                                        BATCH_SIZE,
                                        T_LEN,
                                        SHAPE,
                                        normalize=NORMALIZE)
NDDIval2 = datareader.generatorv2('NDDI',val_set,
                                        BATCH_SIZE,
                                        T_LEN,
                                        SHAPE,
                                        normalize=False)


# method 2 of calculating error metrics
def error_metrics_using_tb(y_in, y_pr):
  psnr, ssim, mse = [], [], []
  for i in range(y_in.shape[0]):
    x = tf.cast(y_in[i,...], tf.float32)
    y = tf.cast(y_pr[i,...], tf.float32)
    m1 = tf.reduce_mean(tf.image.psnr(x, y, 2.0))
    m2 = tf.reduce_mean(tf.image.ssim(x, y, 2.0))
    m3 = tf.reduce_mean(tf.keras.metrics.mean_squared_error(x, y))

    psnr.append(m1.numpy())
    ssim.append(m2.numpy())
    mse.append(m3.numpy())

  return psnr, ssim, mse 


# our trained ndwi model
# ndwi_model = load_model('/content/drive/MyDrive/results-lenovo/ndwi_model.h5')

# model from francesco
ndwi_model = load_model('/home/veronica/USANNIOMIT/ndwi_model2.h5')
# ndwi_model = load_model('/content/drive/MyDrive/results-lenovo2/ndwi_model2.h5')

ndvi_model = load_model('/home/veronica/USANNIOMIT/ndvi_model2.h5')
# ndvi_model = load_model('/content/drive/MyDrive/results-lenovo2/ndvi_model2.h5')

# nddi_model = load_model('/home/veronica/USANNIOMIT/nddi_model.model_small_val_is_point_1.h5')
nddi_model = load_model('/home/veronica/USANNIOMIT/nddi_model.model.h5')


NDVI_input, NDVI_gt =  NDVIval2
NDWI_input, NDWI_gt =  NDWIval2
NDDI_input, NDDI_gt =  NDDIval2

ndvi_prediction = ndvi_model.predict(NDVI_input)
ndwi_prediction = ndwi_model.predict(NDWI_input)
nddi_prediction = nddi_model.predict(NDDI_input)

shape = nddi_prediction.shape # Batch (), Width, Height, (NDWI, NNDI or NDVI)

def getNDDI(NDVI,NDWI):
  return (NDVI-NDWI)/(NDVI+NDWI+1)

nddi_from_NDVI_NDWI = []


print(min(ndvi_prediction.flatten()), max(ndvi_prediction.flatten()))

for i in range(shape[0]): # get NDDI from NDWI and NDVI
  ndvi = ndvi_prediction[i]
  ndwi = ndwi_prediction[i] 
  nddi_from_NDVI_NDWI.append(getNDDI(ndvi,ndwi))

# method 2 of error calculations
ndvi_psnr_arr, ndvi_ssim_arr, ndvi_mse_arr = error_metrics_using_tb(NDVI_gt, ndvi_prediction)
ndwi_psnr_arr, ndwi_ssim_arr, ndwi_mse_arr = error_metrics_using_tb(NDWI_gt, ndwi_prediction)
nddi_psnr_arr, nddi_ssim_arr, nddi_mse_arr = error_metrics_using_tb(NDDI_gt, nddi_prediction)
nddi_from_ndvi_ndwi_psnr_arr, nddi_from_ndvi_ndwi_ssim_arr, nddi_from_ndvi_ndwi_mse_arr = error_metrics_using_tb(NDDI_gt, np.array(nddi_from_NDVI_NDWI))


print("ndvi_mse: ", np.mean(ndvi_mse_arr))
print("ndwi_mse: ", np.mean(ndwi_mse_arr))
print("nddi_mse: ", np.mean(nddi_mse_arr))
print("nddi_from_ndvi_ndwi_mse: ", np.mean(nddi_from_ndvi_ndwi_mse_arr))
print("ndvi_ssim :", np.mean(ndvi_ssim_arr))
print("ndwi_ssim: ", np.mean(ndwi_ssim_arr))
print("nddi_ssim: ", np.mean(nddi_ssim_arr))
print("nddi_from_ndvi_ndwi_ssim :", np.mean(nddi_from_ndvi_ndwi_ssim_arr))
print("ndvi_psnr: ", np.mean(ndvi_psnr_arr))
print("ndwi_psnr: ", np.mean(ndwi_psnr_arr))
print("nddi_psnr: ", np.mean(nddi_psnr_arr))
print("nddi_from_ndvi_ndwi_psnr: ", np.mean(nddi_from_ndvi_ndwi_psnr_arr))

fig, axes = plt.subplots(nrows = 2, ncols = BATCH_SIZE, figsize = (BATCH_SIZE*5,3*5))
plt.rcParams["font.size"] = "8"

for i in range(BATCH_SIZE):
  axes[1,i].imshow(ndvi_prediction[i,:,:,0])
  axes[0,i].imshow(NDVI_gt[i,:,:,0])

  axes[1,i].set_title('Pred')
  axes[0,i].set_title('GT')

  axes[0,i].axis(False)
  axes[1,i].axis(False)


# fig.tight_layout()
plt.savefig('img/ndvi.png', transparent = True)
# plt.show()
# plt.close()

fig, axes = plt.subplots(nrows = 2, ncols = BATCH_SIZE, figsize = (BATCH_SIZE*5,3*5))

for i in range(BATCH_SIZE):

  axes[1,i].imshow(ndwi_prediction[i,:,:,0])
  axes[0,i].imshow(NDWI_gt[i,:,:,0])

  axes[1,i].set_title('Pred')
  axes[0,i].set_title('GT')

  axes[1,i].axis(False)
  axes[0,i].axis(False)

# fig.tight_layout()
plt.savefig('img/ndwi.png', transparent = True)
# plt.show()
# plt.close()

fig, axes = plt.subplots(nrows = 3, ncols = BATCH_SIZE, figsize = (BATCH_SIZE*5,3*5))
for i in range(BATCH_SIZE):
#   axes[0,i].imshow(nddi_from_NDVI_NDWI[i,:,:,0])
  axes[0,i].imshow(NDDI_gt[i,:,:,0])
  axes[1,i].imshow(nddi_from_NDVI_NDWI[i])
  axes[2,i].imshow(nddi_prediction[i,:,:,0])

  # axes[0,i].set_title('GT')
  # axes[1,i].set_title('Computed')
  # axes[2,i].set_title('Predicted')

  axes[0,i].axis(False)
  axes[1,i].axis(False)
  axes[2,i].axis(False)

# fig.tight_layout()
plt.savefig('img/nddi.png', transparent = True)
# plt.show()
# plt.close()
