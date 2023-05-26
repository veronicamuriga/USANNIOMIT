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


dataset_root_path = '/home/veronica/USANNIOMIT/sen2dwater_combined' # TODO: adjust path
dh       = datahandler(dataset_root_path)
train_set, val_set = dh.split(SPLIT_FACTOR)


BATCH_SIZE = 30
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
                                        normalize=NORMALIZE)



# our trained ndwi model
# ndwi_model = load_model('/content/drive/MyDrive/results-lenovo/ndwi_model.h5')

# model from francesco
ndwi_model = load_model('/home/veronica/USANNIOMIT/ndwi_model2.h5')
# ndwi_model = load_model('/content/drive/MyDrive/results-lenovo2/ndwi_model2.h5')

ndvi_model = load_model('/home/veronica/USANNIOMIT/ndvi_model2.h5')
# ndvi_model = load_model('/content/drive/MyDrive/results-lenovo2/ndvi_model2.h5')

nddi_model = load_model('/home/veronica/USANNIOMIT/nddi_model.model_small_val_is_point_1.h5')

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
for i in range(shape[0]): # get NDDI from NDWI and NDVI
  ndvi = ndvi_prediction[i]
  ndwi = ndwi_prediction[i]
  nddi_from_NDVI_NDWI.append(getNDDI(ndvi,ndwi))

NDDItheoreticGT = getNDDI(NDVI_gt,NDWI_gt)

fig, axes = plt.subplots(nrows = 2, ncols = BATCH_SIZE, figsize = (BATCH_SIZE*5,3*5))
# font = {'family' : 'normal',
#         'weight' : 'bold',
#         'size'   : 22}

# matplotlib.rc('font', **font)

# plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = "8"

for i in range(BATCH_SIZE):
  axes[1,i].imshow(ndvi_prediction[i,:,:,0])
  axes[0,i].imshow(NDVI_gt[i,:,:,0])

  axes[1,i].set_title('Pred')
  axes[0,i].set_title('GT')

  axes[0,i].axis(False)
  axes[1,i].axis(False)


fig.tight_layout()
plt.savefig('img/ndvi.png')
plt.show()
plt.close()

fig, axes = plt.subplots(nrows = 2, ncols = BATCH_SIZE, figsize = (BATCH_SIZE*5,3*5))
for i in range(BATCH_SIZE):
  axes[1,i].imshow(ndwi_prediction[i,:,:,0])
  axes[0,i].imshow(NDWI_gt[i,:,:,0])

  axes[1,i].set_title('Pred')
  axes[0,i].set_title('GT')

  axes[1,i].axis(False)
  axes[0,i].axis(False)

fig.tight_layout()
plt.savefig('img/ndwi.png')
plt.show()
plt.close()

# fig, axes = plt.subplots(nrows = 3, ncols = BATCH_SIZE, figsize = (BATCH_SIZE*5,3*5))
# for i in range(BATCH_SIZE):
#   axes[0,i].imshow(nddi_prediction[i,:,:,0])
#   axes[1,i].imshow(NDDI_gt[i,:,:,0])
#   axes[2,i].imshow(NDDItheoreticGT[i,:,:,0])

#   axes[0,i].set_title('NDDI Prediction')
#   axes[1,i].set_title('NDDI Ground Truth')
#   axes[2,i].set_title('NDDI Theoretic GT')

#   axes[0,i].axis(False)
#   axes[1,i].axis(False)
#   axes[2,i].axis(False)

# fig.tight_layout()
# plt.show()
# plt.close()

fig, axes = plt.subplots(nrows = 3, ncols = BATCH_SIZE, figsize = (BATCH_SIZE*5,3*5))
for i in range(BATCH_SIZE):
#   axes[0,i].imshow(nddi_from_NDVI_NDWI[i,:,:,0])
  axes[0,i].imshow(NDDI_gt[i,:,:,0])
  axes[1,i].imshow(nddi_from_NDVI_NDWI[i])
  axes[2,i].imshow(nddi_prediction[i,:,:,0])


  axes[0,i].set_title('GT')
  axes[1,i].set_title('Computed')
  axes[2,i].set_title('Predicted')

  axes[0,i].axis(False)
  axes[1,i].axis(False)
  axes[2,i].axis(False)

fig.tight_layout()
plt.savefig('img/nddi.png')
plt.show()
plt.close()