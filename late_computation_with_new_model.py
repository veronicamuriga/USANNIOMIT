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

def calculate_mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')  

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

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

# save error metrics
ndvi_mse_arr = []
ndwi_mse_arr = []
nddi_mse_arr = []
nddi_from_ndvi_ndwi_mse_arr = []
ndvi_ssim_arr = []
ndwi_ssim_arr = []
nddi_ssim_arr = []
nddi_from_ndvi_ndwi_ssim_arr = []
ndvi_psnr_arr = []
ndwi_psnr_arr = []
nddi_psnr_arr = []
nddi_from_ndvi_ndwi_psnr_arr = []

print(min(ndvi_prediction.flatten()), max(ndvi_prediction.flatten()))

for i in range(shape[0]): # get NDDI from NDWI and NDVI
  ndvi = ndvi_prediction[i]
  ndwi = ndwi_prediction[i] 
  nddi_from_NDVI_NDWI.append(getNDDI(ndvi,ndwi))


  # get error metrics
  ndvi_mse = calculate_mse(ndvi_prediction[i], NDVI_gt[i])
  ndwi_mse = calculate_mse(ndwi_prediction[i], NDWI_gt[i])
  nddi_mse = calculate_mse(nddi_prediction[i], NDDI_gt[i])
  nddi_from_ndvi_ndwi_mse = calculate_mse(nddi_prediction[i], nddi_from_NDVI_NDWI[i])

  ndvi_ssim = calculate_ssim(ndvi_prediction[i], NDVI_gt[i])
  ndwi_ssim = calculate_ssim(ndwi_prediction[i], NDWI_gt[i])
  nddi_ssim = calculate_ssim(nddi_prediction[i], NDDI_gt[i])
  nddi_from_ndvi_ndwi_ssim = calculate_ssim(nddi_prediction[i], nddi_from_NDVI_NDWI[i])


  ndvi_psnr = calculate_psnr(ndvi_prediction[i], NDVI_gt[i])
  ndwi_psnr = calculate_psnr(ndwi_prediction[i], NDWI_gt[i])
  nddi_psnr = calculate_psnr(nddi_prediction[i], NDDI_gt[i])
  nddi_from_ndvi_ndwi_psnr = calculate_psnr(nddi_prediction[i], nddi_from_NDVI_NDWI[i])



  # print("ndvi_mse: ", ndvi_mse)
  # print("ndwi_mse: ", ndwi_mse)
  # print("nddi_mse: ", nddi_mse)
  # print("nddi_from_ndvi_ndwi_mse: ", nddi_from_ndvi_ndwi_mse)
  # print("ndvi_ssim :", ndvi_ssim)
  # print("ndwi_ssim: ", ndwi_ssim)
  # print("nddi_ssim: ", nddi_ssim)
  # print("nddi_from_ndvi_ndwi_ssim :", nddi_from_ndvi_ndwi_ssim)
  # print("ndvi_psnr: ", ndvi_psnr)
  # print("ndwi_psnr: ", ndwi_psnr)
  # print("nddi_psnr: ", nddi_psnr)
  # print("nddi_from_ndvi_ndwi_psnr: ", nddi_from_ndvi_ndwi_psnr)

  ndvi_mse_arr.append(ndvi_mse)
  ndwi_mse_arr.append(ndwi_mse)
  nddi_mse_arr.append(nddi_mse)
  nddi_from_ndvi_ndwi_mse_arr.append(nddi_from_ndvi_ndwi_mse)
  ndvi_ssim_arr.append(ndvi_ssim)
  ndwi_ssim_arr.append(ndwi_ssim)
  nddi_ssim_arr.append(nddi_ssim)
  nddi_from_ndvi_ndwi_ssim_arr.append(nddi_from_ndvi_ndwi_ssim)
  ndvi_psnr_arr.append(ndvi_psnr)
  ndwi_psnr_arr.append(ndwi_psnr)
  nddi_psnr_arr.append(nddi_psnr)
  nddi_from_ndvi_ndwi_psnr_arr.append(nddi_from_ndvi_ndwi_psnr)

print("ndvi_mse: ", np.mean(ndvi_mse_arr))
print("ndwi_mse: ", np.mean(ndwi_mse))
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
print("ndvi_psnr: ", ndvi_psnr_arr)
print("ndwi_psnr: ", ndwi_psnr_arr)
print("nddi_psnr: ", nddi_psnr_arr)
print("nddi_from_ndvi_ndwi_psnr: ", nddi_from_ndvi_ndwi_psnr_arr)


# NDDItheoreticGT = getNDDI(NDVI_gt,NDWI_gt)

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


