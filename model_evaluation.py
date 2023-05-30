#!/usr/bin/env python
# coding: utf-8

from dataio.datahandler import datahandler
from dataio.datareader import datareader
from models.TDCNNv1 import TDCNNv1
from config import *

import matplotlib.pyplot as plt
import os

from tensorflow.keras.models import load_model
import sklearn.preprocessing

def plot_metrics(model, name = 'index'):
    dataset_root_path = '/home/veronica/SEN2DWATER'

    dh = datahandler(dataset_root_path)
    keys = list(dh.paths.keys())
    t_len = len(dh.paths[keys[0]])
    
    print("datahandler done")
    model.evaluate()

    LEN_SERIES = t_len
    fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (4*LEN_SERIES, 4))

    axes[0].plot(model.loss, label = 'training loss')
    axes[0].plot(model.history['val_loss'], label = 'validation loss')
    axes[0].set_title('Huber Loss')
    axes[0].legend()

    axes[1].plot(model.history['mae'], label = 'training MAE')
    axes[1].plot(model.history['val_mae'], label = 'validation MAE')
    axes[1].set_title('MAE')
    axes[1].legend()

    axes[2].plot(model.history['mse'], label = 'training MSE')
    axes[2].plot(model.history['val_mse'], label = 'validation MSE')
    axes[2].set_title('MSE')
    axes[2].legend()
    plt.savefig('img/' + name + 'Training.pdf')
    plt.show()


# import trained models 
# ndwi_model = load_model('/content/drive/MyDrive/sen2dwater/USANNIOMIT/ndwi_model2.h5')
ndwi_model = load_model('/home/veronica/USANNIOMIT/ndwi_model2.h5')

# ndvi_model = load_model('//content/drive/MyDrive/sen2dwater/USANNIOMIT/ndvi_model2.h5')
ndvi_model = load_model('/home/veronica/USANNIOMIT/ndvi_model2.h5')

# nddi_model = load_model('/content/drive/MyDrive/sen2dwater/USANNIOMIT/nddi_model2.h5')
# nddi_model = load_model('/content/drive/MyDrive/results-lenovo2/nddi_model2.h5')
nddi_model = load_model('/home/veronica/USANNIOMIT/nddi_model.model_small_val_is_point_1.h5')

plot_metrics(model = nddi_model, name = 'nddi')
plot_metrics(model = nddi_model, name = 'nddi')
plot_metrics(model = nddi_model, name = 'nddi')

# #======================================= Loading the dataset ========================================
# # dataset_root_path = '/content/drive/MyDrive/sen2dwater/USANNIOMIT/DATASET_2016_2022'
# dataset_root_path = '/home/veronica/SEN2DWATER'

# dh       = datahandler(dataset_root_path)
# keys     = list(dh.paths.keys())
# t_len    = len(dh.paths[keys[0]])

# print('{:=^100}'.format(' Loading the dataset '))
# print('\t -{:<50s} {}'.format('Number of GeoLocation', len(keys)))
# print('\t -{:<50s} {}'.format('Number of Images per GeoLocation', t_len))

# #========================================== Split dataset ===========================================
# train_set, val_set = dh.split(SPLIT_FACTOR)
# print('{:=^100}'.format(' Splitting the dataset '))
# print('\t -{:<50s} {}'.format('Number of GeoLocation (training)', len(train_set.keys())))
# print('\t -{:<50s} {}'.format('Number of GeoLocation (validation)', len(val_set.keys())))

# tdcnnv1 = TDCNNv1(shape=(T_LEN, PATCH_WIDTH, PATCH_HEIGHT, BANDS))
# print('{:=^100}'.format(' Building the Model '))
# print(tdcnnv1.model.summary())
# #========================================== Train model =============================================
# print('{:=^100}'.format(' Training the Model '))
# # history = tdcnnv1.train(train_set, val_set, normalize = True)

# tdcnnv1.generateData(train_set, val_set)
