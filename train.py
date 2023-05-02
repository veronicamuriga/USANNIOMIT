# -*- coding: utf-8 -*-


#####################################################################################################
#                         Define Paths for Files, Dataset, and Save                                 #
#####################################################################################################


# PATH_TO_DATASET = 'INSERT'
# path_save_NDWI_results = 'INSERT'
# path_save_NDVI_results = 'INSERT'
# path_save_NDDI_results = 'INSERT'

PATH_TO_DATASET = '/home/veronica/USANNIOMIT/sen2dwater_combined'
path_save_NDWI_results = '/home/veronica/USANNIOMIT/ndwi'
path_save_NDVI_results = '/home/veronica/USANNIOMIT/ndvi'
path_save_NDDI_results = '/home/veronica/USANNIOMIT/nddi'



#####################################################################################################
#                     Import Files and Dataset, Load and Split Dataset                              #
#####################################################################################################


from dataio.datahandler import datahandler
from dataio.datareader import datareader

from dataio.datahandler import datahandler
from dataio.datareader import datareader

from models.TDCNNv1 import TDCNNv1

# from utils.plot_dataset import plot_series 

from config import *

import matplotlib.pyplot as plt
import os

#======================================= Loading the dataset ========================================
dataset_root_path = PATH_TO_DATASET

dh       = datahandler(dataset_root_path)
keys     = list(dh.paths.keys())
t_len    = len(dh.paths[keys[0]])

print('{:=^100}'.format(' Loading the dataset '))
print('\t -{:<50s} {}'.format('Number of GeoLocation', len(keys)))
print('\t -{:<50s} {}'.format('Number of Images per GeoLocation', t_len))

#========================================== Split dataset ===========================================
train_set, val_set = dh.split(SPLIT_FACTOR)
print('{:=^100}'.format(' Splitting the dataset '))
print('\t -{:<50s} {}'.format('Number of GeoLocation (training)', len(train_set.keys())))
print('\t -{:<50s} {}'.format('Number of GeoLocation (validation)', len(val_set.keys())))


#####################################################################################################
#                                   Time Distributed CNN                                            #
#####################################################################################################

#========================================== Build model =============================================
tdcnnv1 = TDCNNv1(shape=(T_LEN, PATCH_WIDTH, PATCH_HEIGHT, BANDS))
print('{:=^100}'.format(' Building the Model '))
print(tdcnnv1.model.summary())
#========================================== Train model =============================================
print('{:=^100}'.format(' Training the Model '))
# history = tdcnnv1.train(train_set, val_set, normalize = True)

tdcnnv1.generateData(train_set, val_set)

# history = tdcnnv1.train(train_set, val_set, normalize = True)
# tdcnnv1.model.save('/content/drive/MyDrive/sen2dwater/USANNIOMIT/tdcnnv1.model.h5')

# ================================ train on NDWI ==================================  #
ndwi_history, ndwi_model = tdcnnv1.train('NDWI', train_set, val_set, normalize = True)
ndwi_model.save(path_save_NDWI_results)

# ================================ train on NDVI ==================================  #
ndvi_history, ndvi_model = tdcnnv1.train('NDVI', train_set, val_set, normalize = True)
ndvi_model.save(path_save_NDVI_results)

# ================================ train on NDDI ==================================  #
nddi_history, nddi_model = tdcnnv1.train('NDDI', train_set, val_set, normalize = True)
nddi_model.save(path_save_NDDI_results)


#####################################################################################################
#                                 Plot and Visualize Results                                        #
#####################################################################################################


# might error....
LEN_SERIES = t_len
fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (4*LEN_SERIES, 4))

axes[0].plot(tdcnnv1.model.history.loss, label = 'training loss')
axes[0].plot(tdcnnv1.model.history['val_loss'], label = 'validation loss')
axes[0].set_title('Huber Loss')
axes[0].legend()

axes[1].plot(tdcnnv1.model.history['mae'], label = 'training MAE')
axes[1].plot(tdcnnv1.model.history['val_mae'], label = 'validation MAE')
axes[1].set_title('MAE')
axes[1].legend()

axes[2].plot(tdcnnv1.model.history['mse'], label = 'training MSE')
axes[2].plot(tdcnnv1.model.history['val_mse'], label = 'validation MSE')
axes[2].set_title('MSE')
axes[2].legend()
plt.savefig(name + 'Training.pdf')
plt.show()

