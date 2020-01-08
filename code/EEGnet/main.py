#Copyright (c) 2019 ETH Zurich, Michael Hersche


import matplotlib

matplotlib.use('Agg')
from eegnet_controllers import train_superposition_eegnet,train_subject_specific,global_all

import glob as glob

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


###########################################################################
###########################################################################
crossValidation = False

PATH = '/usr/scratch/xavier/herschmi/HD_superpos/'
model_path = PATH+'models/EEGNet/initialModels/'
save_path = PATH+'models/EEGNet/superposModels/'
data_path = PATH+'dataset/IV2a/'    
result_path=PATH+'results/EEGNet/'
lr = 1e-4
batch_size = 64
Niter = 1000
epochs = 5

addon = 'lr={}_bs={}_Niter={}_epochs={}'.format(lr,batch_size, Niter, epochs)

# global model 
# global_all(data_path=data_path)

# train subject specific model and save it 
train_subject_specific(model_path=model_path,data_path=data_path,result_path=result_path,crossValidation=crossValidation)

train_superposition_eegnet(lr=lr,batch_size=batch_size, epochs = epochs, save_path=save_path,
 	model_path=model_path, data_path=data_path,result_path=result_path, addon=addon,crossValidation=crossValidation)

    
