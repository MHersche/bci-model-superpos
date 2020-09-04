#*----------------------------------------------------------------------------*
#* Copyright (C) 2019 ETH Zurich, Switzerland                                 *
#* SPDX-License-Identifier: Apache-2.0                                        *
#*                                                                            *
#* Licensed under the Apache License, Version 2.0 (the "License");            *
#* you may not use this file except in compliance with the License.           *
#* You may obtain a copy of the License at                                    *
#*                                                                            *
#* http://www.apache.org/licenses/LICENSE-2.0                                 *
#*                                                                            *
#* Unless required by applicable law or agreed to in writing, software        *
#* distributed under the License is distributed on an "AS IS" BASIS,          *
#* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
#* See the License for the specific language governing permissions and        *
#* limitations under the License.                                             *
#*                                                                            *
#* Authors: Michael Hersche, Philipp Rupp          							  *
#*----------------------------------------------------------------------------*

import matplotlib
from convnet_helper import train_subject_specific,train_superposition,validate_subject_specific
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"



crossValidation = False

PATH = '/usr/scratch/xavier/herschmi/HD_superpos/'
model_path = PATH+'models/ShallowConvnet/initialModels/' # storing/loding subject-specific models
model_sup_path = PATH+'models/ShallowConvnet/superposModels/' #stotring/loading superimposed models 
result_path = PATH+'results/ShallowConvNet/' # numerical results 
data_path = PATH+'dataset/IV2a_braindecode/' # data


# Training subject specific 
lr = 1e-3
batch_size = 64
epochs = 500
addon = 'lr={}_bs={}_epochs={}'.format(lr,batch_size, epochs)

train_subject_specific(addon=addon,model_path =model_path,data_path= data_path,
                 result_path=result_path,lr = lr,batch_size=batch_size,crossValidation=crossValidation, 
                 epochs = epochs,cuda = True,earlyStopping=True)

validate_subject_specific(addon='',model_path =model_path ,data_path= data_path,
				result_path='')
# Training superposition
compressedLayers = 'conv_class' # {'conv_class', 'conv_class_spat', 'conv_spat'}
lr = 1e-4
batch_size = 64
epochs = 5
Niter = 1000
addon = 'lr={}_bs={}_Niter={}_epochs={}_comprL={}'.format(lr,batch_size,Niter, epochs,compressedLayers)
train_superposition(Niter = Niter,epochs=5, batch_size=64,lr=1e-4, crossValidation = crossValidation,
                             addon=addon, compressedLayers=compressedLayers, result_path=result_path,
                              data_path=data_path, model_path = model_path, model_sup_path=model_sup_path)
