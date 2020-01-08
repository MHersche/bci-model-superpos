#Copyright (c) 2019 ETH Zurich, Michael Hersche

import keras
from eegnet_models import EEGNet
from eegnet_helper import prediction_vector, get_metrics_and_plots,plot_retr_accuracy
from eegnet_helper import metrics_to_csv

import tensorflow as tf
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold
import time
from scipy.linalg import circulant
from get_data import get_data
from keras import backend as K 


from keras.optimizers import Adam
from keras.losses import binary_crossentropy, categorical_crossentropy
from keras.utils import np_utils
from keras.models import load_model
import time
import pdb

class_names = ['Left hand', 'Right hand',
			   'Both Feet', 'Tongue']

#######################################################################################################################
#######################################################################################################################
def train_subject_specific(ch_num=22, class_names=class_names, addon='',model_path = './initialModels/',data_path= './dataset/',
				result_path='', dropoutRate=0.25,lr = 0.001,batch_size=32,dropoutType='Dropout',kernLength=64,numFilters = 8,
				crossValidation=False, epochs = 500):
	
	n_subjects = 9
	if crossValidation:
		nFolds = 5
	else: 
		nFolds = 1

	metrics = np.zeros((n_subjects,nFolds,4))
	
	for subj in range(n_subjects):
		X_train,_,y_train_onehot,X_test,_,y_test_onehot = prepare_features(data_path,subj,crossValidation)	

		for fold in range(nFolds):
		
			model = EEGNet(nb_classes=len(class_names), Chans=ch_num, 
				Samples=1125, dropoutRate=dropoutRate, dropoutType=dropoutType,
				kernLength=kernLength,numFilters = numFilters)

			model.compile(loss=categorical_crossentropy,
						  optimizer=Adam(lr=lr), metrics=['accuracy'])

			filename_ = '{0}{1}{2}'.format(result_path,'standard_user_{}'.format(subj + 1), addon)
			#pdb.set_trace()
			metrics[subj,fold] = standard_unit(model, X_train[fold], y_train_onehot[fold],
									   X_test[fold], y_test_onehot[fold], filename_,
									   class_names, epochs = epochs)

		print("Subject {:} \t {:.4f}".format(subj+1,np.mean(metrics[subj,:,0])))
		# save one model per subject: in xVal and testing
		model.save(model_path+'model{:}.h5'.format(subj))

	print("Average Accuracy {:.4f}".format(metrics[:,:,0].mean()))
	metrics_to_csv(np.mean(metrics,axis=1), '{}Standard_testing_{}'.format(result_path, addon))


def train_superposition_eegnet(model_path='./EEGnet/initialModels/',save_path='./EEGnet/superposModels/',Niter = 1000,epochs=5, batch_size = 64, 
                                ch_size=64,lr=1e-4,verbose=1,result_path='',addon='', data_path = './dataset/',crossValidation=False): 
	
	dim_in=272
	dim_out = 4
	d = dim_in*dim_out
	ch_num = 22
	n_subjects = 9

	X_train,y_train,y_train_onehot,X_test,y_test,y_test_onehot = prepare_features_allsub(data_path,n_subjects,crossValidation)

	if crossValidation:
		nFolds=5
	else:
		nFolds=1

	if verbose: 
		print("Initial superpostion of model weights")
	_,S, keys = superposition(model_path,n_subjects,dim_in)

	order = np.arange(n_subjects)
	for fold in range(nFolds):
		print("Fold {} of {}".format(fold+1,nFolds))
		metrics = np.zeros((n_subjects,Niter,2))
		
		for itr in range(Niter):
			K.clear_session()
			if itr ==1: 
				model_path = save_path

			print("iteration: "+str(itr))
			np.random.shuffle(order)
			t1 = time.time()
			for sub in order:
				
				# retrieve superimpoesed weights 
				weights_retr = retrieve(S,keys[sub])
				
				model=keras.models.load_model(model_path+'model{:}.h5'.format(sub))
				# set new weights 
				mod_setweights(model,weights_retr,dim_in,dim_out)
				# recompile and retrain model 
				model.compile(loss=categorical_crossentropy,
							  optimizer=Adam(lr = lr), metrics=['accuracy'])

				# test accuracy of model
				metrics[sub,itr] = train_test_acc_new(model,X_train[sub][fold], 
					y_train[sub][fold],X_test[sub][fold], y_test[sub][fold])
				

				_ = model.fit(X_train[sub][fold], y_train_onehot[sub][fold],
							batch_size=batch_size, epochs=epochs, verbose=0)
				
				# get new weights 
				weights_new = mod_getweights(model,d)
				S = S + bind(weights_new-weights_retr,keys[sub])

				model.save(save_path+'model{:}.h5'.format(sub))

			if verbose: 
				print("Iter {:}, \t Training: {:.3f} \t Testing: {:.3f}".format(itr,metrics[:,itr,0].mean(),
					metrics[:,itr,1].mean()))
				t2 = time.time()
				print(t2-t1)
			
			if itr%10==0:
				metrics_to_csv(metrics[:,:,0], '{}Superposition_testing_fold{}_{}_training'.format(result_path,fold,addon))
				metrics_to_csv(metrics[:,:,1], '{}Superposition_testing_fold{}_{}_testing'.format(result_path,fold, addon))
				np.savez(save_path+'superpos_fold{}'.format(fold),S=S,keys=keys)



######################################### Global ###########################################################
def global_all(ch_num=22, class_names=class_names, addon='',data_path= './dataset/',
				result_path='', dropoutRate=0.25,lr = 0.001,batch_size=32,dropoutType='Dropout',
				kernLength=64,numFilters = 8, epochs = 500):
	
	
	n_subjects = 9
	metrics = np.zeros((n_subjects, 2))

	X_train,y_train,X_test,y_test= load_global_data(data_path,n_subjects)

	model = EEGNet(nb_classes=len(class_names), Chans=ch_num, 
			Samples=1125, dropoutRate=dropoutRate, dropoutType=dropoutType,
			kernLength=kernLength,numFilters = numFilters)

	model.compile(loss=categorical_crossentropy,
					  optimizer=Adam(lr=lr), metrics=['accuracy'])

	
	_ = model.fit(X_train, y_train,
						batch_size=batch_size, epochs=epochs, verbose=0)

	
	for sub in range(n_subjects):

		metrics[sub] = train_test_acc(model,X_train,y_train,X_test[sub],y_test[sub])
		
		print("Subject {:} \t {:.4f}".format(sub+1,metrics[sub,1]))


	print("Average Accuracy {:.4f}".format(metrics[:,1].mean()))


############################################################################################################	

def standard_unit(model, training_data, training_labels,
				  testing_data, testing_labels, filename,
				  class_names=class_names, epochs = 10,batch_size=32):
	history = model.fit(training_data, training_labels,
						batch_size=batch_size, epochs=epochs, verbose=0,
						validation_data=(testing_data, testing_labels))
	y_prob = model.predict(testing_data)
	values = get_metrics_and_plots(testing_labels, y_prob, history,
								   filename, class_names)

	return values

def train_test_acc(model,training_data, training_labels,
				  testing_data, testing_labels): 
	

	y_pred = model.predict(testing_data).argmax(axis=-1)
	test_acc = np.sum(np.reshape(np.array(testing_labels)-1,-1)==y_pred)/len(y_pred)

	y_pred = model.predict(training_data).argmax(axis=-1)
	train_acc = np.sum(np.reshape(np.array(training_labels)-1,-1)==y_pred)/len(y_pred)
	return [train_acc,test_acc]


	
def superposition(model_path,n_subjects,dim_in): 
	
	dim_out = 4
	d = dim_in*dim_out
	
	keys = 1/np.sqrt(d)*np.random.randn(n_subjects,d)
	S = np.zeros(d)
	models = []
	for sub in range(n_subjects): 
		models.append(keras.models.load_model(model_path+'model{:}.h5'.format(sub)))
		weights = models[sub].get_weights()[-2]
		weights = np.reshape(weights,d)
		
		S = S + bind(weights,keys[sub])
	
	return models,S,keys

def load_superposition(model_path): 
	
	data = np.load(model_path+'superpos.npz')
	return data['S'],data['keys']
		

def mod_getweights(model,d): 
	weights = model.get_weights()[-2]
	weights = np.reshape(weights,d)
	return weights

def mod_setweights(model,X,dim_in,dim_out): 
	
	all_weights = model.get_weights()
	
	all_weights[-2]= X.reshape((dim_in,dim_out))
	model.set_weights(all_weights)
	return model

def bind(X,K): 

	C = circulant(K)
	return np.matmul(C,X)

def retrieve(S,K):
	 C = circulant(K)
	 return np.matmul(C.T,S)
def train_test_acc_new(model,training_data, training_labels,
				  testing_data, testing_labels): 
	

	y_pred = model.predict(testing_data).argmax(axis=-1)
	test_acc = float(np.sum((testing_labels-1)==y_pred))/len(y_pred)

	y_pred = model.predict(training_data).argmax(axis=-1)
	train_acc = float(np.sum((training_labels-1)==y_pred))/len(y_pred)
	return [train_acc,test_acc]


######################################## Data loading functions #########################################################

# Feature loading with new dataloading procedure 
def prepare_features(path,subject,crossValidation):

	fs = 250 
	t1 = int(1.5*fs)
	t2 = int(6*fs)
	T = t2-t1
	# load data
	X_train, y_train = get_data(subject+1,True,path)

	X_tr = []; y_tr = []; X_tst= []; y_tst= []; y_tr_onehot=[]; y_tst_onehot=[]
	if crossValidation: 
		nFold = 5
		kf = KFold(n_splits=nFold)
		for train_index, test_index in kf.split(X_train):
			X_tr.append(X_train[train_index])
			y_tr.append(y_train[train_index])
			X_tst.append(X_train[test_index])
			y_tst.append(y_train[test_index])
	else:
		nFold=1
		X_tr.append(X_train)
		y_tr.append(y_train)
		X_test, y_test = get_data(subject+1,False , path)
		X_tst.append(X_test)
		y_tst.append(y_test)
	
	# prepare training data 	
	for fold in range(nFold):
		N_tr,N_ch,_ =X_tr[fold].shape 
		N_test,_,_ =X_tst[fold].shape 

		X_tr[fold] = X_tr[fold][:,:,t1:t2].reshape(N_tr,1,N_ch,T)
		y_tr_onehot.append((y_tr[fold]-1).astype(int))
		y_tr_onehot[fold] = np_utils.to_categorical(y_tr_onehot[fold])
		# prepare testing data 
		
		X_tst[fold] = X_tst[fold][:,:,t1:t2].reshape(N_test,1,N_ch,T)
		y_tst_onehot.append((y_tst[fold]-1).astype(int))
		#pdb.set_trace()
		y_tst_onehot[fold] = np_utils.to_categorical(y_tst_onehot[fold])	

	return X_tr,y_tr,y_tr_onehot,X_tst,y_tst,y_tst_onehot

def prepare_features_allsub(path,n_subjects,crossValidation=False): 

	# prepare training and testing data 
	X_train = []; y_train = [];	y_train_onehot=[]

	X_test = []; y_test = []; y_test_onehot =[]

	for sub in range(n_subjects):
		X_tr,y_tr,y_tr_onehot,X_tst,y_tst,y_tst_onehot = prepare_features(
			path,sub,crossValidation)
		
		X_train.append(X_tr)
		y_train.append(y_tr)
		y_train_onehot.append(y_tr_onehot)
		X_test.append(X_tst)
		y_test.append(y_tst)
		y_test_onehot.append(y_tst_onehot)

	
	return X_train,y_train,y_train_onehot,X_test,y_test,y_test_onehot


def load_global_data(path,n_subjects):

	# prepare training and testing data 
	X_train = np.zeros([0,1,22,1125])
	y_train = np.zeros([0,4])

	X_test = []
	y_test = []

	for sub in range(n_subjects):
		X_tr,_,y_tr_onehot,X_tst,y_tst,_ = prepare_features(path,sub,crossValidation=False)
		# append training
		X_train=np.append(X_train,X_tr[0],axis=0)
		y_train= np.append(y_train,y_tr_onehot[0],axis=0)
		# append testing 
		X_test.append(X_tst[0])
		#y_tst = np.argmax(y_tst[0],axis=-1)+1

		y_test.append(y_tst[0])


	#y_test = np.argmax(y_test,axis=-1)
	#print(y_test)
	return X_train,y_train,X_test,y_test












