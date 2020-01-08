#Copyright (c) 2019 ETH Zurich, Michael Hersche

import logging
import os.path
from collections import OrderedDict
import io, sys
import warnings

import numpy as np
import torch.nn.functional as F
from torch import optim
import csv
import random
from bcic_iv_2a import BCICompetition4Set2A
from braindecode.experiments.experiment import Experiment
from braindecode.experiments.monitors import LossMonitor, MisclassMonitor, \
	RuntimeMonitor
from braindecode.experiments.stopcriteria import MaxEpochs, NoDecrease, Or
from braindecode.datautil.iterators import BalancedBatchSizeIterator
from braindecode.datautil.splitters import split_into_train_test
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.torch_ext.constraints import MaxNormDefaultConstraint
from braindecode.torch_ext.util import set_random_seeds
from braindecode.mne_ext.signalproc import mne_apply
from braindecode.datautil.signalproc import (bandpass_cnt,
											 exponential_running_standardize)
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne
from braindecode.datautil.splitters import concatenate_sets

import torch as th
from scipy.linalg import circulant, toeplitz
import pdb
import copy
import itertools
import math

log = logging.getLogger(__name__)



'''
the following function is copied and modified from the below referenced source code.
The corresponding paper is 
	
R. T. Schirrmeister,  J. T. Springenberg,  L. D. J. Fiederer,  M. Glasstetter,
K.  Eggensperger,  M.  Tangermann,  F.  Hutter,  W.  Burgard,  and  T.  Ball,
“Deep  learning  with  convolutional  neural  networks  for  EEG  decoding  and
visualization,” mar 2017. [Online]. Available: http://arxiv.org/abs/1703.05051http:
//dx.doi.org/10.1002/hbm.23730
***************************************************************************************
*    Title: braindecode
*    Code: examples/bcic_iv_2a.py
*    Author: R. Schirrmeister et al.
*    Date: 03.2019
*    Availability: https://github.com/TNTLFreiburg/braindecode
*
***************************************************************************************
'''

def train_subject_specific(addon='',model_path = './initialModels/',data_path= './dataset/',
				result_path='',lr = 0.001,batch_size=64,crossValidation=False, epochs = 10,cuda = False,earlyStopping=True):
	
	
	max_increase_epochs = 160
	n_classes = 4
	numberOfsubjects = 9
	n_folds = 5 if crossValidation else 1
	test_mode = 'Xval' if crossValidation else 'test' 
	metrics = np.zeros((numberOfsubjects,n_folds,3))

	print(test_mode)
	print('Train \t Val \t Test')
	for sub in range(numberOfsubjects):
		# do folding if XVal, else 
		print("Sub {:}".format(sub+1))
		for fold in range(n_folds):

			# load data 
			train_set, valid_set, test_set = loadData(data_path, sub+1, fold)

			# init Shallow Model 
			set_random_seeds(seed=20190706, cuda=cuda)
			n_chans = int(train_set.X.shape[1])
			input_time_length = train_set.X.shape[2]
			model = ShallowFBCSPNet(n_chans, n_classes, input_time_length=input_time_length,
								final_conv_length='auto').create_network()
			if cuda:
				model.cuda()

			log.info("Model: \n{:s}".format(str(model)))
			optimizer = optim.Adam(model.parameters(),lr=lr)
		
			iterator = BalancedBatchSizeIterator(batch_size=batch_size)
			
			# stop criterion:  otherwise epochs and early stopping
			if crossValidation:
				# XVal only do epochs without early stopping
				stop_criterion = MaxEpochs(epochs)
				earlyStopping = False 
			else: 
				stop_criterion = Or([MaxEpochs(epochs),
								 NoDecrease('valid_misclass', max_increase_epochs)])
			


			monitors = [LossMonitor(), MisclassMonitor(), RuntimeMonitor()]
		
			model_constraint = MaxNormDefaultConstraint()
			
			# run experiment 
			exp = Experiment(model, train_set, valid_set, test_set, iterator=iterator,
							 loss_function=F.nll_loss, optimizer=optimizer,
							 model_constraint=model_constraint,
							 monitors=monitors,
							 stop_criterion=stop_criterion,
							 remember_best_column='valid_misclass',
							 run_after_early_stop=earlyStopping, cuda=cuda)
			#print("Start training")
			exp.run()
			
			# save accuracy
			metrics[sub,fold,0] = (1-exp.epochs_df.train_misclass.values)[-1]
			metrics[sub,fold,1] = (1-exp.epochs_df.valid_misclass.values)[-1]
			metrics[sub,fold,2]  = (1-exp.epochs_df.test_misclass.values)[-1]

			
			# save and delete model 
			th.save(exp,model_path+'model_subject{:}{:}{:}.pt'.format(test_mode,sub+1,fold))
			del exp
			print('\t{:.3} \t {:.3} \t {:.3}'.format(np.mean(metrics[sub,fold,0]),np.mean(metrics[sub,fold,1]), np.mean(metrics[sub,fold,2])))

	# average folds if Xval
	#metrics = np.mean(metrics, axis=1)
	print("Average")
	print('\t{:.3} \t {:.3} \t {:.3}'.format(np.mean(metrics[:,:,0]), np.mean(metrics[:,:,1]), np.mean(metrics[:,:,2])))

	metrics_to_csv(np.mean(metrics,axis=1), result_path+'sub_spec')
	return


def train_superposition(Niter = 500,epochs=5, batch_size=64,lr=1e-4, 
			crossValidation = False,addon="", compressedLayers = 'conv_spat', 
			result_path='', data_path = '',model_path = '',model_sup_path=''):


	randomizeSubjects = True
	numberOfSubjects = 9
	train_sets, valid_sets, test_sets = [], [], []
	subjects = list(range(1, numberOfSubjects+1))
	iterations = list(range(0, Niter))

	n_folds = 5 if crossValidation else 1
	test_mode = 'Xval' if crossValidation else 'test' 


	for fold in range(n_folds):
		# ---------------------- load model and create inital S -----------------------
		models = [None] * numberOfSubjects
		contextVectors = [None]*numberOfSubjects
		x=0
		
		for subject in subjects:
			s = str(subject)
			#load model
			filename= 'model_subject{:}{:}{:}.pt'.format(test_mode,subject,fold)
			
			pretrainedModel =th.load(model_path+filename)
			models[x] = pretrainedModel
			#load all sets
			train_set, valid_set, test_set =  loadData(data_path, subject)
			train_sets.append(train_set)
			valid_sets.append(valid_set)
			test_sets.append(test_set)
			#get weights from trained model
			weights = getWeights (pretrainedModel, compressedLayers)
			#create contextVector
			contextVector = getContextVector(weights.size, subject)
			contextVectors[x] = contextVector
			#create circular matrix
			C = circulant(contextVector)
			#pdb.set_trace()
			# convolve context vector with weights
			convolutions = np.matmul(C, weights)
			#superimpose weights
			if(subjects[0]==subject): S = np.array(convolutions).view()
			else: S += convolutions

			x+=1

		with open(result_path+'testAccuracies'+addon+test_mode+str(fold)+'.csv', 'w') as csvfile:
							filewriter = csv.writer(csvfile, delimiter=',',
									quotechar='|', quoting=csv.QUOTE_MINIMAL)
							filewriter.writerow(['training', 'subject', 'train_misclass', 'valid_misclass', 'test_accuracy'])

		print("{:}: Fold {:} of {:}".format(test_mode,fold+1,n_folds))

		# ----------------------retrieve weights, retrain layer(s), plug back in----------------------

		for itr in iterations:
			if randomizeSubjects:
				np.random.seed()
				np.random.shuffle(subjects)

			acc_test = 0 

			for subject in subjects:
				s = str(subject)
				
				#load model, data sets and context vectors
				trainedModel = models[subject-1]
				train_set, valid_set, test_set = train_sets[subject-1], valid_sets[subject-1], test_sets[subject-1]
				contextVector = contextVectors[subject-1]

				# correlate weights
				C = circulant(contextVector)
				weights_retrieved = np.matmul(C.T,S)

				#set retrieved weights into model and save it
				trainedModel_retrievedWeights = setWeights (trainedModel, weights_retrieved, compressedLayers)

				# calculate testAccuracy
				if crossValidation:
					testAccBeforeRetraining = testOnSet(trainedModel_retrievedWeights, valid_set)
				else:
					testAccBeforeRetraining = testOnSet(trainedModel_retrievedWeights, test_set)
				
				acc_test += testAccBeforeRetraining

				#retrain on one subject
				retrainedModel = copy.deepcopy(retrainNetwork (copy.deepcopy(trainedModel_retrievedWeights), 
					epochs, train_set, valid_set, test_set, lr=lr, crossValidation = crossValidation,  batch_size=batch_size))
				
				#calculate delta
				newWeights = getWeights (retrainedModel, compressedLayers)
				delta = newWeights - weights_retrieved

				#convolve delta with context vector
				deltaConv = np.matmul(C, delta)

				#update S
				S += np.array(deltaConv)

				#save misclassifications
				trainAcc = retrainedModel.epochs_df.train_misclass.values
				validAcc = retrainedModel.epochs_df.valid_misclass.values

				#save network
				models[subject-1] = copy.deepcopy(retrainedModel)
				del retrainedModel

				#save data and model to files
				if (itr == Niter-1):
						fn = 'retrainedModel_subject'+s+'.pt'
						th.save(models[subject-1],model_sup_path+fn)

				with open(result_path+'misclassifications'+addon+'.csv',"a") as csvfile:
						filewriter = csv.writer(csvfile)
						filewriter.writerow([itr, 's'+s, trainAcc, validAcc, testAccBeforeRetraining])

			print('Itr: {:} \t Acc {:.3}'.format(itr,acc_test/numberOfSubjects))

def retrainNetwork (exp, retrainingEpochs, train_set, valid_set, test_set,  lr=1e-3, crossValidation=False, batch_size=60):
	cuda = True
	set_random_seeds(seed=20190706, cuda=cuda)
	if cuda: exp.model.cuda()
	log.info("Model: \n{:s}".format(str(exp.model)))
	optimizer = optim.Adam(exp.model.parameters())
	iterator = BalancedBatchSizeIterator(batch_size=batch_size)

	stop_criterion = MaxEpochs(retrainingEpochs)

	optimizer = optim.Adam(exp.model.parameters(), lr=lr)

	iterator = BalancedBatchSizeIterator(batch_size=batch_size)

	monitors = [LossMonitor(), MisclassMonitor(), RuntimeMonitor()]

	model_constraint = MaxNormDefaultConstraint()


	exp = Experiment(exp.model, train_set, valid_set, test_set, iterator=iterator,
					 loss_function=F.nll_loss, optimizer=optimizer,
					 model_constraint=model_constraint,
					 monitors=monitors,
					 stop_criterion=stop_criterion,
					 remember_best_column='valid_misclass',
					 cuda=cuda,
					 run_after_early_stop= False,
					 do_early_stop = False)

	if not crossValidation:
		exp.datasets['train'] = concatenate_sets([exp.datasets['train'],
											 exp.datasets['valid']])
	exp.run()

	return exp



def loadData(data_folder, subject_id,fold=0):
	text_trap = io.StringIO()
	sys.stdout = text_trap
	warnings.filterwarnings("ignore")

	low_cut_hz = 4 # 0 or 4
	ival = [-500, 4000]
	high_cut_hz = 38
	factor_new = 1e-3
	init_block_size = 1000
	valid_set_fraction = 0.2

	train_filename = 'A{:02d}T.gdf'.format(subject_id)
	test_filename = 'A{:02d}E.gdf'.format(subject_id)
	train_filepath = os.path.join(data_folder, train_filename)
	test_filepath = os.path.join(data_folder, test_filename)
	train_label_filepath = train_filepath.replace('.gdf', '.mat')

	test_label_filepath = test_filepath.replace('.gdf', '.mat')

	train_loader = BCICompetition4Set2A(
			train_filepath, labels_filename=train_label_filepath)
	test_loader = BCICompetition4Set2A(
			test_filepath, labels_filename=test_label_filepath)
   
	train_cnt = train_loader.load()
	test_cnt = test_loader.load()

	# Preprocessing

	train_cnt = train_cnt.drop_channels(['STI 014', 'EOG-left',
										 'EOG-central', 'EOG-right'])
	assert len(train_cnt.ch_names) == 22
	# lets convert to millvolt for numerical stability of next operations
	train_cnt = mne_apply(lambda a: a * 1e6, train_cnt)
	train_cnt = mne_apply(
		lambda a: bandpass_cnt(a, low_cut_hz, high_cut_hz, train_cnt.info['sfreq'],
							   filt_order=3,
							   axis=1), train_cnt)
	train_cnt = mne_apply(
		lambda a: exponential_running_standardize(a.T, factor_new=factor_new,
												  init_block_size=init_block_size,
												  eps=1e-4).T,
		train_cnt)

	test_cnt = test_cnt.drop_channels(['STI 014', 'EOG-left',
									   'EOG-central', 'EOG-right'])
	assert len(test_cnt.ch_names) == 22
	test_cnt = mne_apply(lambda a: a * 1e6, test_cnt)
	test_cnt = mne_apply(
		lambda a: bandpass_cnt(a, low_cut_hz, high_cut_hz, test_cnt.info['sfreq'],
							   filt_order=3,
							   axis=1), test_cnt)
	test_cnt = mne_apply(
		lambda a: exponential_running_standardize(a.T, factor_new=factor_new,
												  init_block_size=init_block_size,
												  eps=1e-4).T,
		test_cnt)

	marker_def = OrderedDict([('Left Hand', [1]), ('Right Hand', [2],),
							  ('Foot', [3]), ('Tongue', [4])])

	train_set = create_signal_target_from_raw_mne(train_cnt, marker_def, ival)
	test_set = create_signal_target_from_raw_mne(test_cnt, marker_def, ival)

	# split training set according to 5-fold validation. If no Xval, training and
	# validation will be merged later   
	train_set, valid_set = split_into_train_test(
			train_set, n_folds = 5, i_test_fold = fold,rng=None)

	# now restore stdout function
	sys.stdout = sys.__stdout__
	return train_set, valid_set, test_set


def getContextVector (length, subject):
		mean = 0
		variance = 1/(math.sqrt(length))
		seed = int(subject)*23072019
		th.manual_seed(seed)
		contextVector = th.normal(th.zeros(length,1)*mean, th.ones(length,1)*variance)
		return contextVector

def toVector(x):
	return np.reshape(x, (len(x)*len(x[0]),1))

def toMatrix(x, whichLayer):
	if whichLayer == 'classifier': return np.reshape(x, (4, 40, 69, 1))
	elif whichLayer == 'conv_spat': return np.reshape(x, (40,40,1,22))
	elif whichLayer == 'conv_time': return np.reshape(x, (40,1,25,1))

def getAccuracy (preds, true_labels):
		predictedLabels = []
		assert len(preds)==len(true_labels)
		for pred in preds: predictedLabels.append(np.argmax(pred))
		accuracy =1- np.count_nonzero(predictedLabels-true_labels)/len(true_labels)

		return accuracy

def testOnSet (trainedModel, sets):
	true_accuracy_test = 0
	accuracy_test= []
	trainedModel.model.cuda()

	for batch in trainedModel.iterator.get_batches(sets, shuffle=False):
		preds, loss = trainedModel.eval_on_batch(batch[0], batch[1])
		accuracy_test.append(getAccuracy (preds, batch[1]))

	true_accuracy_test= np.mean(accuracy_test)
	return  true_accuracy_test


def getWeights (model, compressedLayers):
	if compressedLayers == 'conv_class':
		weights_classifier = model.model.state_dict().__getitem__("conv_classifier.weight").cpu().numpy().view()
		weights = np.reshape(weights_classifier, (weights_classifier.size, 1))
	elif compressedLayers == 'conv_class_spat':
		weights1 = model.model.state_dict().__getitem__("conv_classifier.weight").cpu().numpy().view()
		weights2 = model.model.state_dict().__getitem__("conv_spat.weight").cpu().numpy().view()
		weights1 = np.reshape(weights1, (weights1.size,1))
		weights2 = np.reshape(weights2, (weights2.size,1))
		weights = np.concatenate([weights1, weights2])
	elif compressedLayers == 'conv_spat':
				weights_ = model.model.state_dict().__getitem__("conv_spat.weight").cpu().numpy().view()
				weights = np.reshape(weights_, (weights_.size, 1))
	return weights

def setWeights (model, weights_retrieved, compressedLayers):
		length_weights_classifier = 4*40*69

		if compressedLayers == 'conv_class':
			weights_retrieved_matrix = th.tensor(np.asarray(toMatrix(weights_retrieved, 'classifier')).view(), dtype=th.float)
			model.model.conv_classifier.weight = th.nn.Parameter(weights_retrieved_matrix)

		if compressedLayers == 'conv_class_spat':
			weights_classifier = weights_retrieved[:length_weights_classifier]
			weights_conv_spat = weights_retrieved[length_weights_classifier:]
			weights_classifier_matrix = th.tensor(np.asarray(toMatrix(weights_classifier, 'classifier')).view(), dtype=th.float)
			weights_conv_spat_matrix = th.tensor(np.asarray(toMatrix(weights_conv_spat, 'conv_spat')).view(), dtype=th.float)
			model.model.conv_classifier.weight = th.nn.Parameter(weights_classifier_matrix)
			model.model.conv_spat.weight = th.nn.Parameter(weights_conv_spat_matrix)
		if compressedLayers == 'conv_spat':
			weights_retrieved_matrix = th.tensor(np.asarray(toMatrix(weights_retrieved, 'conv_spat')).view(), dtype=th.float)
			model.model.conv_spat.weight = th.nn.Parameter(weights_retrieved_matrix)
		return model

def metrics_to_csv(metrics, filename):
	np.savetxt('{}.csv'.format(filename), metrics, delimiter=',')


