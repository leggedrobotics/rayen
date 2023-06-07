import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
import argparse
import itertools
import pandas as pd
from torch.utils.data import DataLoader
from early_stopping import EarlyStopping

from constraint_layer import ConstraintLayer
from cost_computer import CostComputer
from create_dataset import createProjectionDataset, getCorridorDatasetsAndConstraints
from examples_sets import getExample

import utils
import tqdm

import waitGPU

import random
import time

from torch.utils.tensorboard import SummaryWriter

import uuid

class SplittedDatasetAndGenerator():
	def __init__(self, dataset, percent_train, percent_val, batch_size):
		assert percent_train<=1
		assert percent_val<=1
		assert (percent_train+percent_val)<=1

		train_size = int(percent_train * len(dataset))
		val_size = int(percent_val * len(dataset))
		test_size = len(dataset) - train_size - val_size

		# train_size = 1
		# val_size = 1
		# test_size = 1

		#First option
		# self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

		#Second option (Matlab is already doing the randomness). Don't randomize here so that all the methods use the same datasets
		#See https://stackoverflow.com/a/70278974
		self.train_dataset = torch.utils.data.Subset(dataset, range(train_size))
		self.val_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))
		self.test_dataset = torch.utils.data.Subset(dataset, range(train_size + val_size, train_size + val_size + test_size))


		# assert len(self.train_dataset)>0
		# assert len(self.val_dataset)>0
		# assert len(self.test_dataset)>0

		utils.printInBoldRed(f"Elements [train, val, test]={[len(self.train_dataset), len(self.val_dataset), len(self.test_dataset)]}")

		self.batch_size=batch_size

		if(len(self.train_dataset)>0):
			self.train_generator = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
			utils.printInBoldRed(f"Train batches {len(self.train_generator)}")


		if(len(self.val_dataset)>0):
			self.val_generator = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
			utils.printInBoldRed(f"Val batches {len(self.val_generator)}")


		if(len(self.test_dataset)>0): #len(self.test_dataset)
			self.test_generator = DataLoader(self.test_dataset, batch_size=len(self.test_dataset), shuffle=False) #One batch only for testing [better inference time estimation]
			utils.printInBoldRed(f"Test batches {len(self.test_generator)}")



def onePassOverDataset(model, params, sdag, my_type, cs):

	cost_computer=CostComputer(cs)

	device = torch.device(params['device'])
	model = model.to(device)
	cost_computer = cost_computer.to(device)

	if(my_type=='train'):
		model.train()
		generator=sdag.train_generator
		enable_grad=True;
		optimizer = torch.optim.Adam(model.parameters(),lr=params['learning_rate'])

	elif(my_type=='val'):
		model.eval()
		generator=sdag.val_generator
		enable_grad=False;
	elif(my_type=='test'):
		model.eval()
		generator=sdag.test_generator
		enable_grad=False;
	else:
		raise NotImplementedError

	sum_all_losses=0.0
	sum_time_s=0.0

	if(my_type=='test'):
		sum_all_violations=0.0
		sum_all_losses_optimization=0.0
		sum_all_time_s_optimization=0.0
		sum_all_violations_optimization=0.0


	num_samples_dataset=len(generator.dataset);

	cuda_timer=utils.CudaTimer()

	with torch.set_grad_enabled(enable_grad):
		for x, y, Pobj, qobj, robj, opt_time_s, cost in generator: #For each of the batches	

			#----------------------
			x = x.to(device)
			y = y.to(device)
			
			cuda_timer.start()
			y_predicted = model(x)
			sum_time_s += cuda_timer.endAndGetTimeSeconds()
			loss=cost_computer.getSumLossAllSamples(params, y, y_predicted, Pobj, qobj, robj, isTesting=(my_type=='test'))
			# print(f"Loss={loss.item()}")
			sum_all_losses +=  loss.item();
			#----------------------

			if(my_type=='train'):
				num_samples_this_batch=x.shape[0];
				loss_per_sample_in_batch=loss/num_samples_this_batch;
				optimizer.zero_grad()
				loss_per_sample_in_batch.backward()
				optimizer.step()


			if(my_type=='test'):
				print("Computing violations...")
				sum_all_violations += np.sum(np.apply_along_axis(cs.getViolation,axis=1, arr=y_predicted.cpu().numpy())).item()
				print("Violations computed")

				###### compute the results from the optimization. TODO: Change to a different place?
				y_predicted=y
				loss_optimization=cost_computer.getSumLossAllSamples(params, y, y_predicted, Pobj, qobj, robj, isTesting=True)
				sum_all_losses_optimization += loss_optimization.item();
				# print(f"Loss Opt={loss_optimization.item()}")
				# print(f"Original Loss Opt={torch.sum(cost).item()}")
				assert abs(loss_optimization.item()-torch.sum(cost).item())<0.001

				print("Computing violations optimization...")
				sum_all_violations_optimization += np.sum(np.apply_along_axis(cs.getViolation,axis=1, arr=y_predicted.cpu().numpy())).item()
				print("Violations computed")
				sum_all_time_s_optimization += torch.sum(opt_time_s).item()
				#########################################################


	#############################

	metrics={};
	metrics['loss']=sum_all_losses/num_samples_dataset

	if(my_type=='test'):
		metrics['violation']=                    sum_all_violations/num_samples_dataset
		metrics['time_s']=                       sum_time_s/num_samples_dataset
		metrics["optimization_loss"]=            sum_all_losses_optimization/num_samples_dataset
		metrics["optimization_violation"]=       sum_all_violations_optimization/num_samples_dataset
		metrics["optimization_time_s"]=    sum_all_time_s_optimization/num_samples_dataset

	#############################

	return metrics



def train_model(model, params, sdag, tensorboard_writer, cs):
	model = model.to(torch.device(params['device']))
	optimizer = torch.optim.Adam(model.parameters(),lr=params['learning_rate'])

	metrics_all_epochs = {'train_loss': [], 'val_loss': []}
	
	my_early_stopping = EarlyStopping(patience=1e100, verbose=False)

	#See https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

	# with tqdm.trange(params['num_epochs'], ncols=120) as pbar:
		# for epoch in pbar:
	for epoch in range(params['num_epochs']):

		# pbar.set_description(f"Epoch {epoch}")

		metrics_training_this_epoch=onePassOverDataset(model, params, sdag, 'train', cs)
		metrics_validation_this_epoch=onePassOverDataset(model, params, sdag, 'val', cs)
		my_early_stopping(metrics_validation_this_epoch['loss'], model)

		metrics_all_epochs['train_loss'].append(metrics_training_this_epoch['loss']) 
		metrics_all_epochs['val_loss'].append(metrics_validation_this_epoch['loss']) 
		
		if epoch % params['verbosity'] == 0:
			print(f"[{params['method']}, w={params['weight_soft_cost']}] {epoch}: train: {metrics_all_epochs['train_loss'][-1]}, val: {metrics_all_epochs['val_loss'][-1]}")

		#This creates two separate plots
		# tensorboard_writer.add_scalar("Loss/train", metrics_all_epochs['train_loss'][-1], epoch)
		# tensorboard_writer.add_scalar("Loss/val", metrics_all_epochs['val_loss'][-1], epoch)

		#This createst one plot
		tensorboard_writer.add_scalars('loss', {'train':metrics_all_epochs['train_loss'][-1], 'val':metrics_all_epochs['val_loss'][-1]}, epoch)

		# pbar.set_postfix(loss=metrics_all_epochs['train_loss'][-1], val=metrics_all_epochs['val_loss'][-1])

		if my_early_stopping.early_stop:
			print("Early stopping")
			#Delete the last elements, see https://stackoverflow.com/a/15715924
			del metrics_all_epochs['train_loss'][-my_early_stopping.patience:]
			del metrics_all_epochs['val_loss'][-my_early_stopping.patience:]
			break

	my_early_stopping.load_best_model(model)
	tensorboard_writer.flush()

	return metrics_all_epochs


def main(params):


	################# To launch tensorboard directly
	# import os
	# import subprocess
	# folder="runs"
	# os.system("pkill -f tensorboard")
	# os.system("rm -rf "+folder)
	# proc1 = subprocess.Popen(["tensorboard","--logdir",folder,"--bind_all"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
	# proc2 = subprocess.Popen(["google-chrome","http://localhost:6006/"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
	############################################

	torch.set_default_dtype(torch.float64) #This is very important. If you use float32, you may end up with a "large" negative discriminant (like -0.000259) in solveSecondOrderEq 
										   #This is also related to the fact that the P matrices need to be PSD matrices (while numerically this is sometimes difficult to achieve)

	tensorboard_writer = SummaryWriter()

	## PROJECTION EXAMPLES
	# cs=getExample(4)
	# my_dataset=createProjectionDataset(200, cs, 4.0);
	# my_dataset_out_dist=createProjectionDataset(200, cs, 7.0);

	## CORRIDOR EXAMPLES
	my_dataset, my_dataset_out_dist, cs=getCorridorDatasetsAndConstraints(params['dimension_dataset'])

	############### THIS AVOIDS CREATING the dataset all the time
	# import pickle

	# try:
	# 	my_dataset = utils.loadpickle("my_dataset.pkl",)
	# 	my_dataset_out_dist = utils.loadpickle("my_dataset_out_dist.pkl",)
	# 	cs = utils.loadpickle("cs.pkl",)
	# except:
	# 	my_dataset, my_dataset_out_dist, cs=getCorridorDatasetsAndConstraints(params['dimension_dataset'])
	# 	utils.savepickle(my_dataset, "my_dataset.pkl")
	# 	utils.savepickle(my_dataset_out_dist, "my_dataset_out_dist.pkl")
	# 	utils.savepickle(cs, "cs.pkl")

	###################

	sdag=SplittedDatasetAndGenerator(my_dataset, percent_train=0.5045, percent_val=0.2, batch_size=params['batch_size'])
	sdag_out_dist=SplittedDatasetAndGenerator(my_dataset_out_dist, percent_train=0.0, percent_val=0.0, batch_size=params['batch_size'])

	if(params['method']=='DC3'):
		args_DC3={}
		args_DC3['lr'] = params['DC3_lr']
		args_DC3['eps_converge'] = params['DC3_eps_converge']
		args_DC3['momentum'] = params['DC3_momentum']
		args_DC3['max_steps_training'] = params['DC3_max_steps_training']
		args_DC3['max_steps_testing'] = params['DC3_max_steps_testing']
	else:
		args_DC3 = None

	######################### TRAINING
	folder="./scripts/results/"
	name_file='dataset'+str(params['dimension_dataset'])+'d_'+params['method']+"_weight_soft_cost_"+str(params["weight_soft_cost"])    #+uuid.uuid4().hex #https://stackoverflow.com/a/62277811
	path_policy = folder + name_file +".pt"
	path_training_metrics = folder + "metrics_train_" + name_file +".pkl"
	path_testing_in_dist_metrics = folder + "metrics_test_in_dist_" + name_file +".pkl"
	path_testing_out_dist_metrics = folder + "metrics_test_out_dist_" + name_file +".pkl"

	if(params['train']==True):

		#Slide 4 of https://fleuret.org/dlc/materials/dlc-handout-4-6-writing-a-module.pdf
		model = nn.Sequential(nn.Flatten(),
							  nn.Linear(my_dataset.getNumelX(), 64), 
							  # nn.BatchNorm1d(64),
							  nn.ReLU(),
							  nn.Linear(64, 64),
							  nn.ReLU(),
							  # nn.Dropout(p=0.2), 
							  # nn.Dropout(p=0.2),
							  nn.Linear(64, 64),
							  ConstraintLayer(cs, input_dim=64, method=params['method'], create_map=True, args_DC3=args_DC3)) 

		for i in range(torch.cuda.device_count()):
		   print(torch.cuda.get_device_properties(i).name)

		sleep_time=random.randint(0,20)
		print(f"Sleeping for {sleep_time} s")
		time.sleep(sleep_time) #Without this, you get the errors CUBLAS_STATUS_NOT_INITIALIZED or CUDA out of memory when running several training processes in parallel
		waitGPU.wait(utilization=50, memory_ratio=0.5, interval=random.randint(1, 20), available_memory=2000) #This is to avoid erros like "CUDA error: CUBLAS_STATUS_NOT_INITIALIZED" when launching many trainings in parallel
		print("Done waiting for GPU")

		training_metrics = train_model(model, params, sdag, tensorboard_writer, cs)

		utils.savepickle(training_metrics, path_training_metrics)

		#torch.save(model.state_dict(), path_policy)  #We don't use this one because DC3 has some randomness in the selection of partial_vars, and they may be different if this file is called twice (first with train=True/test=False and then with train=False/test=True)
		torch.save(model, path_policy)                #Save entire model, see https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-entire-model

		training_summary=f"k = {cs.k}, n = {cs.n}\n"\
						 f"Training: \n"\
						 f"  [{params['method']}] loss: {training_metrics['train_loss'][-1]:.6} \n"\
						 f"  [{params['method']}] val loss: {training_metrics['val_loss'][-1]:.6}"


		utils.printInBoldBlue(training_summary)


	if(params['test']==True):
		# model.load_state_dict(torch.load(path_policy)) #We don't use this one because of the reason above
		model = torch.load(path_policy) #See # https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-entire-model
		
		utils.printInBoldGreen("Warming up GPU for a better estimate of the computation time...")
		x_dummy=torch.Tensor(500, my_dataset.getNumelX(), 1).uniform_(-5.0, 5.0) #Just run some dummy operations on the GPU to warm it up
		x_dummy=x_dummy.to(torch.device(params['device']))
		_ = model(x_dummy)

		utils.printInBoldGreen("Testing model inside dist...")
		testing_metrics_in_dist = onePassOverDataset(model, params, sdag, 'test', cs)
		utils.printInBoldGreen("Testing model outside dist...")
		testing_metrics_out_dist = onePassOverDataset(model, params, sdag_out_dist, 'test', cs)

		utils.savepickle(testing_metrics_in_dist, path_testing_in_dist_metrics)
		utils.savepickle(testing_metrics_out_dist, path_testing_out_dist_metrics)

		num_trainable_params=sum(	p.numel() for p in model.parameters() if p.requires_grad)

		d = {'method': 				 [name_file,                         'dataset'+str(params['dimension_dataset'])+'d_'+'Optimization'],
			 'num_trainable_params': [num_trainable_params,         	  0], 
			'[In dist] loss':        [testing_metrics_in_dist['loss'],      	  testing_metrics_in_dist['optimization_loss']      ], 
			'[In dist] violation':   [testing_metrics_in_dist['violation'], 	  testing_metrics_in_dist['optimization_violation'] ],
			'[In dist] time_us':     [1e6*testing_metrics_in_dist['time_s'],     1e6*testing_metrics_in_dist['optimization_time_s'] ],
			#
			'[Out dist] loss':        [testing_metrics_out_dist['loss'],      	  testing_metrics_out_dist['optimization_loss']      ], 
			'[Out dist] violation':   [testing_metrics_out_dist['violation'], 	  testing_metrics_out_dist['optimization_violation'] ],
			'[Out dist] time_us':     [1e6*testing_metrics_out_dist['time_s'],     1e6*testing_metrics_out_dist['optimization_time_s'] ]}


		df = pd.DataFrame(data=d)

		pd.set_option('display.max_columns', None)
		print(df)

		path_pkl = folder + name_file +".pkl"

		df.to_pickle(path_pkl) 	

	tensorboard_writer.close()

#See https://stackoverflow.com/a/43357954/6057617
def str2bool(v):
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':

	utils.printInBoldGreen("\n\n\n==========================================")

	parser = argparse.ArgumentParser()
	parser.add_argument('--method', type=str, default='Bar') #RAYEN_old, RAYEN, Bar, UU, PP, UP, DC3
	parser.add_argument('--dimension_dataset', type=int, default=2)
	parser.add_argument('--use_supervised', type=str2bool, default=False)
	parser.add_argument('--weight_soft_cost', type=float, default=0.0)
	parser.add_argument('--device', type=str, default='cuda:0')
	parser.add_argument('--num_epochs', type=int, default=2000)
	parser.add_argument('--batch_size', type=int, default=256)
	parser.add_argument('--verbosity', type=int, default=1)
	parser.add_argument('--learning_rate', type=float, default=1e-4)
	parser.add_argument('--train', type=str2bool, default=True)
	parser.add_argument('--test', type=str2bool, default=True)
	#Parameters specific to DC3
	parser.add_argument('--DC3_lr', type=float, default=1e-5)            #Sometimes the DC3 inner gradient correction does not converge if this lr is high 
	parser.add_argument('--DC3_eps_converge', type=float, default=4e-7)
	parser.add_argument('--DC3_momentum', type=float, default=0.5)
	parser.add_argument('--DC3_max_steps_training', type=int, default=10)
	parser.add_argument('--DC3_max_steps_testing', type=int, default=500) #float("inf")


	args = parser.parse_args()
	params = vars(args)

	shouldnt_have_soft_cost=(params['method']=='RAYEN_old' or params['method']=='RAYEN' or params['method']=='Bar' or params['method']=='PP')
	

	# should_have_soft_cost=(
	# 						#Note that DC3 should have soft cost when training, see third paragraph of Section 3.2 of the DC3 paper
	# 						(params['method']=='DC3') or
	# 						(params['method']=='UP' and params['use_supervised']==False) or
	# 						(params['method']=='UU' and params['use_supervised']==False)
	# 						)

	# if(should_have_soft_cost):
	# 	assert params['weight_soft_cost']>0

	if(shouldnt_have_soft_cost):
		assert params['weight_soft_cost']==0

	print('Parameters:\n', params)


	main(params)