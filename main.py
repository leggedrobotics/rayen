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

# import random

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

	device_id = params['device']
	device = torch.device('cuda:{}'.format(device_id) if device_id >= 0 else 'cpu')
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


	with torch.set_grad_enabled(enable_grad):
		for x, y, Pobj, qobj, robj, opt_time_s, cost in generator: #For each of the batches	

			#----------------------
			x = x.to(device)
			y = y.to(device)
			time_start=time.time()
			y_predicted = model(x)
			sum_time_s += (time.time()-time_start)
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
				sum_all_violations += np.sum(np.apply_along_axis(cs.getViolation,axis=1, arr=y_predicted.cpu().numpy())).item()

				###### compute the results from the optimization. TODO: Change to a different place?
				y_predicted=y
				loss_optimization=cost_computer.getSumLossAllSamples(params, y, y_predicted, Pobj, qobj, robj, isTesting=True)
				sum_all_losses_optimization += loss_optimization.item();
				# print(f"Loss Opt={loss_optimization.item()}")
				# print(f"Original Loss Opt={torch.sum(cost).item()}")
				assert abs(loss_optimization.item()-torch.sum(cost).item())<0.001

				sum_all_violations_optimization += np.sum(np.apply_along_axis(cs.getViolation,axis=1, arr=y_predicted.cpu().numpy())).item()
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
	model = model.to(torch.device('cuda:{}'.format(params['device']) if params['device'] >= 0 else 'cpu'))
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
			print('{}: train: {}, val: {}, lr: {:.2E}'.format(
				epoch,
				metrics_all_epochs['train_loss'][-1],
				metrics_all_epochs['val_loss'][-1],
				optimizer.param_groups[0]['lr']))

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

	tensorboard_writer = SummaryWriter()

	torch.set_default_dtype(torch.float64) ##Use float32 here??

	## PROJECTION EXAMPLES
	cs=getExample(4)
	my_dataset=createProjectionDataset(200, cs, 4.0);
	my_dataset_out_dist=createProjectionDataset(200, cs, 7.0);

	## CORRIDOR EXAMPLES
	# my_dataset, my_dataset_out_dist, cs=getCorridorDatasetsAndConstraints()

	############### THIS AVOIDS CREATING the dataset all the time
	# import pickle

	# try:
	# 	my_dataset = utils.loadpickle("my_dataset.pkl",)
	# 	my_dataset_out_dist = utils.loadpickle("my_dataset_out_dist.pkl",)
	# 	cs = utils.loadpickle("cs.pkl",)
	# except:
	# 	my_dataset, my_dataset_out_dist, cs=getCorridorDatasetsAndConstraints()
	# 	utils.savepickle(my_dataset, "my_dataset.pkl")
	# 	utils.savepickle(my_dataset_out_dist, "my_dataset_out_dist.pkl")
	# 	utils.savepickle(cs, "cs.pkl")

	###################

	sdag=SplittedDatasetAndGenerator(my_dataset, percent_train=0.8, percent_val=0.1, batch_size=params['batch_size'])
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
	#Slide 4 of https://fleuret.org/dlc/materials/dlc-handout-4-6-writing-a-module.pdf
	constraint_layer=ConstraintLayer(cs, input_dim=64, method=params['method'], create_map=True, args_DC3=args_DC3) 
	model = nn.Sequential(nn.Flatten(),
						  nn.Linear(my_dataset.getNumelX(), 64), 
						  # nn.BatchNorm1d(64),
						  nn.ReLU(),
						  nn.Linear(64, 64),
						  nn.ReLU(),
						  # nn.Dropout(p=0.2), 
						  # nn.Dropout(p=0.2),
						  nn.Linear(64, 64),
						  constraint_layer
									 ) 
	### TRAIN THE MODEL 
	training_metrics = train_model(model, params, sdag, tensorboard_writer, cs)

	#Save the best model found
	folder="./scripts/results/"
	name_file=params['method']+"_weight_soft_cost_"+str(params["weight_soft_cost"])    #+uuid.uuid4().hex #https://stackoverflow.com/a/62277811

	torch.save(model.state_dict(), folder+name_file+".pt")
	# model.load_state_dict(torch.load('checkpoint.pt'))

	# model.load_state_dict(torch.load('./results/UU.pt'))

	print("Testing model...")
	testing_metrics = onePassOverDataset(model, params, sdag, 'test', cs)
	testing_metrics_out_dist = onePassOverDataset(model, params, sdag_out_dist, 'test', cs)
	print("Model tested...")

	#####PRINT STUFF
	print("\n\n-----------------------------------------------------")
	method=params['method']

	num_trainable_params=sum(	p.numel() for p in model.parameters() if p.requires_grad)

	training_summary=f"k = {cs.k}, n = {cs.n}, dim_after_map={constraint_layer.dim_after_map}\n"\
					 f"Num of trainable params = {num_trainable_params}\n\n"\
						f"Training: \n"\
						  f"  [{method}] loss: {training_metrics['train_loss'][-1]:.6} \n"\
						  f"  [{method}] val loss: {training_metrics['val_loss'][-1]:.6}"

	# testing_summary=f"Testing: \n"\
	# 					 f"  [{method}] loss: {testing_metrics['loss']:.6} \n"\
	# 					 f"  [{method}] violation: {testing_metrics['violation']:.6} \n"\
	# 					 f"  [{method}] time_ms: {1000.0*testing_metrics['time_s']:.6} \n"\
	# 					 f"  [Opt] loss: {testing_metrics['optimization_loss']:.6} \n"\
	# 					 f"  [Opt] violation: {testing_metrics['optimization_violation']:.6} \n"\
	# 					 f"  [Opt] time_us: {1e6*testing_metrics['optimization_time_s']:.6} \n";


	# testing_out_dist_summary=f"Testing outside distrib: \n"\
	# 					 f"  [{method}] loss: {testing_metrics_out_dist['loss']:.6} \n"\
	# 					 f"  [{method}] violation: {testing_metrics_out_dist['violation']:.6} \n"\
	# 					 f"  [{method}] time_ms: {1000.0*testing_metrics_out_dist['time_s']:.6} \n"\
	# 					 f"  [Opt] loss: {testing_metrics_out_dist['optimization_loss']:.6} \n"\
	# 					 f"  [Opt] violation: {testing_metrics_out_dist['optimization_violation']:.6} \n"\
	# 					 f"  [Opt] time_us: {1e6*testing_metrics_out_dist['optimization_time_s']:.6} \n";


	utils.printInBoldBlue(training_summary)
	# utils.printInBoldRed(testing_summary)
	# utils.printInBoldGreen(testing_out_dist_summary)

	# index=                       [method+" In dist", method+"Out of dist", "Opt In dist", "Opt Out dist"]
	# d = {'num_trainable_params': [num_trainable_params,         	 num_trainable_params,                     0, 												 0], 
	#     'loss':                  [testing_metrics['loss'],      	 testing_metrics_out_dist['loss'],         testing_metrics['optimization_loss'],            testing_metrics_out_dist['optimization_loss']     ], 
	#     'violation':             [testing_metrics['violation'], 	 testing_metrics_out_dist['violation'],    testing_metrics['optimization_violation'],       testing_metrics_out_dist['optimization_violation'] ],
	#     'time_us':               [1e6*testing_metrics['time_s'],     1e6*testing_metrics_out_dist['time_s'],   1e6*testing_metrics['optimization_time_s'],      1e6*testing_metrics_out_dist['optimization_time_s']             ]   }


	index=                       [name_file, "Optimization"]
	d = {'num_trainable_params': [num_trainable_params,         	  0], 
	    '[In dist] loss':        [testing_metrics['loss'],      	  testing_metrics['optimization_loss']      ], 
	    '[In dist] violation':   [testing_metrics['violation'], 	  testing_metrics['optimization_violation'] ],
	    '[In dist] time_us':     [1e6*testing_metrics['time_s'],     1e6*testing_metrics['optimization_time_s'] ],
	    #
	    '[Out dist] loss':        [testing_metrics_out_dist['loss'],      	  testing_metrics_out_dist['optimization_loss']      ], 
	    '[Out dist] violation':   [testing_metrics_out_dist['violation'], 	  testing_metrics_out_dist['optimization_violation'] ],
	    '[Out dist] time_us':     [1e6*testing_metrics_out_dist['time_s'],     1e6*testing_metrics_out_dist['optimization_time_s'] ]}


	df = pd.DataFrame(data=d, index=index)
	print(df)



	# f = open(name+".txt", "w")
	# f.write(df.to_string())
	# f.write("\n\n\n"+training_summary)
	# f.close()

	# df.to_csv(name+".csv")

	df.to_pickle(folder+name_file+".pkl")  


	tensorboard_writer.close()


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--method', type=str, default='walker_1') #walker_2, walker_1, Bar, UU, PP, UP, DC3
	parser.add_argument('--use_supervised', type=bool, default=False)
	parser.add_argument('--weight_soft_cost', type=float, default=0.0)
	parser.add_argument('--device', type=int, default=0)
	parser.add_argument('--num_epochs', type=int, default=100)
	parser.add_argument('--batch_size', type=int, default=400)
	parser.add_argument('--verbosity', type=int, default=1)
	parser.add_argument('--learning_rate', type=float, default=1e-4)
	#Parameters specific to DC3
	parser.add_argument('--DC3_lr', type=float, default=3e-4)
	parser.add_argument('--DC3_eps_converge', type=float, default=1e-5)
	parser.add_argument('--DC3_momentum', type=float, default=0.5)
	parser.add_argument('--DC3_max_steps_training', type=int, default=10)
	parser.add_argument('--DC3_max_steps_testing', type=int, default=10) #float("inf")


	args = parser.parse_args()
	params = vars(args)

	shouldnt_have_soft_cost=(params['method']=='walker_2' or params['method']=='walker_1' or params['method']=='Bar' or params['method']=='PP')
	

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