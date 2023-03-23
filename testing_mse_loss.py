import torch
import torch.nn as nn


input = torch.Tensor([  [[0],[0]],
						 [[0],[0]],
						 [[0],[0]],
	                  ])
target = torch.Tensor([  [[1],[2]],
						 [[3],[4]],
						 [[5],[6]],
	                  ])

n_batches=target.shape[0]

print("----------------------")

sum_reduction = nn.MSELoss(reduction='sum')(input, target)
print(f"sum_reduction={sum_reduction}")
print("----------------------")

mean_reduction = nn.MSELoss(reduction='mean')(input, target)
print(f"mean_reduction={mean_reduction}")
print("----------------------")

none_reduction = nn.MSELoss(reduction='none')(input, target)
print(f"none_reduction={none_reduction}")
print("----------------------")

tmp=torch.sum(none_reduction, dim=1);
print(f"loss per sample ={tmp}")
print("----------------------")

print(f"loss per batch ={torch.sum(tmp)/n_batches}")

print("----------------------")

print(f"loss per batch ={sum_reduction/n_batches}")
