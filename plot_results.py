import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset_util import MNIST

result_dir = 'results'

#Load raw data
img_transform = transforms.Compose([lambda x : 2 * x.float()/255 - 1])
test_data = MNIST(
    root='data',
    partition='test',
    transform=img_transform,
    download=False)
x_test = next(iter(DataLoader(test_data, batch_size=len(test_data), shuffle=False)))[0].numpy()

#V-parameterization data
data = pickle.load(open(os.path.join(result_dir, 'v_parameterization', 'v_parameterization_data.p'), 'rb'))
v_param_metrics = data['training_metrics']
v_param_argmins = data['argmins']
v_param_metrics['avg_cum_time'] = v_param_metrics.groupby(['epoch']).transform(np.mean)['cumulative_epoch_time']

#Plot some samples
num_samples = len(v_param_argmins)
samples = []
for n in range(10):
    samples.append({'sample': x_test[n], 'method': 'original_'+str(n)})
    samples.append({'sample': v_param_argmins[n], 'method': 'v_parameterization_'+str(n)})

dirname = result_dir
if not os.path.exists(dirname):
    os.makedirs(dirname)
for s in samples:
    s_plt = s['sample']
    if len(s_plt.shape) == 3:
        s_plt = s_plt[0]
    s_plt = s_plt.squeeze()
    plt.axis('off')
    filename = os.path.join(dirname, s['method'])
    plt.imsave(filename + '.png', s_plt, cmap='gray', vmin=-1, vmax=1, format='png', dpi=300)
