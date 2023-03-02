import sys
import os
import json
import time
import pickle
import pathlib
import numpy as np
import torch
import torch.nn as nn
import argparse
import itertools
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader

from constrained_nets import H_to_V
from models import ParameterizationModel, ProjectionModel
from dataset_util import MNIST
from osqp_projection import BatchProjector


def append_filename(params, filename):
    dir_path = os.path.join(params['result_dir'], params['method'])
    pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
    return os.path.join(dir_path, filename)

def train_model(model, loss_fn, params, data):
    device_id = params['device']
    device = torch.device('cuda:{}'.format(device_id) if device_id >= 0 else 'cpu')
    print('Using device:\n', device)
    model = model.to(device)
    print(model)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params['learning_rate']
    )

    train_data, val_data, test_data = data
    print('train size:', len(train_data))
    print('val size:', len(val_data))
    print('test size:', len(test_data))

    training_generator = DataLoader(
        train_data, batch_size=params['batch_size'], shuffle=True)
    validation_generator = DataLoader(
        val_data, batch_size=params['batch_size'], shuffle=False)
    test_generator = DataLoader(test_data, batch_size=params['batch_size'], shuffle=False)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, cooldown=5)

    training_metrics = {'train_loss': [], 'val_loss': [], 'cumulative_epoch_time': []}
    cumulative_epoch_time = 0
    for epoch in range(params['num_epochs']):
        model.train()
        train_loss = 0
        epoch_start = time.time()
        for x, _ in training_generator:
            x = x.to(device)
            out = model(x, epoch)
            loss = loss_fn(out, x)
            train_loss += loss.item()
            if epoch > 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        cumulative_epoch_time += epoch_time
        training_metrics['train_loss'].append(train_loss / len(training_generator))
        training_metrics['cumulative_epoch_time'].append(cumulative_epoch_time)

        with torch.set_grad_enabled(False):
            model.eval()
            val_loss = 0
            for x, _ in validation_generator:
                x = x.to(device)
                out = model(x)
                loss = loss_fn(out, x)
                val_loss += loss.item()
            training_metrics['val_loss'].append(val_loss / len(validation_generator))
            scheduler.step(val_loss / len(validation_generator))

        if epoch % params['verbosity'] == 0:
            print('{}: train loss: {}, validation loss: {}, lr: {:.2E}, epoch time: {}'.format(
                epoch,
                training_metrics['train_loss'][-1],
                training_metrics['val_loss'][-1],
                optimizer.param_groups[0]['lr'],
                epoch_time))

    with torch.set_grad_enabled(False):
        model.eval()
        for x, _ in test_generator:
            x = x.to(device)
            test_out = model(x)
            break

    return test_out.cpu().numpy(), training_metrics

def eval_model(model, params, data):
    device_id = params['device']
    device = torch.device('cuda:{}'.format(device_id) if device_id >= 0 else 'cpu')
    print('Using device:\n', device)
    model = model.to(device)
    print(model)
    
    batch_size = 256

    train_data, val_data, test_data = data

    validation_generator = DataLoader(train_data, batch_size=batch_size, shuffle=False)

    model.eval()
    val_loss = 0
    eval_times = np.zeros(params['n_evals'])
    with torch.set_grad_enabled(False):
        for n in range(params['n_evals']):
            start_eval_time = time.time()
            for x, _ in validation_generator:
                x = x.to(device)
                out = model(x)
            end_eval_time = time.time()
            eval_time = end_eval_time - start_eval_time
            eval_times[n] = eval_time
    print('Evaluated {} runs of {} samples with batch size {}.'.format(params['n_evals'], len(train_data), batch_size))
    print('mean eval time: {}, standard deviation: {}'.format(np.mean(eval_times), np.std(eval_times)))

def main(params):
    img_transform = lambda x : 2 * x.float()/255 - 1
    train_data = MNIST(
        root='data',
        partition='train',
        transform=img_transform,
        download=True)
    val_data = MNIST(
        root='data',
        partition='val',
        transform=img_transform,
        download=False)
    test_data = MNIST(
        root='data',
        partition='test',
        transform=img_transform,
        download=False)
    data = (train_data, val_data, test_data)


    # construct checkerboard constraint
    A = np.zeros((16, 28, 28)).astype(np.float32)
    b = np.zeros(16).astype(np.float32)
    for k in range(16):
        i = 7 * k % 28
        j = 7 * (k // 4)
        if k % 2 == 0:
            A_val = (-1)**j
        else:
            A_val = (-1)**(j + 1)
        A[k, i:i + 7, j:j + 7] = A_val
    A = A.reshape(16, -1)

    device_id = params['device']
    device = torch.device('cuda:{}'.format(device_id) if device_id >= 0 else 'cpu')


    if params['box_constraints']:
        box_constraints = (-1, 1)
    else:
        box_constraints = None

    # solve problem using specified method
    results = []  # a list of dicts
    for trial in range(params['n_trials']):
        if params['method'] == 'opt_projection':
            x_train = next(iter(DataLoader(train_data, batch_size=len(train_data), shuffle=False)))[0].numpy()
            x_val = next(iter(DataLoader(val_data, batch_size=len(val_data), shuffle=False)))[0].numpy()
            x_test = next(iter(DataLoader(test_data, batch_size=len(test_data), shuffle=False)))[0].numpy()
            print(np.min(x_train), np.max(x_train))
            num_pixels = x_train[0].flatten().shape[0]
            projector = BatchProjector(A, b, box_constraints)
            avg_min_train, argmins_train = projector.project(x_train)
            avg_min_val, argmins_val = projector.project(x_val)
            avg_min_test, argmins_test = projector.project(x_test)

            training_metrics = {}
            print('Batch averaged projection min train loss:',avg_min_train / num_pixels)
            print('Batch averaged projection min val loss:', avg_min_val / num_pixels)
            print('Batch averaged projection min test loss:', avg_min_test / num_pixels)
            training_metrics['val_loss'] = [avg_min_val / num_pixels]
            training_metrics['train_loss'] = [avg_min_train / num_pixels]
            training_metrics['cumulative_epoch_time'] = [None]

            argmins = argmins_test
        else:
            if params['method'] == 'test_time_projection':
                d = A.shape[1]
                mapping = nn.Sequential(nn.Linear(d,d))
                model = ProjectionModel(A, b, mapping, box_constraints)
                loss_fn = nn.MSELoss(size_average=True)
            elif params['method'] == 'v_parameterization':
                model = ParameterizationModel(A, b, mapping=None, box_constraints=box_constraints)
                loss_fn = nn.MSELoss(size_average=True)
            else:
                raise ValueError('Method "{}" not known.'.format(params['method']))

            argmins, training_metrics = train_model(model, loss_fn, params, data)
            argmins = argmins.reshape((-1, 28, 28))

        for it, t in enumerate(zip(training_metrics['train_loss'],
                                   training_metrics['val_loss'], training_metrics['cumulative_epoch_time'])):
            single_result = {}
            train_loss, val_loss, cumulative_epoch_time = t
            single_result['method'] = params['method']
            single_result['epoch'] = it
            single_result['val_loss'] = val_loss
            single_result['train_loss'] = train_loss
            single_result['cumulative_epoch_time'] = cumulative_epoch_time
            single_result['trial'] = trial
            results.append(single_result)

    # evaluate model
    if params['method'] != 'opt_projection':
        print('Entering evaluation')
        eval_model(model, params, data)
    # convert to pandas dataframe
    df = pd.DataFrame(results)

    # dump results to pickle
    if params['result_dir'] is not None:
        filename = params['method'] + '_data.p'
        to_dump = {'training_metrics': df, 'argmins': argmins}
        pickle.dump(to_dump, open(append_filename(params, filename), 'wb'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default=None)
    parser.add_argument('--result_dir', type=str, default=None)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--n_trials', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--n_evals', type=int, default=1)
    parser.add_argument('--verbosity', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--log_to_file', dest='log_to_file', action='store_true', default=False)
    parser.add_argument('--box_constraints', dest='box_constraints', action='store_true', default=False)
    args = parser.parse_args()
    params = vars(args)

    print('Parameters:\n', params)

    main(params)
