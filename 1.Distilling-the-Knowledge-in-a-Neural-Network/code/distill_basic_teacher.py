### Import required packages and limit GPU usage
import numpy as np
import math

import matplotlib.pyplot as plt

import pickle
import argparse
import time
import itertools
from copy import deepcopy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import networks
import utils


use_gpu = True    # set use_gpu to True if system has gpu
gpu_id = 1        # id of gpu to be used
cpu_device = torch.device('cpu')
# fast_device is where computation (training, inference) happens
fast_device = torch.device('cpu')
if use_gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'    # set visible devices depending on system configuration
    fast_device = torch.device('cuda:' + str(gpu_id))

def reproducibilitySeed():
    """
    Ensure reproducibility of results; Seeds to 42
    """
    torch_init_seed = 42
    torch.manual_seed(torch_init_seed)
    numpy_init_seed = 42
    np.random.seed(numpy_init_seed)
    if use_gpu:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

reproducibilitySeed()


### Load dataset

import torchvision
import torchvision.transforms as transforms

mnist_image_shape = (28, 28)
random_pad_size = 2
# Training images augmented by randomly shifting images by at max. 2 pixels in any of 4 directions
transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(mnist_image_shape, random_pad_size),
                    transforms.ToTensor(),
                    # transforms.Normalize((0.5, 0.5), (0.5, 0.5))
                    transforms.Normalize((0.5,), (0.5,))  # 标准化，只提供一个均值和一个标准差
                ]
            )

transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    # transforms.Normalize((0.5, 0.5), (0.5, 0.5))
                    transforms.Normalize((0.5,), (0.5,))  # 标准化，只提供一个均值和一个标准差
                ]
            )

train_val_dataset = torchvision.datasets.MNIST(root='./MNIST_dataset/', train=True, 
                                            download=True, transform=transform_train)

test_dataset = torchvision.datasets.MNIST(root='./MNIST_dataset/', train=False, 
                                            download=True, transform=transform_test)


num_train = int(1.0 * len(train_val_dataset) * 95 / 100)
num_val = len(train_val_dataset) - num_train
train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [num_train, num_val])


batch_size = 128
num_workers = 2
train_val_loader = torch.utils.data.DataLoader(train_val_dataset, batch_size=128, shuffle=True, num_workers=num_workers)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=num_workers)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=num_workers)


### Train teacher network

checkpoints_path = 'checkpoints_teacher/'
if not os.path.exists(checkpoints_path):
    os.mkdir(checkpoints_path)

num_epochs = 60
print_every = 100    # Interval size for which to print statistics of training


# Hyperparamters can be tuned by setting required range below
# learning_rates = list(np.logspace(-4, -2, 3))
learning_rates = [1e-2]
learning_rate_decays = [0.95]    # learning rate decays at every epoch
# weight_decays = [0.0] + list(np.logspace(-5, -1, 5))
weight_decays = [1e-5]           # regularization weight
momentums = [0.9]
# dropout_probabilities = [(0.2, 0.5), (0.0, 0.0)]
dropout_probabilities = [(0.0, 0.0)]
hparams_list = []
for hparam_tuple in itertools.product(dropout_probabilities, weight_decays, learning_rate_decays, 
                                        momentums, learning_rates):
    hparam = {}
    hparam['dropout_input'] = hparam_tuple[0][0]
    hparam['dropout_hidden'] = hparam_tuple[0][1]
    hparam['weight_decay'] = hparam_tuple[1]
    hparam['lr_decay'] = hparam_tuple[2]
    hparam['momentum'] = hparam_tuple[3]
    hparam['lr'] = hparam_tuple[4]
    hparams_list.append(hparam)

results = {}
for hparam in hparams_list:
    print('Training with hparams ' + utils.hparamToString(hparam))
    reproducibilitySeed()
    teacher_net = networks.TeacherNetwork()
    teacher_net = teacher_net.to(fast_device)
    hparam_tuple = utils.hparamDictToTuple(hparam)
    results[hparam_tuple] = utils.trainTeacherOnHparam(teacher_net, hparam, num_epochs, 
                                                        train_val_loader, None, 
                                                        print_every=print_every, 
                                                        fast_device=fast_device)
    save_path = checkpoints_path + utils.hparamToString(hparam) + '_final.tar'
    torch.save({'results' : results[hparam_tuple], 
                'model_state_dict' : teacher_net.state_dict(), 
                'epoch' : num_epochs}, save_path)

# Calculate test accuracy
_, test_accuracy = utils.getLossAccuracyOnDataset(teacher_net, test_loader, fast_device)
print('test accuracy: ', test_accuracy)