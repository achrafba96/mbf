import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
import scipy
import copy
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import time
import pickle

import os
import math
import psutil
import itertools
import datetime
import shutil

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from .utils_functions import *
import warnings
warnings.filterwarnings('error')
# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings("ignore", message="numpy.dtype size changed")


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def add_res_block(layers_, in_channels_1, out_channels_1, stride1, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias):
    layer_ = {}

    if if_conv_bias == False:
        assert if_downsample_only == True
        if shortcut_type == 'padding':
            if stride1 > 1:
                if if_BNNoAffine:
                    layer_['name'] = 'ResBlock-BNNoAffine-PaddingShortcut-NoBias'
                else:
                    layer_['name'] = 'ResBlock-BN-PaddingShortcut-NoBias'
            else:
                if if_BNNoAffine:
                    layer_['name'] = 'ResBlock-BNNoAffine-identityShortcut-NoBias'
                else:
                    layer_['name'] = 'ResBlock-BN-identityShortcut-NoBias'
                
        elif shortcut_type == 'conv':
            assert if_BNNoAffine == False
            assert if_BN_shortcut == True
            if stride1 > 1:
                layer_['name'] = 'ResBlock-BN-BNshortcut-NoBias'
            else:
                layer_['name'] = 'ResBlock-BN-identityShortcut-NoBias'
        else:
            print('shortcut_type')
            print(shortcut_type)
            sys.exit()
 
    else:
        if if_downsample_only:
            assert if_BNNoAffine == False
            assert if_BN_shortcut == True
            if stride1 > 1:
                layer_['name'] = 'ResBlock-BN-BNshortcut'
            else:
                layer_['name'] = 'ResBlock-BN-identityShortcut'

        else:
            if if_BNNoAffine:
                assert if_BN_shortcut == False
                layer_['name'] = 'ResBlock-BNNoAffine'
            else:
                if if_BN_shortcut:
                    layer_['name'] = 'ResBlock-BN-BNshortcut'
                else:
                    layer_['name'] = 'ResBlock-BN'

    layer_['conv1'] = {}
    
    layer_['conv1']['conv_in_channels'] = in_channels_1
    layer_['conv1']['conv_out_channels'] = out_channels_1
    layer_['conv1']['conv_kernel_size'] = 3
    layer_['conv1']['conv_stride'] = stride1
    layer_['conv1']['conv_padding'] = 1
    
    layer_['conv1']['conv_bias'] = if_conv_bias
    if if_BNNoAffine:
        layer_['BNNoAffine1'] = {}
        layer_['BNNoAffine1']['num_features'] = out_channels_1
    else:
        layer_['BN1'] = {}
        layer_['BN1']['num_features'] = out_channels_1

    layer_['conv2'] = {}
    layer_['conv2']['conv_in_channels'] = out_channels_1
    layer_['conv2']['conv_out_channels'] = out_channels_1
    layer_['conv2']['conv_kernel_size'] = 3
    layer_['conv2']['conv_stride'] = 1
    layer_['conv2']['conv_padding'] = 1
    layer_['conv2']['conv_bias'] = if_conv_bias

    if if_BNNoAffine:
        layer_['BNNoAffine2'] = {}
        layer_['BNNoAffine2']['num_features'] = out_channels_1
    else:
        layer_['BN2'] = {}
        layer_['BN2']['num_features'] = out_channels_1
    
    if layer_['name'] in ['ResBlock-BN-identityShortcut',
                          'ResBlock-BN-identityShortcut-NoBias',
                          'ResBlock-BN-PaddingShortcut-NoBias']:
        1
    else:
        layer_['conv3'] = {}
        layer_['conv3']['conv_in_channels'] = in_channels_1
        layer_['conv3']['conv_out_channels'] = out_channels_1
        layer_['conv3']['conv_kernel_size'] = 1
        layer_['conv3']['conv_stride'] = stride1
        layer_['conv3']['conv_padding'] = 0
        layer_['conv3']['conv_bias'] = if_conv_bias

        if if_BN_shortcut:
            assert if_BNNoAffine == False
            layer_['BN3'] = {}
            layer_['BN3']['num_features'] = out_channels_1
    layers_.append(layer_)
    return layers_

def add_conv_block(layers_, in_channels, out_channels, kernel_size, stride, padding, params):
    layer_2 = {}
    if params['name_dataset'] in ['CIFAR-10-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias-no-regularization',
                                   'SVHN-ResNet34']:
        layer_2['name'] = 'conv-no-bias-no-activation'
    elif params['name_dataset'] in ['CIFAR-10-AllCNNC',
                                    'CIFAR-10-N1-128-AllCNNC',
                                    'CIFAR-10-N1-512-AllCNNC',
                                    'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN-no-regularization',
                                    'CIFAR-100-onTheFly-vgg16-NoLinear-BN-no-regularization',
                                    'CIFAR-100-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout-no-regularization',
                                    'CIFAR-100-onTheFly-AllCNNC',
                                    'SVHN-vgg11']:
        layer_2['name'] = 'conv-no-activation'
    else:
        print('params[name_dataset]')
        print(params['name_dataset'])
        sys.exit()
    
    layer_2['conv_in_channels'] = in_channels
    layer_2['conv_out_channels'] = out_channels
    layer_2['conv_kernel_size'] = kernel_size
    layer_2['conv_stride'] = stride
    layer_2['conv_padding'] = padding
    layer_2['activation'] = None
    layers_.append(layer_2)
    
    if params['name_dataset'] in ['CIFAR-10-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias-no-regularization',
                                  'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN-no-regularization',
                                  'CIFAR-100-onTheFly-vgg16-NoLinear-BN-no-regularization',
                                  'SVHN-ResNet34', 'SVHN-vgg11']:
        layer_2 = {}
        layer_2['name'] = 'BN'
        layer_2['num_features'] = out_channels
        layer_2['activation'] = None
        layers_.append(layer_2)
    elif params['name_dataset'] in ['CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine-no-regularization',
                                    'CIFAR-100-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine']:
        layer_2 = {}
        layer_2['name'] = 'BNNoAffine'
        layer_2['num_features'] = out_channels
        layer_2['activation'] = None
        layers_.append(layer_2)
    else:
        1
    layer_2 = {}
    layer_2['name'] = 'relu'
    layers_.append(layer_2)
    return layers_

def get_post_activation(pre_, activation):
    if activation == 'relu':
        post_ = F.relu(pre_)
    elif activation == 'sigmoid':
        post_ = torch.sigmoid(pre_)
    elif activation == 'tanh':
        post_ = torch.tanh(pre_)
    elif activation == 'linear':
        post_ = pre_
    else:
        print('Error: unsupported activation for ' + activation)
        sys.exit()
    return post_

def get_layer_forward(input_, layer_, activation_, layer_params):
    if layer_params['name'] == 'fully-connected':
            
        a_ = layer_(input_)
        h_ = get_post_activation(a_, activation_)
        a_.retain_grad()
        
        output_ = h_
        pre_ = a_
        
    elif layer_params['name'] in ['conv',
                                  '1d-conv']:
        
        
        a_ = layer_(input_)

        
        h_ = get_post_activation(a_, activation_)
        

        a_.retain_grad()
        
        output_ = h_
        pre_ = a_
        
    elif layer_params['name'] in ['conv-no-activation',
                                  'conv-no-bias-no-activation']:
        a_ = layer_(input_)
        

        
        a_.retain_grad()
        output_ = a_
        pre_ = a_
        
    elif layer_params['name'] in ['BN']:
        a_ = layer_(input_)
        a_.retain_grad()
        output_ = a_
        pre_ = a_
        
    else:
        print('layer_params[name]')
        print(layer_params['name'])
        print('Error: unkown layer')
        sys.exit()
    

    return output_, pre_

def get_layers_params(name_model, layersizes, activations, params):
    if name_model == 'fully-connected':
        layers_ = []
        for l in range(len(layersizes) - 1):
            layer_i = {}
            layer_i['name'] = 'fully-connected'
            layer_i['input_size'] = layersizes[l]
            layer_i['output_size'] = layersizes[l+1]
            layer_i['activation'] = activations[l]
            layers_.append(layer_i)
    elif name_model == 'simple-CNN':
        # https://arxiv.org/pdf/1910.05446.pdf
        # https://arxiv.org/pdf/1811.03600.pdf

        # "same" padding:
        # i.e. H_in = H_out
        # by https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
        # H_out = H_in + 2 * padding - dilation * (kernel_size - 1)
        # (when stride = 1)
        # Hence, since dilation = 1, H_in = H_out => padding = (kernel_size - 1) / 2
        # this is also endorsed by https://discuss.pytorch.org/t/same-convolution-in-pytorch/19937
        

        layersizes = [32, 64, 1024]

        layers_ = []

        layer_1 = {}
        layer_1['name'] = 'conv'
        
        
        if params['name_dataset'] == 'Subsampled-ImageNet-simple-CNN':
            layer_1['conv_in_channels'] = 3
        elif params['name_dataset'] in ['Fashion-MNIST',
                                        'Fashion-MNIST-N1-60',
                                        'Fashion-MNIST-N1-60-no-regularization',
                                        'Fashion-MNIST-N1-256-no-regularization',
                                        'Fashion-MNIST-GAP-N1-60-no-regularization']:
            layer_1['conv_in_channels'] = 1
        else:
            print('error: need to check conv_in_channels for ' + params['name_dataset'])
            sys.exit()
            
        
        layer_1['conv_out_channels'] = layersizes[0]
        layer_1['conv_kernel_size'] = 5
        layer_1['conv_stride'] = 1
        layer_1['conv_padding'] = int((layer_1['conv_kernel_size'] - 1)/2)
        
        

        layer_1['activation'] = 'relu'
    
        layers_.append(layer_1)
        
        layer_1 = {}
        layer_1['name'] = 'max_pool'
        layer_1['max_pool_kernel_size'] = 2
        layer_1['max_pool_stride'] = 2
        
        layers_.append(layer_1)
        

        layer_2 = {}
        layer_2['name'] = 'conv'
        layer_2['conv_in_channels'] = layersizes[0]
        layer_2['conv_out_channels'] = layersizes[1]
        layer_2['conv_kernel_size'] = 5
        layer_2['conv_stride'] = 1
        layer_2['conv_padding'] = int((layer_2['conv_kernel_size'] - 1)/2)
        

        layer_2['activation'] = 'relu'
        
        layers_.append(layer_2)
        
        if params['name_dataset'] == 'Fashion-MNIST-GAP-N1-60-no-regularization':
            # https://teaching.pages.centralesupelec.fr/deeplearning-lectures-build/00-pytorch-fashionMnist.html
            
            layer_ = {}
            layer_['name'] = 'global_average_pooling'
            layers_.append(layer_)
    
        elif params['name_dataset'] in ['Fashion-MNIST-N1-60-no-regularization',
                                        'Fashion-MNIST-N1-256-no-regularization']:
            
            layer_2 = {}
            layer_2['name'] = 'max_pool'
            layer_2['max_pool_kernel_size'] = 2
            layer_2['max_pool_stride'] = 2

            layers_.append(layer_2)


            layer_5 = {}
            layer_5['name'] = 'flatten'

            layers_.append(layer_5)
    
        else:
            print('error: need to check for ' + params['name_dataset'])
            sys.exit()
        
            
        

        layer_3 = {}
        layer_3['name'] = 'fully-connected'
        
        if params['name_dataset'] == 'Subsampled-ImageNet-simple-CNN':
            layer_3['input_size'] = 64 * 64 * layersizes[1]
        elif params['name_dataset'] in ['Fashion-MNIST',
                                        'Fashion-MNIST-N1-60',
                                        'Fashion-MNIST-N1-60-no-regularization',
                                        'Fashion-MNIST-N1-256-no-regularization']:
            layer_3['input_size'] = 7 * 7 * layersizes[1]
        elif params['name_dataset'] in ['Fashion-MNIST-GAP-N1-60-no-regularization']:
            layer_3['input_size'] = layersizes[1]
        else:
            print('error: need to check input_size for ' + params['name_dataset'])
            sys.exit()
            
        
        layer_3['output_size'] = layersizes[2]
        
        layer_3['activation'] = 'relu'

        layers_.append(layer_3)
        

        layer_4 = {}
        layer_4['name'] = 'fully-connected'
        layer_4['input_size'] = layer_3['output_size']
        
        if params['name_dataset'] == 'Subsampled-ImageNet-simple-CNN':
            layer_4['output_size'] = 200
        elif params['name_dataset'] in ['Fashion-MNIST',
                                        'Fashion-MNIST-N1-60',
                                        'Fashion-MNIST-N1-60-no-regularization',
                                        'Fashion-MNIST-N1-256-no-regularization',
                                        'Fashion-MNIST-GAP-N1-60-no-regularization']:
            layer_4['output_size'] = 10
        else:
            print('error: need to check output_size for ' + params['name_dataset'])
            sys.exit()
            
        layer_4['activation'] = 'linear'

        layers_.append(layer_4)
        
        
    elif name_model == 'CNN':
        layersizes = [96, 192, 192, 100]
        
        layers_ = []
        
        for l in range(3):
        
            layer_l = {}
            layer_l['name'] = 'conv'
            
            if l == 0:
                layer_l['conv_in_channels'] = 3
            else:
                layer_l['conv_in_channels'] = layersizes[l-1]
            
            
            layer_l['conv_out_channels'] = layersizes[l]
            
            layer_l['conv_kernel_size'] = 5
            layer_l['conv_stride'] = 1
            layer_l['conv_padding'] = int((layer_l['conv_kernel_size'] - 1)/2)
            
            layer_l['activation'] = 'relu'
    
            layers_.append(layer_l)
            
            layer_l = {}
            layer_l['name'] = 'max_pool'
            layer_l['max_pool_kernel_size'] = 2
            layer_l['max_pool_stride'] = 2
            layers_.append(layer_l)
            
        layer_5 = {}
        layer_5['name'] = 'flatten'
        layers_.append(layer_5)
            
        layer_4 = {}
        layer_4['name'] = 'fully-connected'
        layer_4['input_size'] = layersizes[2] * 4 * 4
        layer_4['output_size'] = layersizes[3]
        

        layer_4['activation'] = 'linear'
        
        
        layers_.append(layer_4)
    elif name_model == '1d-CNN':
        # https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/
        
        layersizes = [64, 64, 100, 6]
        
        layers_ = []
        
        for l in range(2):
        
            layer_l = {}
            layer_l['name'] = '1d-conv'
            
            if l == 0:
                layer_l['conv_in_channels'] = 1
            else:
                layer_l['conv_in_channels'] = layersizes[l-1]
            
            
            layer_l['conv_out_channels'] = layersizes[l]
            
            layer_l['conv_kernel_size'] = 3
            layer_l['conv_stride'] = 1
            layer_l['conv_padding'] = 0
            
            layer_l['activation'] = activations[l]
            
            layers_.append(layer_l)
            
        layer_6 = {}
        layer_6['name'] = 'max_pool_1d'
        layer_6['max_pool_kernel_size'] = 2
        layer_6['max_pool_stride'] = 0
        layers_.append(layer_6)
            
        layer_5 = {}
        layer_5['name'] = 'flatten'
        layers_.append(layer_5)
            
        layer_3 = {}
        layer_3['name'] = 'fully-connected'
        layer_3['input_size'] = layersizes[1] * 278
        layer_3['output_size'] = layersizes[2]
        layer_3['activation'] = activations[2]
        
        layers_.append(layer_3)
        
        layer_4 = {}
        layer_4['name'] = 'fully-connected'
        layer_4['input_size'] = layersizes[2]
        layer_4['output_size'] = layersizes[3]
        layer_4['activation'] = activations[3]
        
        layers_.append(layer_4)
        
    elif name_model == 'AllCNNC':
        # https://arxiv.org/pdf/1412.6806.pdf
        # https://github.com/mateuszbuda/ALL-CNN/blob/master/ALL_CNN_C.png
        
        # except that in the following, for the last 3 * 3 conv in Table 1,
        # we use padding = 1
        
        # see also:
        # 'CNN' in https://arxiv.org/pdf/1910.05446.pdf
        
        layers_ = []
        
        layers_ = add_conv_block(layers_, 3, 96, 3, 1, 1, params)
        layers_ = add_conv_block(layers_, 96, 96, 3, 1, 1, params)
        layers_ = add_conv_block(layers_, 96, 96, 3, 2, 1, params)
        
        layers_ = add_conv_block(layers_, 96, 192, 3, 1, 1, params)
        layers_ = add_conv_block(layers_, 192, 192, 3, 1, 1, params)
        layers_ = add_conv_block(layers_, 192, 192, 3, 2, 1, params)
        
        layers_ = add_conv_block(layers_, 192, 192, 3, 1, 1, params)
        
        layers_ = add_conv_block(layers_, 192, 192, 1, 1, 0, params)

        
        if params['name_dataset'] == 'CIFAR-100-onTheFly-AllCNNC':
            layers_ = add_conv_block(layers_, 192, 100, 1, 1, 0, params)
        elif params['name_dataset'] in ['CIFAR-10-AllCNNC',
                                        'CIFAR-10-N1-128-AllCNNC',
                                        'CIFAR-10-N1-512-AllCNNC']:
            layers_ = add_conv_block(layers_, 192, 10, 1, 1, 0, params)
        else:
            print('params[name_dataset]')
            print(params['name_dataset'])
            sys.exit()
        
        
        layer_ = {}
        layer_['name'] = 'global_average_pooling'
        layers_.append(layer_)
        
    elif name_model == 'ConvPoolCNNC':
        # https://arxiv.org/pdf/1412.6806.pdf
        
        layers_ = []
        
        layers_ = add_conv_block(layers_, 3, 96, params)
        
        layers_ = add_conv_block(layers_, 96, 96, params)
        
        layers_ = add_conv_block(layers_, 96, 96, params)
        
        
        
        sys.exit()
        
    elif name_model in ['ResNet32']:
        
        # ResNet 32 in sec 4.2 and Table 6 of https://arxiv.org/pdf/1512.03385.pdf
        
        # see also:
        # https://pytorch.org/vision/stable/models.html
        
        # see also:
        # https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
        
        # see also:
        # https://github.com/km1414/CNN-models/blob/master/resnet-32/resnet-32.py
        # (this one seems to have bias)
        
        # in the NoBias mode, conv layers don't have bias, including the first conv
        # see e.g. https://pytorch.org/docs/stable/_modules/torchvision/models/resnet.html#resnet34
        # (this webpage is no longer available)
        
        
        
        
        if params['name_dataset'] in ['CIFAR-10-onTheFly-ResNet32-BN',
                                      'CIFAR-10-onTheFly-ResNet32-BN-BNshortcut',
                                      'CIFAR-10-onTheFly-ResNet32-BN-BNshortcutDownsampleOnly',
                                      'CIFAR-10-onTheFly-ResNet32-BN-BNshortcutDownsampleOnly-NoBias',
                                      'CIFAR-10-onTheFly-N1-128-ResNet32-BN-BNshortcutDownsampleOnly-NoBias',
                                      'CIFAR-10-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias',
                                      'CIFAR-10-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias-no-regularization',
                                      'CIFAR-100-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias',]:
            if_BNNoAffine = False
        elif params['name_dataset'] in ['CIFAR-10-onTheFly-ResNet32-BNNoAffine',
                                        'CIFAR-10-onTheFly-ResNet32-BNNoAffine-NoBias',
                                        'CIFAR-10-onTheFly-N1-128-ResNet32-BNNoAffine-PaddingShortcutDownsampleOnly-NoBias-no-regularization',]:
            if_BNNoAffine = True
        else:
            print('params[name_dataset]')
            print(params['name_dataset'])
            sys.exit()
            
        if params['name_dataset'] in ['CIFAR-10-onTheFly-N1-128-ResNet32-BNNoAffine-PaddingShortcutDownsampleOnly-NoBias-no-regularization',
                                      'CIFAR-10-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias',
                                      'CIFAR-10-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias-no-regularization',
                                      'CIFAR-100-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias',]:
            shortcut_type = 'padding'
            if_BN_shortcut = None
        elif params['name_dataset'] in ['CIFAR-10-onTheFly-ResNet32-BN-BNshortcutDownsampleOnly-NoBias',
                                        'CIFAR-10-onTheFly-N1-128-ResNet32-BN-BNshortcutDownsampleOnly-NoBias',]:
            shortcut_type = 'conv'
            if_BN_shortcut = True
        elif params['name_dataset'] in ['CIFAR-10-onTheFly-ResNet32-BNNoAffine-NoBias',]:
            shortcut_type = 'conv'
            if_BN_shortcut = False
        else:
            print('params[name_dataset]')
            print(params['name_dataset'])
            sys.exit()
            
            if params['name_dataset'] in ['CIFAR-10-onTheFly-ResNet32-BN-BNshortcut',
                                          'CIFAR-10-onTheFly-ResNet32-BN-BNshortcutDownsampleOnly',]:
                if_BN_shortcut = True
            elif params['name_dataset'] in ['CIFAR-10-onTheFly-ResNet32-BNNoAffine',
                                            'CIFAR-10-onTheFly-ResNet32-BN']:
                if_BN_shortcut = False
            else:
                print('params[name_dataset]')
                print(params['name_dataset'])

                sys.exit()
            
        if params['name_dataset'] in ['CIFAR-10-onTheFly-ResNet32-BN-BNshortcutDownsampleOnly',
                                      'CIFAR-10-onTheFly-ResNet32-BN-BNshortcutDownsampleOnly-NoBias',
                                      'CIFAR-10-onTheFly-N1-128-ResNet32-BNNoAffine-PaddingShortcutDownsampleOnly-NoBias-no-regularization',
                                      'CIFAR-10-onTheFly-N1-128-ResNet32-BN-BNshortcutDownsampleOnly-NoBias',
                                      'CIFAR-10-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias',
                                      'CIFAR-10-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias-no-regularization',
                                      'CIFAR-100-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias',]:
            if_downsample_only = True
        elif params['name_dataset'] in ['CIFAR-10-onTheFly-ResNet32-BNNoAffine-NoBias',]:
            if_downsample_only = False
        else:
            print('params[name_dataset]')
            print(params['name_dataset'])

            sys.exit()
            
            
        if params['name_dataset'] in ['CIFAR-10-onTheFly-ResNet32-BNNoAffine-NoBias',
                                      'CIFAR-10-onTheFly-ResNet32-BN-BNshortcutDownsampleOnly-NoBias',
                                      'CIFAR-10-onTheFly-N1-128-ResNet32-BNNoAffine-PaddingShortcutDownsampleOnly-NoBias-no-regularization',
                                      'CIFAR-10-onTheFly-N1-128-ResNet32-BN-BNshortcutDownsampleOnly-NoBias',
                                      'CIFAR-10-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias',
                                      'CIFAR-10-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias-no-regularization',
                                      'CIFAR-100-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias',]:
            if_conv_bias = False
        elif params['name_dataset'] in ['CIFAR-10-onTheFly-ResNet32-BN-BNshortcutDownsampleOnly']:
            if_conv_bias = True
        else:
            print('params[name_dataset]')
            print(params['name_dataset'])

            sys.exit()
            
        

        
        
        layers_ = []
        
        layers_ = add_conv_block(layers_, 3, 16, 3, 1, 1, params)
        
        layers_ = add_res_block(layers_, 16, 16, 1, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        layers_ = add_res_block(layers_, 16, 16, 1, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        layers_ = add_res_block(layers_, 16, 16, 1, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        layers_ = add_res_block(layers_, 16, 16, 1, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        layers_ = add_res_block(layers_, 16, 16, 1, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        
        layers_ = add_res_block(layers_, 16, 32, 2, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        layers_ = add_res_block(layers_, 32, 32, 1, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        layers_ = add_res_block(layers_, 32, 32, 1, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        layers_ = add_res_block(layers_, 32, 32, 1, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        layers_ = add_res_block(layers_, 32, 32, 1, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        
        layers_ = add_res_block(layers_, 32, 64, 2, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        layers_ = add_res_block(layers_, 64, 64, 1, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        layers_ = add_res_block(layers_, 64, 64, 1, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        layers_ = add_res_block(layers_, 64, 64, 1, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        layers_ = add_res_block(layers_, 64, 64, 1, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        
        layer_ = {}
        layer_['name'] = 'global_average_pooling'
        layers_.append(layer_)
        
        layer_ = {}
        layer_['name'] = 'fully-connected'
        layer_['input_size'] = 64
        
        if params['name_dataset'] in ['CIFAR-100-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias',]:
            layer_['output_size'] = 100
        elif params['name_dataset'] in ['CIFAR-10-onTheFly-N1-128-ResNet32-BNNoAffine-PaddingShortcutDownsampleOnly-NoBias-no-regularization',
                                        'CIFAR-10-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias',
                                        'CIFAR-10-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias-no-regularization',]:
            layer_['output_size'] = 10
        else:
            print('params[name_dataset]')
            print(params['name_dataset'])
            sys.exit()
        
            
        
        layer_['activation'] = 'linear'
        layers_.append(layer_)
        
    elif name_model in ['ResNet34']:
        
        # ResNet 34 in Fig 3 of https://arxiv.org/pdf/1512.03385.pdf
        
        # see also
        # https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/resnet.py
        
        if params['name_dataset'] == 'CIFAR-100-onTheFly-ResNet34-BNNoAffine':
            if_BNNoAffine = True
            shortcut_type = 'conv'
            if_BN_shortcut = False
            if_downsample_only = False
            if_conv_bias = True
        elif params['name_dataset'] == 'CIFAR-100-onTheFly-ResNet34-BN':
            if_BNNoAffine = False
            shortcut_type = 'conv'
            if_BN_shortcut = False
            if_downsample_only = False
            if_conv_bias = True
        elif params['name_dataset'] == 'CIFAR-100-onTheFly-ResNet34-BN-BNshortcut':
            if_BNNoAffine = False
            shortcut_type = 'conv'
            if_BN_shortcut = True
            if_downsample_only = False
            if_conv_bias = True
        elif params['name_dataset'] == 'CIFAR-100-onTheFly-ResNet34-BN-BNshortcutDownsampleOnly':
            if_BNNoAffine = False
            shortcut_type = 'conv'
            if_BN_shortcut = True
            if_downsample_only = True
            if_conv_bias = True
        elif params['name_dataset'] in ['CIFAR-100-onTheFly-ResNet34-BN-BNshortcutDownsampleOnly-NoBias',
                                        'CIFAR-100-onTheFly-N1-128-ResNet34-BN-BNshortcutDownsampleOnly-NoBias',]:
            if_BNNoAffine = False
            shortcut_type = 'conv'
            if_BN_shortcut = True
            if_downsample_only = True
            if_conv_bias = False
        elif params['name_dataset'] == 'CIFAR-100-onTheFly-N1-128-ResNet34-BN-PaddingShortcutDownsampleOnly-NoBias':
            if_BNNoAffine = False
            shortcut_type = 'padding'
            if_BN_shortcut = None
            if_downsample_only = True
            if_conv_bias = False
        elif params['name_dataset'] == 'SVHN-ResNet34':
            if_BNNoAffine = False
            shortcut_type = 'padding'
            if_BN_shortcut = None
            if_downsample_only = True
            if_conv_bias = False
        else:
            print('params[name_dataset]')
            print(params['name_dataset'])
            sys.exit()
        
        
        layers_ = []
        
        layers_ = add_conv_block(layers_, 3, 64, 3, 1, 1, params)
        
        
        
        layers_ = add_res_block(layers_, 64, 64, 1, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        layers_ = add_res_block(layers_, 64, 64, 1, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        layers_ = add_res_block(layers_, 64, 64, 1, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        
        layers_ = add_res_block(layers_, 64, 128, 2, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        layers_ = add_res_block(layers_, 128, 128, 1, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        layers_ = add_res_block(layers_, 128, 128, 1, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        layers_ = add_res_block(layers_, 128, 128, 1, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        
        layers_ = add_res_block(layers_, 128, 256, 2, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        layers_ = add_res_block(layers_, 256, 256, 1, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        layers_ = add_res_block(layers_, 256, 256, 1, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        layers_ = add_res_block(layers_, 256, 256, 1, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        layers_ = add_res_block(layers_, 256, 256, 1, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        layers_ = add_res_block(layers_, 256, 256, 1, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        
        layers_ = add_res_block(layers_, 256, 512, 2, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        layers_ = add_res_block(layers_, 512, 512, 1, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        layers_ = add_res_block(layers_, 512, 512, 1, if_BNNoAffine, shortcut_type, if_BN_shortcut, if_downsample_only, if_conv_bias)
        
        
        
        layer_ = {}
        layer_['name'] = 'global_average_pooling'
        layers_.append(layer_)
        
        layer_ = {}
        layer_['name'] = 'fully-connected'
        layer_['input_size'] = 512
        
        if params['name_dataset'] == 'SVHN-ResNet34':
            layer_['output_size'] = 10
        else:
            layer_['output_size'] = 100
            
        layer_['activation'] = 'linear'
        layers_.append(layer_)
    
    elif name_model == 'vgg16':
        # referece:
        # import torchvision.models as models
        # vgg16 = models.vgg16()
        # print(vgg16.eval())
        
        # model D in https://arxiv.org/pdf/1409.1556.pdf
        
        # when use BN (or BNNoAffine), bias term is NOT omitted in conv
        # see e.g. https://pytorch.org/docs/stable/_modules/torchvision/models/vgg.html#vgg16_bn
        # see also https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py
        
        
        if params['name_dataset'] == 'Subsampled-ImageNet-vgg16':
            print('error: need to check all below')
            sys.exit()
            
        if params['name_dataset'] == 'CIFAR-10-vgg16':
            print('error: not supported anymore')
            sys.exit()
            
            
        
        
        if params['name_dataset'] == 'CIFAR-10-onTheFly-N1-256-vgg16-NoAdaptiveAvgPool':
            print('error: not supported anymore')
            sys.exit()
        
        
        
        layers_ = []
        
        layers_ = add_conv_block(layers_, 3, 64, 3, 1, 1, params)
        
        layers_ = add_conv_block(layers_, 64, 64, 3, 1, 1, params)
        
        layer_ = {}
        layer_['name'] = 'max_pool'
        layer_['max_pool_kernel_size'] = 2
        layer_['max_pool_stride'] = 2
        layers_.append(layer_)
        
        layers_ = add_conv_block(layers_, 64, 128, 3, 1, 1, params)
        
        layers_ = add_conv_block(layers_, 128, 128, 3, 1, 1, params)
        
        layer_ = {}
        layer_['name'] = 'max_pool'
        layer_['max_pool_kernel_size'] = 2
        layer_['max_pool_stride'] = 2
        layers_.append(layer_)
        
        layers_ = add_conv_block(layers_, 128, 256, 3, 1, 1, params)
        
        layers_ = add_conv_block(layers_, 256, 256, 3, 1, 1, params)
        
        layers_ = add_conv_block(layers_, 256, 256, 3, 1, 1, params)
        
        layer_ = {}
        layer_['name'] = 'max_pool'
        layer_['max_pool_kernel_size'] = 2
        layer_['max_pool_stride'] = 2
        layers_.append(layer_)

        
        layers_ = add_conv_block(layers_, 256, 512, 3, 1, 1, params)
     
        
        layers_ = add_conv_block(layers_, 512, 512, 3, 1, 1, params)
       
        
        layers_ = add_conv_block(layers_, 512, 512, 3, 1, 1, params)
        
        layer_ = {}
        layer_['name'] = 'max_pool'
        layer_['max_pool_kernel_size'] = 2
        layer_['max_pool_stride'] = 2
        layers_.append(layer_)

        
        layers_ = add_conv_block(layers_, 512, 512, 3, 1, 1, params)
   
        layers_ = add_conv_block(layers_, 512, 512, 3, 1, 1, params)
        
        layers_ = add_conv_block(layers_, 512, 512, 3, 1, 1, params)
        
        layer_ = {}
        layer_['name'] = 'max_pool'
        layer_['max_pool_kernel_size'] = 2
        layer_['max_pool_stride'] = 2
        layers_.append(layer_)
     
  
    

        
        layer_5 = {}
        layer_5['name'] = 'flatten'
        layers_.append(layer_5)
        

        
        if params['name_dataset'] in ['CIFAR-100-onTheFly-vgg16-NoLinear-BN-no-regularization']:
            layer_ = {}
            layer_['name'] = 'fully-connected'
            layer_['input_size'] = 512
            layer_['output_size'] = 100
            layer_['activation'] = 'linear'
            layers_.append(layer_)
        elif params['name_dataset'] in ['CIFAR-10-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout',
                                        'CIFAR-10-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout-no-regularization',
                                        'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN-no-regularization',
                                        'CIFAR-100-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout-no-regularization',]:
        
        

            layer_ = {}
            layer_['name'] = 'fully-connected'

            layer_['input_size'] = 512
            layer_['output_size'] = 4096
            layer_['activation'] = 'relu'
            layers_.append(layer_)

            layer_ = {}
            layer_['name'] = 'fully-connected'
            layer_['input_size'] = 4096
            layer_['output_size'] = 4096
            layer_['activation'] = 'relu'
            layers_.append(layer_)


            layer_ = {}
            layer_['name'] = 'fully-connected'
            layer_['input_size'] = 4096

            if params['name_dataset'] in ['CIFAR-10-vgg16-NoAdaptiveAvgPoolNoDropout',
                                          'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout',
                                          'CIFAR-10-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout',
                                          'CIFAR-10-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout-no-regularization',
                                          'CIFAR-10-onTheFly-N1-512-vgg16-NoAdaptiveAvgPoolNoDropout',
                                          'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN',
                                          'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN-NoBias',
                                          'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine',
                                          'CIFAR-10-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout',
                                          'CIFAR-10-NoAugmentation-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout',
                                          'CIFAR-10-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout-BN',
                                          'CIFAR-10-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine',]:
                layer_['output_size'] = 10
            elif params['name_dataset'] in ['CIFAR-100-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout',
                                            'CIFAR-100-NoAugmentation-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout',
                                            'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout',
                                            'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine-no-regularization',
                                            'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN',
                                            'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN-no-regularization',
                                            'CIFAR-100-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout',
                                            'CIFAR-100-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout-no-regularization',
                                            'CIFAR-100-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine']:
                layer_['output_size'] = 100
            else:
                print('error: need to check for ' + params['name_dataset'])
                sys.exit()
                layer_['output_size'] = 1000


            layer_['activation'] = 'linear'
            layers_.append(layer_)
            
        else:
            print('params[name_dataset]')
            print(params['name_dataset'])
            sys.exit()
        
    elif name_model in ['vgg11']:
        
        
        
        # referece:
        # https://arxiv.org/pdf/1409.1556.pdf
        # Table 1, model A
        
        
        
        layers_ = []
        
        
        
        layer_2 = {}
        layer_2['name'] = 'conv'
        layer_2['conv_in_channels'] = 3
        layer_2['conv_out_channels'] = 64
        layer_2['conv_kernel_size'] = 3
        layer_2['conv_stride'] = 1
        layer_2['conv_padding'] = 1
        layer_2['activation'] = 'relu'
        layers_.append(layer_2)
        
        layer_ = {}
        layer_['name'] = 'max_pool'
        layer_['max_pool_kernel_size'] = 2
        layer_['max_pool_stride'] = 2
        layers_.append(layer_)
        
        # working here
        

        layer_2 = {}
        layer_2['name'] = 'conv'
        layer_2['conv_in_channels'] = 64
        layer_2['conv_out_channels'] = 128
        layer_2['conv_kernel_size'] = 3
        layer_2['conv_stride'] = 1
        layer_2['conv_padding'] = 1
        layer_2['activation'] = 'relu'
        layers_.append(layer_2)
        

        layer_ = {}
        layer_['name'] = 'max_pool'
        layer_['max_pool_kernel_size'] = 2
        layer_['max_pool_stride'] = 2
        layers_.append(layer_)
        
        # working here
        

        layer_2 = {}
        layer_2['name'] = 'conv'
        layer_2['conv_in_channels'] = 128
        layer_2['conv_out_channels'] = 256
        layer_2['conv_kernel_size'] = 3
        layer_2['conv_stride'] = 1
        layer_2['conv_padding'] = 1
        layer_2['activation'] = 'relu'
        layers_.append(layer_2)
        

        layer_2 = {}
        layer_2['name'] = 'conv'
        layer_2['conv_in_channels'] = 256
        layer_2['conv_out_channels'] = 256
        layer_2['conv_kernel_size'] = 3
        layer_2['conv_stride'] = 1
        layer_2['conv_padding'] = 1
        layer_2['activation'] = 'relu'
        layers_.append(layer_2)
        

        layer_ = {}
        layer_['name'] = 'max_pool'
        layer_['max_pool_kernel_size'] = 2
        layer_['max_pool_stride'] = 2
        layers_.append(layer_)
        
        # working here
        
        
        

        layer_2 = {}
        layer_2['name'] = 'conv'
        layer_2['conv_in_channels'] = 256
        layer_2['conv_out_channels'] = 512
        layer_2['conv_kernel_size'] = 3
        layer_2['conv_stride'] = 1
        layer_2['conv_padding'] = 1
        layer_2['activation'] = 'relu'
        layers_.append(layer_2)
        

        layer_2 = {}
        layer_2['name'] = 'conv'
        layer_2['conv_in_channels'] = 512
        layer_2['conv_out_channels'] = 512
        layer_2['conv_kernel_size'] = 3
        layer_2['conv_stride'] = 1
        layer_2['conv_padding'] = 1
        layer_2['activation'] = 'relu'
        layers_.append(layer_2)
        

        layer_ = {}
        layer_['name'] = 'max_pool'
        layer_['max_pool_kernel_size'] = 2
        layer_['max_pool_stride'] = 2
        layers_.append(layer_)
        
        # start to not work here
        
        

        layer_2 = {}
        layer_2['name'] = 'conv'
        layer_2['conv_in_channels'] = 512
        layer_2['conv_out_channels'] = 512
        layer_2['conv_kernel_size'] = 3
        layer_2['conv_stride'] = 1
        layer_2['conv_padding'] = 1
        layer_2['activation'] = 'relu'
        layers_.append(layer_2)
        

        layer_2 = {}
        layer_2['name'] = 'conv'
        layer_2['conv_in_channels'] = 512
        layer_2['conv_out_channels'] = 512
        layer_2['conv_kernel_size'] = 3
        layer_2['conv_stride'] = 1
        layer_2['conv_padding'] = 1
        layer_2['activation'] = 'relu'
        layers_.append(layer_2)
        

        layer_ = {}
        layer_['name'] = 'max_pool'
        layer_['max_pool_kernel_size'] = 2
        layer_['max_pool_stride'] = 2
        layers_.append(layer_)
        

        layer_5 = {}
        layer_5['name'] = 'flatten'
        layers_.append(layer_5)

        

        layer_ = {}
        layer_['name'] = 'fully-connected'
        layer_['input_size'] = 512

        layer_['output_size'] = 4096
        layer_['activation'] = 'relu'
        layers_.append(layer_)
    

        layer_ = {}
        layer_['name'] = 'fully-connected'
        layer_['input_size'] = 4096
        layer_['output_size'] = 4096
        layer_['activation'] = 'relu'
        layers_.append(layer_)
        


        layer_ = {}
        layer_['name'] = 'fully-connected'
        layer_['input_size'] = 4096
        layer_['output_size'] = 10
        layer_['activation'] = 'linear'
        layers_.append(layer_)
    
    else:
        print('Error: unknown model name in get_layers_params for ' + name_model)
        sys.exit()
    return layers_

class Model_3(nn.Module):
    def __init__(self, params):
        name_dataset = params['name_dataset']
        super(Model_3, self).__init__()
        self.name_loss = params['name_loss']
        if name_dataset in ['MNIST-autoencoder',
                            'MNIST-autoencoder-no-regularization',
                            'MNIST-autoencoder-N1-1000',
                            'MNIST-autoencoder-N1-1000-no-regularization',
                            'MNIST-autoencoder-N1-1000-sum-loss-no-regularization',
                            'MNIST-autoencoder-relu-N1-1000-sum-loss-no-regularization',
                            'MNIST-autoencoder-relu-N1-1000-sum-loss',
                            'MNIST-autoencoder-relu-N1-100-sum-loss',
                            'MNIST-autoencoder-relu-N1-500-sum-loss',
                            'MNIST-autoencoder-relu-N1-1-sum-loss',
                            'MNIST-autoencoder-reluAll-N1-1-sum-loss',
                            'MNIST-autoencoder-N1-1000-sum-loss',
                            'CURVES-autoencoder',
                            'CURVES-autoencoder-no-regularization',
                            'CURVES-autoencoder-sum-loss-no-regularization',
                            'CURVES-autoencoder-sum-loss',
                            'CURVES-autoencoder-relu-sum-loss-no-regularization',
                            'CURVES-autoencoder-relu-sum-loss',
                            'CURVES-autoencoder-relu-N1-100-sum-loss',
                            'CURVES-autoencoder-relu-N1-500-sum-loss',
                            'CURVES-autoencoder-Botev',
                            'CURVES-autoencoder-Botev-sum-loss-no-regularization',
                            'CURVES-autoencoder-shallow',
                            'FACES-autoencoder',
                            'FACES-autoencoder-no-regularization',
                            'FACES-autoencoder-sum-loss-no-regularization',
                            'FACES-autoencoder-relu-sum-loss-no-regularization',
                            'FACES-autoencoder-relu-sum-loss',
                            'FACES-autoencoder-sum-loss',
                            'FacesMartens-autoencoder-relu',
                            'FacesMartens-autoencoder-relu-no-regularization',
                            'FacesMartens-autoencoder-relu-N1-500',
                            'FacesMartens-autoencoder-relu-N1-100',
                            'MNIST',
                            'MNIST-no-regularization',
                            'MNIST-N1-1000',
                            'MNIST-one-layer',
                            'DownScaledMNIST-no-regularization',
                            'DownScaledMNIST-N1-1000-no-regularization',
                            'webspam',
                            'CIFAR',
                            'CIFAR-deep',
                            'sythetic-linear-regression',
                            'sythetic-linear-regression-N1-1']:
            self.name_model = 'fully-connected'
            
        elif name_dataset in ['Fashion-MNIST',
                              'Fashion-MNIST-N1-60',
                              'Fashion-MNIST-N1-60-no-regularization',
                              'Fashion-MNIST-N1-256-no-regularization',
                              'Fashion-MNIST-GAP-N1-60-no-regularization',
                              'STL-10-simple-CNN',
                              'Subsampled-ImageNet-simple-CNN']:
            # https://arxiv.org/pdf/1910.05446.pdf
            self.name_model = 'simple-CNN'
        elif name_dataset in ['CIFAR-100',
                              'CIFAR-100-NoAugmentation']:
            self.name_model = 'CNN'
        elif name_dataset in ['CIFAR-10-AllCNNC',
                              'CIFAR-10-N1-128-AllCNNC',
                              'CIFAR-10-N1-512-AllCNNC',
                              'CIFAR-100-onTheFly-AllCNNC']:
            self.name_model = 'AllCNNC'
        elif name_dataset == 'CIFAR-10-ConvPoolCNNC':
            self.name_model = 'ConvPoolCNNC'
        elif name_dataset == 'UCI-HAR':
            self.name_model = '1d-CNN'
        elif name_dataset in ['CIFAR-10-vgg16',
                              'CIFAR-10-vgg16-NoAdaptiveAvgPoolNoDropout',
                              'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout',
                              'CIFAR-10-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout',
                              'CIFAR-10-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout-no-regularization',
                              'CIFAR-10-onTheFly-N1-256-vgg16-NoAdaptiveAvgPool',
                              'CIFAR-10-onTheFly-N1-512-vgg16-NoAdaptiveAvgPoolNoDropout',
                              'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN',
                              'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN-NoBias',
                              'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine',
                              'CIFAR-10-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout',
                              'CIFAR-10-NoAugmentation-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout',
                              'CIFAR-10-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout-BN',
                              'CIFAR-10-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine',
                              'CIFAR-10-vgg16-GAP',
                              'CIFAR-100-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout',
                              'CIFAR-100-NoAugmentation-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout',
                              'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout',
                              'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine-no-regularization',
                              'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN',
                              'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN-no-regularization',
                              'CIFAR-100-onTheFly-vgg16-NoLinear-BN-no-regularization',
                              'CIFAR-100-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout',
                              'CIFAR-100-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout-no-regularization',
                              'CIFAR-100-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine',
                              'Subsampled-ImageNet-vgg16',]:
            self.name_model = 'vgg16'
        elif name_dataset in ['CIFAR-10-vgg11',
                              'CIFAR-10-NoAugmentation-vgg11', 'SVHN-vgg11']:
            self.name_model = 'vgg11'
        elif name_dataset in ['CIFAR-10-onTheFly-ResNet32-BNNoAffine',
                              'CIFAR-10-onTheFly-ResNet32-BN',
                              'CIFAR-10-onTheFly-ResNet32-BN-BNshortcut',
                              'CIFAR-10-onTheFly-ResNet32-BN-BNshortcutDownsampleOnly',
                              'CIFAR-10-onTheFly-ResNet32-BN-BNshortcutDownsampleOnly-NoBias',
                              'CIFAR-10-onTheFly-N1-128-ResNet32-BNNoAffine-PaddingShortcutDownsampleOnly-NoBias-no-regularization',
                              'CIFAR-10-onTheFly-N1-128-ResNet32-BN-BNshortcutDownsampleOnly-NoBias',
                              'CIFAR-10-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias',
                              'CIFAR-10-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias-no-regularization',
                              'CIFAR-10-onTheFly-ResNet32-BNNoAffine-NoBias',
                              'CIFAR-100-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias',]:
            self.name_model = 'ResNet32'
        elif name_dataset in ['CIFAR-100-onTheFly-ResNet34-BNNoAffine',
                              'CIFAR-100-onTheFly-ResNet34-BN',
                              'CIFAR-100-onTheFly-ResNet34-BN-BNshortcut',
                              'CIFAR-100-onTheFly-ResNet34-BN-BNshortcutDownsampleOnly',
                              'CIFAR-100-onTheFly-ResNet34-BN-BNshortcutDownsampleOnly-NoBias',
                              'CIFAR-100-onTheFly-N1-128-ResNet34-BN-BNshortcutDownsampleOnly-NoBias',
                              'CIFAR-100-onTheFly-N1-128-ResNet34-BN-PaddingShortcutDownsampleOnly-NoBias', 'SVHN-ResNet34']:
            self.name_model = 'ResNet34'
        else:
            print('Error: unkown dataset')
            sys.exit()
        if self.name_model == 'fully-connected':
            if name_dataset in ['MNIST',
                                'MNIST-no-regularization',
                                'MNIST-N1-1000']:
                layersizes = [784, 500, 10]
                self.activations_all = ['sigmoid', 'linear']
            elif name_dataset in ['DownScaledMNIST-no-regularization',
                                  'DownScaledMNIST-N1-1000-no-regularization']:
                # https://arxiv.org/pdf/1503.05671.pdf
                layersizes = [256, 20, 20, 20, 20, 20, 10]
                self.activations_all = ['tanh', 'tanh', 'tanh', 'tanh', 'tanh', 'linear']
            elif name_dataset == 'MNIST-one-layer':
                layersizes = [784, 10]
                self.activations_all = ['linear']
            elif name_dataset in ['sythetic-linear-regression',
                                  'sythetic-linear-regression-N1-1']:
                layersizes = [100, 50] # http://proceedings.mlr.press/v70/zhou17a/zhou17a.pdf

                self.activations_all = ['linear']
            elif name_dataset == 'CIFAR':
                
                layersizes = [3072, 400, 400, 10]
                self.activations = ['sigmoid', 'sigmoid', 'linear']
                
            elif name_dataset == 'CIFAR-deep':
                
                layersizes = [3072, 128, 128, 128, 128, 10]
                self.activations_all = ['relu', 'relu', 'relu', 'relu', 'linear']
                
            elif name_dataset == 'Fashion-MNIST':
                # self.layersizes = [784, 400, 400, 10]
                layersizes = [784, 400, 400, 10]
                self.activations = ['sigmoid', 'sigmoid', 'linear']
            elif name_dataset == 'webspam':
                
                layersizes = [254, 400, 400, 1]
                self.activations = ['sigmoid', 'sigmoid', 'linear']
            elif name_dataset in ['MNIST-autoencoder',
                                  'MNIST-autoencoder-no-regularization',
                                  'MNIST-autoencoder-N1-1000',
                                  'MNIST-autoencoder-N1-1000-no-regularization',
                                  'MNIST-autoencoder-N1-1000-sum-loss-no-regularization',
                                  'MNIST-autoencoder-N1-1000-sum-loss']:
                # reference: https://arxiv.org/pdf/1301.3641.pdf,
                # https://www.cs.toronto.edu/~hinton/science.pdf
                
                layersizes = [784, 1000, 500, 250, 30, 250, 500, 1000, 784]
                self.activations_all =\
                ['sigmoid', 'sigmoid', 'sigmoid', 'linear', 'sigmoid', 'sigmoid', 'sigmoid', 'linear']
                
            elif name_dataset in ['MNIST-autoencoder-relu-N1-1000-sum-loss-no-regularization',
                                  'MNIST-autoencoder-relu-N1-1000-sum-loss',
                                  'MNIST-autoencoder-relu-N1-100-sum-loss',
                                  'MNIST-autoencoder-relu-N1-500-sum-loss',
                                  'MNIST-autoencoder-relu-N1-1-sum-loss']:
                
                layersizes = [784, 1000, 500, 250, 30, 250, 500, 1000, 784]
                self.activations_all =\
                ['relu', 'relu', 'relu', 'linear', 'relu', 'relu', 'relu', 'linear']
                
            elif name_dataset in ['MNIST-autoencoder-reluAll-N1-1-sum-loss']:
                
                layersizes = [784, 1000, 500, 250, 30, 250, 500, 1000, 784]
                self.activations_all =\
                ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'linear']
            
            elif name_dataset in ['CURVES-autoencoder',
                                  'CURVES-autoencoder-no-regularization',
                                  'CURVES-autoencoder-sum-loss-no-regularization',
                                  'CURVES-autoencoder-sum-loss']:
                # https://www.cs.toronto.edu/~hinton/science.pdf
                # https://arxiv.org/pdf/1301.3641.pdf
                
                layersizes =\
                [784, 400, 200, 100, 50, 25, 6, 25, 50, 100, 200, 400, 784]
                self.activations_all =\
                ['sigmoid', 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid',
                 'linear',
                 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid', 'linear']
                
            elif name_dataset in ['CURVES-autoencoder-relu-sum-loss-no-regularization',
                                  'CURVES-autoencoder-relu-sum-loss',
                                  'CURVES-autoencoder-relu-N1-100-sum-loss',
                                  'CURVES-autoencoder-relu-N1-500-sum-loss']:
                
                layersizes =\
                [784, 400, 200, 100, 50, 25, 6, 25, 50, 100, 200, 400, 784]
                self.activations_all =\
                ['relu', 'relu', 'relu', 'relu', 'relu',
                 'linear',
                 'relu', 'relu', 'relu', 'relu', 'relu', 'linear']
                
            elif name_dataset in ['CURVES-autoencoder-Botev',
                                  'CURVES-autoencoder-Botev-sum-loss-no-regularization']:
                # https://arxiv.org/pdf/1706.03662.pdf
                layersizes = [784, 1000, 500, 250, 30, 250, 500, 1000, 784]
                self.activations_all =\
                ['sigmoid', 'sigmoid', 'sigmoid', 'linear', 'sigmoid', 'sigmoid', 'sigmoid', 'linear']
                
                
            elif name_dataset == 'CURVES-autoencoder-shallow':
                # https://www.cs.toronto.edu/~hinton/science.pdf
                
                layersizes = [784, 532, 6, 532, 784]
                self.activations =\
                ['sigmoid', 'linear', 'sigmoid', 'linear']
                
            elif name_dataset in ['FACES-autoencoder',
                                  'FACES-autoencoder-no-regularization',
                                  'FACES-autoencoder-sum-loss-no-regularization',
                                  'FACES-autoencoder-sum-loss']:
                # https://www.cs.toronto.edu/~hinton/science.pdf
                layersizes = [625, 2000, 1000, 500, 30,
                             500, 1000, 2000, 625]
                self.activations_all =\
                ['sigmoid', 'sigmoid', 'sigmoid', 'linear',
                 'sigmoid', 'sigmoid', 'sigmoid', 'linear']
                
            elif name_dataset in ['FACES-autoencoder-relu-sum-loss-no-regularization',
                                  'FACES-autoencoder-relu-sum-loss',
                                  'FacesMartens-autoencoder-relu',
                                  'FacesMartens-autoencoder-relu-no-regularization',
                                  'FacesMartens-autoencoder-relu-N1-500',
                                  'FacesMartens-autoencoder-relu-N1-100']:
                layersizes = [625, 2000, 1000, 500, 30,
                             500, 1000, 2000, 625]
                self.activations_all =\
                ['relu', 'relu', 'relu', 'linear',
                 'relu', 'relu', 'relu', 'linear']
            else:
                print('Dateset not supported!')
                sys.exit()
            
        elif self.name_model == 'simple-CNN':
            layersizes = []
            self.activations_all = []
        elif self.name_model == 'CNN':
            layersizes = []
            self.activations_all = []
        elif self.name_model == 'AllCNNC':
            layersizes = []
            self.activations_all = []
        elif self.name_model == 'ConvPoolCNNC':
            layersizes = []
            self.activations_all = []
        elif self.name_model == '1d-CNN':
            layersizes = []
            self.activations_all = ['relu', 'relu', 'relu', 'linear']
        elif self.name_model == 'vgg16':
            layersizes = []
            self.activations_all = []
        elif self.name_model == 'vgg11':
            layersizes = []
            self.activations_all = []
        elif self.name_model == 'ResNet32':
            layersizes = []
            self.activations_all = []
        elif self.name_model == 'ResNet34':
            layersizes = []
            self.activations_all = []
        else:
            print('Error: model name not supported for ' + self.name_model)
            sys.exit()
            
        self.layersizes = layersizes

        layers_params = get_layers_params(self.name_model, layersizes, self.activations_all, params)
        
        self.layers_all = []
        
        for l in range(len(layers_params)):
            
            if layers_params[l]['name'] == 'fully-connected':

                self.layers_all.append(
                nn.Linear(layers_params[l]['input_size'], layers_params[l]['output_size'], bias=True)
                )
                
            elif layers_params[l]['name'] in ['conv-no-bias-no-activation']:

                self.layers_all.append(
                    nn.Conv2d(
                        in_channels=layers_params[l]['conv_in_channels'],
                        out_channels=layers_params[l]['conv_out_channels'],
                        kernel_size=layers_params[l]['conv_kernel_size'],
                        stride=layers_params[l]['conv_stride'],
                        padding=layers_params[l]['conv_padding'],
                        bias=False
                    )
                )
                
            elif layers_params[l]['name'] in ['conv',
                                              'conv-no-activation']:

                self.layers_all.append(
                nn.Conv2d(
                    in_channels=layers_params[l]['conv_in_channels'],
                    out_channels=layers_params[l]['conv_out_channels'],
                    kernel_size=layers_params[l]['conv_kernel_size'],
                    stride=layers_params[l]['conv_stride'],
                    padding=layers_params[l]['conv_padding'])
                )
                
            elif layers_params[l]['name'] in ['ResBlock-BNNoAffine',
                                              'ResBlock-BNNoAffine-identityShortcut-NoBias',
                                              'ResBlock-BNNoAffine-PaddingShortcut-NoBias',
                                              'ResBlock-BN',
                                              'ResBlock-BN-BNshortcut',
                                              'ResBlock-BN-identityShortcut',
                                              'ResBlock-BN-identityShortcut-NoBias',
                                              'ResBlock-BN-BNshortcut-NoBias',
                                              'ResBlock-BN-PaddingShortcut-NoBias',]:
                
                self.layers_all.append([])
                
                # conv
                self.layers_all[-1].append(
                    nn.Conv2d(
                        in_channels=layers_params[l]['conv1']['conv_in_channels'],
                        out_channels=layers_params[l]['conv1']['conv_out_channels'],
                        kernel_size=layers_params[l]['conv1']['conv_kernel_size'],
                        stride=layers_params[l]['conv1']['conv_stride'],
                        padding=layers_params[l]['conv1']['conv_padding'],
                        bias=layers_params[l]['conv1']['conv_bias']
                    )
                )
                
                # BN or BNNoAffine
                if layers_params[l]['name'] in ['ResBlock-BNNoAffine',
                                                'ResBlock-BNNoAffine-identityShortcut-NoBias',
                                                'ResBlock-BNNoAffine-PaddingShortcut-NoBias',]:
                    self.layers_all[-1].append(
                        nn.BatchNorm2d(layers_params[l]['BNNoAffine1']['num_features'], affine=False).to(params['device'])
                    )
                elif layers_params[l]['name'] in ['ResBlock-BN',
                                                  'ResBlock-BN-BNshortcut',
                                                  'ResBlock-BN-identityShortcut',
                                                  'ResBlock-BN-identityShortcut-NoBias',
                                                  'ResBlock-BN-BNshortcut-NoBias',
                                                  'ResBlock-BN-PaddingShortcut-NoBias']:
                    self.layers_all[-1].append(
                        nn.BatchNorm2d(layers_params[l]['BN1']['num_features'], affine=True).to(params['device'])
                    )
                else:
                    print('layers_params[l][name]')
                    print(layers_params[l]['name'])
                    sys.exit()
                
                # conv
                self.layers_all[-1].append(
                    nn.Conv2d(
                        in_channels=layers_params[l]['conv2']['conv_in_channels'],
                        out_channels=layers_params[l]['conv2']['conv_out_channels'],
                        kernel_size=layers_params[l]['conv2']['conv_kernel_size'],
                        stride=layers_params[l]['conv2']['conv_stride'],
                        padding=layers_params[l]['conv2']['conv_padding'],
                        bias=layers_params[l]['conv2']['conv_bias']
                    )
                )
                
                # BN or BNNoAffine
                
                
                if layers_params[l]['name'] in ['ResBlock-BNNoAffine',
                                                'ResBlock-BNNoAffine-identityShortcut-NoBias',
                                                'ResBlock-BNNoAffine-PaddingShortcut-NoBias',]:
                    self.layers_all[-1].append(
                        nn.BatchNorm2d(layers_params[l]['BNNoAffine2']['num_features'], affine=False).to(params['device'])
                    )
                elif layers_params[l]['name'] in ['ResBlock-BN',
                                                  'ResBlock-BN-BNshortcut',
                                                  'ResBlock-BN-identityShortcut',
                                                  'ResBlock-BN-identityShortcut-NoBias',
                                                  'ResBlock-BN-BNshortcut-NoBias',
                                                  'ResBlock-BN-PaddingShortcut-NoBias']:
                    self.layers_all[-1].append(
                        nn.BatchNorm2d(layers_params[l]['BN2']['num_features'], affine=True).to(params['device'])
                    )
                else:
                    print('layers_params[l][name]')
                    print(layers_params[l]['name'])
                    sys.exit()
                
                if layers_params[l]['name'] in ['ResBlock-BNNoAffine-identityShortcut-NoBias',
                                                'ResBlock-BNNoAffine-PaddingShortcut-NoBias',
                                                'ResBlock-BN-identityShortcut',
                                                'ResBlock-BN-identityShortcut-NoBias',
                                                'ResBlock-BN-PaddingShortcut-NoBias',]:
                    1
                elif layers_params[l]['name'] in ['ResBlock-BNNoAffine',
                                                  'ResBlock-BN',
                                                  'ResBlock-BN-BNshortcut',
                                                  'ResBlock-BN-BNshortcut-NoBias']:
                    # 1*1 conv
                    self.layers_all[-1].append(
                        nn.Conv2d(
                            in_channels=layers_params[l]['conv3']['conv_in_channels'],
                            out_channels=layers_params[l]['conv3']['conv_out_channels'],
                            kernel_size=layers_params[l]['conv3']['conv_kernel_size'],
                            stride=layers_params[l]['conv3']['conv_stride'],
                            padding=layers_params[l]['conv3']['conv_padding'],
                            bias=layers_params[l]['conv3']['conv_bias']
                        )
                    )

                    if layers_params[l]['name'] in ['ResBlock-BN-BNshortcut',
                                                    'ResBlock-BN-BNshortcut-NoBias']:
                        self.layers_all[-1].append(
                            nn.BatchNorm2d(layers_params[l]['BN3']['num_features'], affine=True).to(params['device'])
                        )
                else:
                    print('layers_params[l][name]')
                    print(layers_params[l]['name'])
                    sys.exit()
                
                    

                    
            elif layers_params[l]['name'] == '1d-conv':

                self.layers_all.append(
                nn.Conv1d(
                    in_channels=layers_params[l]['conv_in_channels'],
                    out_channels=layers_params[l]['conv_out_channels'],
                    kernel_size=layers_params[l]['conv_kernel_size'],
                    stride=layers_params[l]['conv_stride'],
                    padding=layers_params[l]['conv_padding'])
                )

                
            elif layers_params[l]['name'] == 'flatten':
                self.layers_all.append(nn.Flatten())

            elif layers_params[l]['name'] == 'max_pool_1d':
                self.layers_all.append(
                    nn.MaxPool1d(
                        kernel_size=layers_params[l]['max_pool_kernel_size'],
                        stride=layers_params[l]['max_pool_stride'])
                )

            elif layers_params[l]['name'] == 'max_pool':
                self.layers_all.append(
                    nn.MaxPool2d(
                        kernel_size=layers_params[l]['max_pool_kernel_size'],
                        stride=layers_params[l]['max_pool_stride'])
                )

            elif layers_params[l]['name'] == 'AdaptiveAvgPool2d':
                self.layers_all.append(
                    nn.AdaptiveAvgPool2d(
                        output_size=layers_params[l]['AdaptiveAvgPool2d_output_size']
                    )
                )
            
            elif layers_params[l]['name'] == 'dropout':
                self.layers_all.append(
                    nn.Dropout(
                        p=layers_params[l]['dropout_p'], 
                        inplace=False
                    )
                    )
                
            elif layers_params[l]['name'] == 'global_average_pooling':
                self.layers_all.append([])
                
            elif layers_params[l]['name'] == 'relu':
                self.layers_all.append([])
                
            elif layers_params[l]['name'] == 'BN':
                self.layers_all.append(
                    nn.BatchNorm2d(layers_params[l]['num_features'])
                )
                
            elif layers_params[l]['name'] == 'BNNoAffine':
                
                self.layers_all.append(
                    nn.BatchNorm2d(layers_params[l]['num_features'], affine=False).to(params['device'])
                )
 
            else:
                print('Error: layer unsupported for ' + layers_params[l]['name'])
                sys.exit()
  
        self.layers_weight = []
        for l in range(len(layers_params)):
            
            if layers_params[l]['name'] in ['fully-connected',
                                            'conv',
                                            'conv-no-activation',
                                            '1d-conv']:
                layers_weight_l = {}
                layers_weight_l['W'] = self.layers_all[l].weight
                layers_weight_l['b'] = self.layers_all[l].bias
                self.layers_weight.append(layers_weight_l)
                
            elif layers_params[l]['name'] in ['conv-no-bias-no-activation']:
                layers_weight_l = {}
                layers_weight_l['W'] = self.layers_all[l].weight
                

                self.layers_weight.append(layers_weight_l)
                
            elif layers_params[l]['name'] in ['ResBlock-BNNoAffine',
                                              'ResBlock-BNNoAffine-identityShortcut-NoBias',
                                              'ResBlock-BNNoAffine-PaddingShortcut-NoBias',
                                              'ResBlock-BN',
                                              'ResBlock-BN-BNshortcut',
                                              'ResBlock-BN-identityShortcut',
                                              'ResBlock-BN-identityShortcut-NoBias',
                                              'ResBlock-BN-BNshortcut-NoBias',
                                              'ResBlock-BN-PaddingShortcut-NoBias',]:
                
                layers_weight_l = {}
                layers_weight_l['W'] = self.layers_all[l][0].weight
                
                if layers_params[l]['name'] in ['ResBlock-BNNoAffine-identityShortcut-NoBias',
                                                'ResBlock-BNNoAffine-PaddingShortcut-NoBias',
                                                'ResBlock-BN-identityShortcut-NoBias',
                                                'ResBlock-BN-BNshortcut-NoBias',
                                                'ResBlock-BN-PaddingShortcut-NoBias']:
                    pass
                elif layers_params[l]['name'] in ['ResBlock-BNNoAffine',
                                                  'ResBlock-BN',
                                                  'ResBlock-BN-identityShortcut',
                                                  'ResBlock-BN-BNshortcut']:
                    layers_weight_l['b'] = self.layers_all[l][0].bias
                else:
                    print('layers_params[l][name]')
                    print(layers_params[l]['name'])
                    sys.exit()
                
                    
                    
                self.layers_weight.append(layers_weight_l)
                
                if layers_params[l]['name'] in ['ResBlock-BN',
                                                'ResBlock-BN-BNshortcut',
                                                'ResBlock-BN-identityShortcut',
                                                'ResBlock-BN-identityShortcut-NoBias',
                                                'ResBlock-BN-BNshortcut-NoBias',
                                                'ResBlock-BN-PaddingShortcut-NoBias']:
                    layers_weight_l = {}
                    layers_weight_l['W'] = self.layers_all[l][1].weight
                    layers_weight_l['b'] = self.layers_all[l][1].bias
                    self.layers_weight.append(layers_weight_l)
                
                layers_weight_l = {}
                layers_weight_l['W'] = self.layers_all[l][2].weight
                
                if layers_params[l]['name'] in ['ResBlock-BNNoAffine-identityShortcut-NoBias',
                                                'ResBlock-BNNoAffine-PaddingShortcut-NoBias',
                                                'ResBlock-BN-identityShortcut-NoBias',
                                                'ResBlock-BN-BNshortcut-NoBias',
                                                'ResBlock-BN-PaddingShortcut-NoBias']:
                    pass
                elif layers_params[l]['name'] in ['ResBlock-BNNoAffine',
                                                  'ResBlock-BN',
                                                  'ResBlock-BN-identityShortcut',
                                                  'ResBlock-BN-BNshortcut']:
                    layers_weight_l['b'] = self.layers_all[l][2].bias
                else:
                    print('layers_params[l][name]')
                    print(layers_params[l]['name'])
                    sys.exit()
                    
                self.layers_weight.append(layers_weight_l)
                
                if layers_params[l]['name'] in ['ResBlock-BN',
                                                'ResBlock-BN-BNshortcut',
                                                'ResBlock-BN-identityShortcut',
                                                'ResBlock-BN-identityShortcut-NoBias',
                                                'ResBlock-BN-BNshortcut-NoBias',
                                                'ResBlock-BN-PaddingShortcut-NoBias']:
                    layers_weight_l = {}
                    layers_weight_l['W'] = self.layers_all[l][3].weight
                    layers_weight_l['b'] = self.layers_all[l][3].bias
                    self.layers_weight.append(layers_weight_l)
                    
                    
                if layers_params[l]['name'] in ['ResBlock-BNNoAffine-identityShortcut-NoBias',
                                                'ResBlock-BNNoAffine-PaddingShortcut-NoBias',
                                                'ResBlock-BN-identityShortcut',
                                                'ResBlock-BN-identityShortcut-NoBias',
                                                'ResBlock-BN-PaddingShortcut-NoBias']:
                    1
                else:
                
                    layers_weight_l = {}
                    layers_weight_l['W'] = self.layers_all[l][4].weight
                    if layers_params[l]['name'] == 'ResBlock-BN-BNshortcut-NoBias':
                        pass
                    elif layers_params[l]['name'] in ['ResBlock-BNNoAffine',
                                                      'ResBlock-BN',
                                                      'ResBlock-BN-BNshortcut']:
                        layers_weight_l['b'] = self.layers_all[l][4].bias
                    else:
                        print('layers_params[l][name]')
                        print(layers_params[l]['name'])
                        sys.exit()
                        
                    self.layers_weight.append(layers_weight_l)


                    if layers_params[l]['name'] in ['ResBlock-BN-BNshortcut',
                                                    'ResBlock-BN-BNshortcut-NoBias']:
                        layers_weight_l = {}
                        layers_weight_l['W'] = self.layers_all[l][5].weight
  
                        layers_weight_l['b'] = self.layers_all[l][5].bias
                        self.layers_weight.append(layers_weight_l)
                
            elif layers_params[l]['name'] == 'BN':

                
                layers_weight_l = {}
                layers_weight_l['W'] = self.layers_all[l].weight
                layers_weight_l['b'] = self.layers_all[l].bias
                self.layers_weight.append(layers_weight_l)
            elif layers_params[l]['name'] in ['flatten',
                                              'max_pool',
                                              'max_pool_1d',
                                              'AdaptiveAvgPool2d',
                                              'dropout',
                                              'global_average_pooling',
                                              'relu',
                                              'BNNoAffine']:
                1
            else:
                print('layers_params[l][name]')
                print(layers_params[l]['name'])
                print('Error: layer unsupported when define weight for ' + layers_params[l]['name'])
                sys.exit()
            




        
        

        
        
        # filter out the layers with no weights
        self.layers_params_all = layers_params
        layers_params = []
        self.layers = []
        for l in range(len(self.layers_params_all)):
            if self.layers_params_all[l]['name'] in ['fully-connected',
                                                     'conv',
                                                     'conv-no-activation',
                                                     'conv-no-bias-no-activation',
                                                     '1d-conv',
                                                     'BN']:
   
                layers_params.append(self.layers_params_all[l])
                self.layers.append(self.layers_all[l])
                
            elif self.layers_params_all[l]['name'] in ['ResBlock-BNNoAffine',
                                                       'ResBlock-BNNoAffine-identityShortcut-NoBias',
                                                       'ResBlock-BNNoAffine-PaddingShortcut-NoBias',
                                                       'ResBlock-BN',
                                                       'ResBlock-BN-BNshortcut',
                                                       'ResBlock-BN-identityShortcut',
                                                       'ResBlock-BN-identityShortcut-NoBias',
                                                       'ResBlock-BN-BNshortcut-NoBias',
                                                       'ResBlock-BN-PaddingShortcut-NoBias',]:
                
                if self.layers_params_all[l]['name'] in ['ResBlock-BNNoAffine-identityShortcut-NoBias',
                                                         'ResBlock-BNNoAffine-PaddingShortcut-NoBias',
                                                         'ResBlock-BN-identityShortcut-NoBias',
                                                         'ResBlock-BN-BNshortcut-NoBias',
                                                         'ResBlock-BN-PaddingShortcut-NoBias']:
                    layers_params.append(
                        {**{'name': 'conv-no-bias-no-activation'}, **self.layers_params_all[l]['conv1']}
                    )
                elif self.layers_params_all[l]['name'] in ['ResBlock-BNNoAffine',
                                                           'ResBlock-BN',
                                                           'ResBlock-BN-identityShortcut',
                                                           'ResBlock-BN-BNshortcut']:
                    
                    layers_params.append(
                        {**{'name': 'conv-no-activation'}, **self.layers_params_all[l]['conv1']}
                    )
                else:
                    print('self.layers_params_all[l][name]')
                    print(self.layers_params_all[l]['name'])
                    sys.exit()
            
    
                    
                
                if self.layers_params_all[l]['name'] in ['ResBlock-BN',
                                                         'ResBlock-BN-BNshortcut',
                                                         'ResBlock-BN-identityShortcut',
                                                         'ResBlock-BN-identityShortcut-NoBias',
                                                         'ResBlock-BN-BNshortcut-NoBias',
                                                         'ResBlock-BN-PaddingShortcut-NoBias']:
                    layers_params.append(
                        {**{'name': 'BN'}, **self.layers_params_all[l]['BN1']}
                    )
                

                if self.layers_params_all[l]['name'] in ['ResBlock-BNNoAffine-identityShortcut-NoBias',
                                                         'ResBlock-BNNoAffine-PaddingShortcut-NoBias',
                                                         'ResBlock-BN-identityShortcut-NoBias',
                                                         'ResBlock-BN-BNshortcut-NoBias',
                                                         'ResBlock-BN-PaddingShortcut-NoBias']:
                    layers_params.append(
                        {**{'name': 'conv-no-bias-no-activation'}, **self.layers_params_all[l]['conv2']}
                    )
                elif self.layers_params_all[l]['name'] in ['ResBlock-BNNoAffine',
                                                           'ResBlock-BN',
                                                           'ResBlock-BN-identityShortcut',
                                                           'ResBlock-BN-BNshortcut',
                                                           'ResBlock-BN-PaddingShortcut-NoBias']:
                    layers_params.append(
                        {**{'name': 'conv-no-activation'}, **self.layers_params_all[l]['conv2']}
                    )
                else:
                    print('self.layers_params_all[l][name]')
                    print(self.layers_params_all[l]['name'])
                    sys.exit()
                    
                
                if self.layers_params_all[l]['name'] in ['ResBlock-BN',
                                                         'ResBlock-BN-BNshortcut',
                                                         'ResBlock-BN-identityShortcut',
                                                         'ResBlock-BN-identityShortcut-NoBias',
                                                         'ResBlock-BN-BNshortcut-NoBias',
                                                         'ResBlock-BN-PaddingShortcut-NoBias']:
                    layers_params.append(
                        {**{'name': 'BN'}, **self.layers_params_all[l]['BN2']}
                    )
                
                if self.layers_params_all[l]['name'] in ['ResBlock-BNNoAffine-identityShortcut-NoBias',
                                                         'ResBlock-BNNoAffine-PaddingShortcut-NoBias',
                                                         'ResBlock-BN-identityShortcut',
                                                         'ResBlock-BN-identityShortcut-NoBias',
                                                         'ResBlock-BN-PaddingShortcut-NoBias']:
                    1
                else:
                    if self.layers_params_all[l]['name'] == 'ResBlock-BN-BNshortcut-NoBias':
                        layers_params.append(
                            {**{'name': 'conv-no-bias-no-activation'}, **self.layers_params_all[l]['conv3']}
                        )
                    elif self.layers_params_all[l]['name'] in ['ResBlock-BNNoAffine',
                                                               'ResBlock-BN',
                                                               'ResBlock-BN-BNshortcut']:
                        layers_params.append(
                            {**{'name': 'conv-no-activation'}, **self.layers_params_all[l]['conv3']}
                        )
                    else:
                        print('self.layers_params_all[l][name]')
                        print(self.layers_params_all[l]['name'])
                        sys.exit()
                        

                    if self.layers_params_all[l]['name'] in ['ResBlock-BN-BNshortcut',
                                                             'ResBlock-BN-BNshortcut-NoBias']:
                        layers_params.append(
                            {**{'name': 'BN'}, **self.layers_params_all[l]['BN3']}
                        )

                self.layers.append(self.layers_all[l][0])
                
                if self.layers_params_all[l]['name'] in ['ResBlock-BN',
                                                         'ResBlock-BN-BNshortcut',
                                                         'ResBlock-BN-identityShortcut',
                                                         'ResBlock-BN-identityShortcut-NoBias',
                                                         'ResBlock-BN-BNshortcut-NoBias',
                                                         'ResBlock-BN-PaddingShortcut-NoBias']:
                    self.layers.append(self.layers_all[l][1])
                
                self.layers.append(self.layers_all[l][2])
                
                if self.layers_params_all[l]['name'] in ['ResBlock-BN',
                                                         'ResBlock-BN-BNshortcut',
                                                         'ResBlock-BN-identityShortcut',
                                                         'ResBlock-BN-identityShortcut-NoBias',
                                                         'ResBlock-BN-BNshortcut-NoBias',
                                                         'ResBlock-BN-PaddingShortcut-NoBias']:
                    self.layers.append(self.layers_all[l][3])
                
                if self.layers_params_all[l]['name'] in ['ResBlock-BNNoAffine-PaddingShortcut-NoBias',
                                                         'ResBlock-BNNoAffine-identityShortcut-NoBias',
                                                         'ResBlock-BN-identityShortcut',
                                                         'ResBlock-BN-identityShortcut-NoBias',
                                                         'ResBlock-BN-PaddingShortcut-NoBias']:
                    1
                else:
                    self.layers.append(self.layers_all[l][4])

                    if self.layers_params_all[l]['name'] in ['ResBlock-BN-BNshortcut',
                                                             'ResBlock-BN-BNshortcut-NoBias']:
                        self.layers.append(self.layers_all[l][5])
                
            elif self.layers_params_all[l]['name'] in ['flatten',
                                                       'max_pool',
                                                       'max_pool_1d',
                                                       'AdaptiveAvgPool2d',
                                                       'dropout',
                                                       'global_average_pooling',
                                                       'relu',
                                                       'BNNoAffine']:
                1
            else:
                print('error: unkown layers_params_all[l][name]: ' + self.layers_params_all[l]['name'])
                sys.exit()
                

                
        
        import torch.nn.init as init
        
        for l in range(len(layers_params)):
            
            if layers_params[l]['name'] == 'fully-connected':
                
                if params['initialization_pkg'] == 'numpy':
                
                    np_W_l =\
                    (np.random.uniform(size=self.layers_weight[l]['W'].size()) * 2 - 1) *\
                    np.sqrt(2 / (layers_params[l]['input_size'] + layers_params[l]['output_size']))

                    np_W_l = torch.from_numpy(np_W_l)

                    self.layers_weight[l]['W'].data = np_W_l.type(torch.FloatTensor)
                
                elif params['initialization_pkg'] == 'torch':
                    
                    self.layers_weight[l]['W'].data =\
                    (
                        torch.distributions.uniform.Uniform(low=0, high=1).sample(
                            sample_shape=self.layers_weight[l]['W'].size()
                        )
                        * 2 - 1
                    ) *\
                    np.sqrt(
                        2 / (layers_params[l]['input_size'] + layers_params[l]['output_size'])
                    )
                    
                elif params['initialization_pkg'] in ['default',
                                                      'normal']:
                    # U(-sqrt(k), sqrt(k)), where k = 1 / in_featurest
                    
                    pass
                    
                elif params['initialization_pkg'] == 'kaiming_normal':
         
                    
                    init.kaiming_normal_(self.layers_weight[l]['W'])
  
                else:
                    print('error: unknown params[initialization_pkg] for ' + params['initialization_pkg'])
                    sys.exit()
                

            elif layers_params[l]['name'] in ['conv',
                                              'conv-no-activation',
                                              'conv-no-bias-no-activation']:
                
                if params['initialization_pkg'] == 'normal':
            
                    # https://github.com/chengyangfu/pytorch-vgg-cifar10
                    
                    if layers_params[l]['name'] == 'conv-no-bias-no-activation':
                        pass
                    elif layers_params[l]['name'] in ['conv-no-activation',
                                                      'conv']:
                        self.layers_weight[l]['b'].data.zero_()
                    else:
                        print('layers_params[l][name]')
                        print(layers_params[l]['name'])
                        sys.exit()

                    self.layers_weight[l]['W'].data.normal_(
                        0, 
                        math.sqrt(2. / (layers_params[l]['conv_kernel_size']**2 * layers_params[l]['conv_out_channels']))
                    )
                elif params['initialization_pkg'] == 'default':
                    # use default initialization
                    1
                elif params['initialization_pkg'] == 'kaiming_normal':
                    
                    # the default arguments are equivalent to relu
                    
                    init.kaiming_normal_(self.layers_weight[l]['W'])
                    
                else:
                    print('error: need to check for ' + params['initialization_pkg'])
                    sys.exit()
                    
                    
            elif layers_params[l]['name'] in ['ResBlock-BNNoAffine']:
                
                print('should not reach here')
                
                sys.exit()
                
                print('need to check how to initialize')
                    
                    
                
            elif layers_params[l]['name'] in ['1d-conv']:
                
                sys.exit()
                
                # use default initialization
                1

            elif layers_params[l]['name'] in ['BN']:
                1

            else:
                print('layers_params[l][name]')
                print(layers_params[l]['name'])
                print('Error: layer unsupported when initialization.')
                sys.exit()
               
        
        
        self.layers_params = layers_params
        
        self.numlayers = len(layers_params)
        self.numlayers_all = len(self.layers_params_all)
        
        self.layers = nn.ModuleList(self.layers)

        
            

    
    def forward(self, x):
        a = []
        h = []

        input_ = x


        for l in range(self.numlayers_all):
            if self.layers_params_all[l]['name'] in ['fully-connected',
                                                     'conv',
                                                     'conv-no-activation',
                                                     'conv-no-bias-no-activation',
                                                     '1d-conv',
                                                     'BN']:
                if self.layers_params_all[l]['name'] in ['conv-no-activation',
                                                         'conv',
                                                         'conv-no-bias-no-activation',
                                                         'fully-connected']:

                    h.append(input_)
                
                elif self.layers_params_all[l]['name'] == 'BN':
                    h.append([])
                    
                else:
                    print('error: need to check for ' + self.layers_params_all[l]['name'])
                    sys.exit()
                
                
                input_, a_l =\
                get_layer_forward(
                    input_, self.layers_all[l], self.layers_params_all[l]['activation'], self.layers_params_all[l])

                a.append(a_l)
                
                
            elif self.layers_params_all[l]['name'] in ['ResBlock-BNNoAffine',
                                                       'ResBlock-BNNoAffine-identityShortcut-NoBias',
                                                       'ResBlock-BNNoAffine-PaddingShortcut-NoBias',
                                                       'ResBlock-BN',
                                                       'ResBlock-BN-BNshortcut',
                                                       'ResBlock-BN-identityShortcut',
                                                       'ResBlock-BN-identityShortcut-NoBias',
                                                       'ResBlock-BN-BNshortcut-NoBias',
                                                       'ResBlock-BN-PaddingShortcut-NoBias',]:
                
                h.append([])
                h.append([])
                
                a.append([])
                a.append([])
                
                if self.layers_params_all[l]['name'] == 'ResBlock-BN':
                    h.append([])
                    h.append([])

                    a.append([])
                    a.append([])
                    
                    h.append([])

                    a.append([])
                    
                    index_mapping = [-1, -5, -4, -3, -2]
                elif self.layers_params_all[l]['name'] in ['ResBlock-BN-BNshortcut',
                                                           'ResBlock-BN-BNshortcut-NoBias']:
                    h.append([])
                    h.append([])
                    h.append([])

                    a.append([])
                    a.append([])
                    a.append([])
                    
                    h.append([])

                    a.append([])
                    
                    index_mapping = [-2, -6, -5, -4, -3]
                elif self.layers_params_all[l]['name'] in ['ResBlock-BN-identityShortcut',
                                                           'ResBlock-BN-identityShortcut-NoBias',
                                                           'ResBlock-BN-PaddingShortcut-NoBias']:
                    h.append([])

                    a.append([])
                    
                    h.append([])

                    a.append([])
                    
                    index_mapping = [None, -4, -3, -2, -1]
                elif self.layers_params_all[l]['name'] in ['ResBlock-BNNoAffine-identityShortcut-NoBias',
                                                           'ResBlock-BNNoAffine-PaddingShortcut-NoBias',]:
                
                    index_mapping = [None, -2, None, -1, None]
                    
                elif self.layers_params_all[l]['name'] == 'ResBlock-BNNoAffine':
                    index_mapping = [-1, -3, None, -2, None]
                    
                
                if self.layers_params_all[l]['name'] in ['ResBlock-BNNoAffine-identityShortcut-NoBias',
                                                         'ResBlock-BN-identityShortcut',
                                                         'ResBlock-BN-identityShortcut-NoBias']:
                    input_shortcut = input_
                elif self.layers_params_all[l]['name'] in ['ResBlock-BNNoAffine-PaddingShortcut-NoBias',
                                                           'ResBlock-BN-PaddingShortcut-NoBias',]:

                    input_shortcut = F.pad(input_[:, :, ::2, ::2], (0,0,0,0,input_.size(1)//2,input_.size(1)//2))
                    
                else:

                    h[index_mapping[0]] = input_
                    input_shortcut, a_l = get_layer_forward(
                        input_, self.layers_all[l][4], None, {'name':'conv-no-activation'})
                    a[index_mapping[0]] = a_l

                    if self.layers_params_all[l]['name'] in ['ResBlock-BN-BNshortcut',
                                                             'ResBlock-BN-BNshortcut-NoBias']:
                        input_shortcut, a_l =\
                        get_layer_forward(
                            input_shortcut, self.layers_all[l][5], None, {'name':'BN'})
                        a[-1] = a_l
                
                
                # conv
                h[index_mapping[1]] = input_
                input_, a_l =\
                get_layer_forward(
                    input_, self.layers_all[l][0], None, {'name':'conv-no-activation'})
                # no-bias is not needed here
                a[index_mapping[1]] = a_l
                
                # BN or BNNoAffine
                if index_mapping[2] == None:
                    input_ = self.layers_all[l][1](input_)
                else:
                    input_, a_l =\
                    get_layer_forward(
                        input_, self.layers_all[l][1], None, {'name':'BN'})
                    a[index_mapping[2]] = a_l
                
                # relu
                input_ = F.relu(input_)
                
                # conv
                h[index_mapping[3]] = input_
                input_, a_l =\
                get_layer_forward(
                    input_, self.layers_all[l][2], None, {'name':'conv-no-activation'})
                a[index_mapping[3]] = a_l
                
                # BN or BNNoAffine
                if index_mapping[4] == None:
                    input_ = self.layers_all[l][3](input_)
                else:
                    input_, a_l =\
                    get_layer_forward(
                        input_, self.layers_all[l][3], None, {'name':'BN'})
                    a[index_mapping[4]] = a_l
                
                input_ = input_ + input_shortcut
                
                # relu
                input_ = F.relu(input_)
                
            elif self.layers_params_all[l]['name'] in ['flatten',
                                                       'max_pool',
                                                       'max_pool_1d',
                                                       'AdaptiveAvgPool2d',
                                                       'dropout',
                                                       'BNNoAffine']:

                

                input_ = self.layers_all[l](input_)
                
            elif self.layers_params_all[l]['name'] == 'global_average_pooling':
                input_ = torch.mean(input_, dim=(2,3))
    
            elif self.layers_params_all[l]['name'] == 'relu':
                input_ = F.relu(input_)
            else:
                print('error: unknown self.layers_params_all[l][name]: ' + self.layers_params_all[l]['name'])
                sys.exit()
        return input_, a, h
