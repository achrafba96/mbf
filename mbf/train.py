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

from .data_utils import *
from .utils_functions import *
from .models import *

import warnings
warnings.filterwarnings('error')
# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def get_params(params, args):
    params['true_algorithm'] = params['algorithm']
    algorithm = params['algorithm']
    
    if algorithm == 'MBNGD-all-to-one-LRdecay':
        params['if_lr_decay'] = True
        params['algorithm'] = 'MBNGD-all-to-one'

    elif algorithm == 'MBNGD-all-to-one-Avg-LRdecay':
        params['if_lr_decay'] = True
        params['algorithm'] = 'MBNGD-all-to-one-Avg'

    elif algorithm == 'L-MBNGD-all-to-one-LRdecay':
        params['if_lr_decay'] = True
        params['algorithm'] = 'L-MBNGD-all-to-one'

    elif algorithm == 'SGD-LRdecay-momentum':
        params['if_lr_decay'] = True
        params['algorithm'] = 'SGD-momentum'

    elif algorithm == 'Adam-noWarmStart-momentum-grad-LRdecay':
        params['if_lr_decay'] = True
        params['algorithm'] = 'Adam-noWarmStart-momentum-grad'

    elif algorithm == 'kfac-correctFisher-warmStart-no-max-no-LM-momentum-grad-LRdecay':
        params['if_lr_decay'] = True
        params['algorithm'] = 'kfac-correctFisher-warmStart-no-max-no-LM-momentum-grad'
        
    elif algorithm == 'shampoo-allVariables-filterFlattening-warmStart-momentum-grad-LRdecay':
        params['if_lr_decay'] = True
        params['algorithm'] = 'shampoo-allVariables-filterFlattening-warmStart-momentum-grad'
        
    elif algorithm == 'shampoo-allVariables-filterFlattening-warmStart-lessInverse-momentum-grad-LRdecay':
        params['if_lr_decay'] = True
        params['algorithm'] = 'shampoo-allVariables-filterFlattening-warmStart-lessInverse-momentum-grad'
        
    elif algorithm in ['SGD-momentum',
                       'Adam-noWarmStart-momentum-grad',
                       'Fisher-BD',
                       'Fisher-BD-momentum-grad',
                       'kfac-correctFisher-warmStart-no-max-no-LM-momentum-grad',
                       'shampoo-allVariables-filterFlattening-warmStart-momentum-grad', 
                       'shampoo-allVariables-filterFlattening-warmStart-lessInverse-momentum-grad']:
        params['if_lr_decay'] = False
    else:
        print('algorithm')
        print(algorithm)
        sys.exit()
    
    algorithm = params['algorithm']
    
    if algorithm == 'MBNGD-all-to-one':
        params['if_momentum_gradient'] = True
    elif algorithm == 'MBNGD-all-to-one-Avg':
        params['if_momentum_gradient'] = True
    elif algorithm == 'L-MBNGD-all-to-one':
        params['if_momentum_gradient'] = True
    elif algorithm == 'Fisher-BD-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Fisher-BD'
    elif algorithm == 'kfac-correctFisher-warmStart-no-max-no-LM-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'kfac-correctFisher-warmStart-no-max-no-LM'
        
    elif algorithm == 'SGD-momentum':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'SGD'
    
    elif algorithm == 'SGD-LRdecay-momentum':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'SGD-LRdecay'
        
    elif algorithm == 'Adam-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Adam'
     
    elif algorithm == 'Adam-noWarmStart-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'Adam-noWarmStart'
        
    elif algorithm == 'shampoo-allVariables-filterFlattening-warmStart-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'shampoo-allVariables-filterFlattening-warmStart'

    elif algorithm == 'shampoo-allVariables-filterFlattening-warmStart-lessInverse-momentum-grad':
        params['if_momentum_gradient'] = True
        params['algorithm'] = 'shampoo-allVariables-filterFlattening-warmStart-lessInverse'
    else:   
        params['if_momentum_gradient'] = True

    
    # get rid of (un)regularized grad
    algorithm = params['algorithm']
    params['if_regularized_grad'] = True     
    params['if_double_grad'] = False
    
    if algorithm in ['MBNGD-all-to-one', 'MBNGD-all-to-one-Avg', 'L-MBNGD-all-to-one']:
        params['if_second_order_algorithm'] = False

    elif algorithm in ['SGD-momentum',
                     'SGD',
                     'shampoo-allVariables-filterFlattening-warmStart',
                     'shampoo-allVariables-filterFlattening-warmStart-lessInverse',
                     'Adam',
                     'Adam-test',
                     'Adam-noWarmStart',
                     'RMSprop-no-sqrt',
                     'BFGS',
                     'BFGS-homo']:
        params['if_second_order_algorithm'] = False
    elif algorithm in ['Fisher-BD',
                       'kfac-correctFisher-warmStart-no-max-no-LM',
                       'GI-Fisher']:
        params['if_second_order_algorithm'] = True
    else:
        print('Error: unknown if_second_order_algorithm for ' + algorithm)
        sys.exit()
        
    if algorithm in ['GI-Fisher']:
        params['if_LM'] = True
    else:
        params['if_LM'] = False
   
        
    if algorithm in ['MBNGD-all-to-one', 'MBNGD-all-to-one-Avg', 'L-MBNGD-all-to-one']:
        params['if_model_grad_N2'] = True
    elif algorithm in ['SGD-momentum',
                       'SGD',
                       'shampoo-allVariables-filterFlattening-warmStart',
                       'shampoo-allVariables-filterFlattening-warmStart-lessInverse',
                       'Adam',
                       'Adam-test',
                       'Adam-noWarmStart',
                       'Fisher-BD',
                       'kfac-correctFisher-warmStart-no-max-no-LM']:
        params['if_model_grad_N2'] = False
    else:
        print('Error: check if need model_grad_N2')
        sys.exit()

    params['if_different_minibatch'] = False
    params['if_sign'] = False
    params['if_momentum_p'] = False
    params['if_VA_p'] = False
    params['if_signVAsqrt'] = False
    params['if_signVA'] = False
    params['if_yura'] = False
    params['if_signVAerf'] = False
    params['if_Adam'] = False
    

    params['keys_params_saved'] = []

    params['keys_params_saved'].append('if_test_mode')
    params['keys_params_saved'].append('tau')
    params['keys_params_saved'].append('seed_number')
    params['keys_params_saved'].append('num_threads')
    
    params['keys_params_saved'].append('initialization_pkg')
    params['keys_params_saved'].append('N1')
    params['keys_params_saved'].append('N2')
    
    params['keys_params_saved'].append('if_max_epoch')
    params['keys_params_saved'].append('max_epoch/time')
    
    params['keys_params_saved'].append('momentum_gradient_dampening')
    
    params['keys_params_saved'].append('if_grafting')
    params['keys_params_saved'].append('weight_decay')
    
    if params['if_lr_decay']:
        params['keys_params_saved'].append('num_epoch_to_decay')
        params['keys_params_saved'].append('lr_decay_rate')

    if params['algorithm'] in ['RMSprop',
                               'RMSprop-warmStart',
                               'RMSprop-test',
                               'Adam',
                               'Adam-test',
                               'Adam-noWarmStart']:
        params['keys_params_saved'].append('RMSprop_epsilon')
        params['keys_params_saved'].append('RMSprop_beta_2')
        
    if params['algorithm'] in ['MBNGD-all-to-one-LRdecay', 
                               'MBNGD-all-to-one', 
                               'MBNGD-all-to-one-Avg', 
                               'L-MBNGD-all-to-one']:
        params['keys_params_saved'].append('mbngd_damping_lambda')
        params['keys_params_saved'].append('kfac_cov_update_freq')
        params['keys_params_saved'].append('kfac_inverse_update_freq') 
        
    if params['algorithm'] in ['L-MBNGD-all-to-one']:
        params['keys_params_saved'].append('window')
        params['keys_params_saved'].append('kfac_inverse_update_freq')
            
    if params['algorithm'] in ['kfac-correctFisher-warmStart-no-max-no-LM']:
        params['keys_params_saved'].append('kfac_if_svd')
        params['keys_params_saved'].append('kfac_if_update_BN')
        params['keys_params_saved'].append('kfac_if_BN_grad_direction')
        params['keys_params_saved'].append('kfac_inverse_update_freq')
        params['keys_params_saved'].append('kfac_cov_update_freq')
        params['keys_params_saved'].append('kfac_damping_lambda')
        
    if params['algorithm'] in ['shampoo-allVariables-filterFlattening-warmStart',
                               'shampoo-allVariables-filterFlattening-warmStart-lessInverse']:
        params['keys_params_saved'].append('shampoo_inverse_freq')
        params['keys_params_saved'].append('shampoo_update_freq')
        params['keys_params_saved'].append('shampoo_decay')
        params['keys_params_saved'].append('shampoo_weight')
        params['keys_params_saved'].append('if_Hessian_action')
        params['keys_params_saved'].append('shampoo_if_coupled_newton')
        params['keys_params_saved'].append('shampoo_epsilon')

    if params['algorithm'] == 'Fisher-BD':
        params['keys_params_saved'].append('Fisher_BD_damping')
   
    return params

def get_next_lr(list_lr_tried, best_lr):
    list_lr_complete =\
    [1e-10, 3e-10, 1e-9, 3e-9, 1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000]
    if best_lr == min(list_lr_tried):
        print('list_lr_complete.index(best_lr)')
        print(list_lr_complete.index(best_lr))
        if list_lr_complete.index(best_lr) == 0:
            print('error: need to expand list_lr_complete')
            sys.exit()
        else:
            return list_lr_complete[list_lr_complete.index(best_lr) - 1]
    elif best_lr == max(list_lr_tried):
        if list_lr_complete.index(best_lr) == len(list_lr_complete) - 1:
            print('error: need to expand list_lr_complete')
            sys.exit()
        else:
            return list_lr_complete[list_lr_complete.index(best_lr) + 1]
    elif best_lr > min(list_lr_tried) and best_lr < max(list_lr_tried):
        return -1
    else:
        print('there is an error')
        sys.exit()
    return learning_rate

def add_some_if_record_to_args(args):
    if not 'if_record_sgd_norm' in args:
        args['if_record_sgd_norm'] = False
    if not 'if_record_sgn_norm' in args:
        args['if_record_sgn_norm'] = False
    if not 'if_record_p_norm' in args:
        args['if_record_p_norm'] = False
    if not 'if_record_kfac_p_norm' in args:
        args['if_record_kfac_p_norm'] = False
    if not 'if_record_kfac_p_cosine' in args:
        args['if_record_kfac_p_cosine'] = False
    if not 'if_record_res_grad_norm' in args:
        args['if_record_res_grad_norm'] = False
    if not 'if_record_res_grad_random_norm' in args:
        args['if_record_res_grad_random_norm'] = False
    if not 'if_record_res_grad_grad_norm' in args:
        args['if_record_res_grad_grad_norm'] = False
    if not 'if_record_res_grad_norm_per_iter' in args:
        args['if_record_res_grad_norm_per_iter'] = False
    return args

def add_matrix_name_to_args(args):
    if args['algorithm'] in ['MBNGD-all-to-one-LRdecay', 
                             'MBNGD-all-to-one', 
                             'MBNGD-all-to-one-Avg-LRdecay',
                             'MBNGD-all-to-one-Avg', 
                             'L-MBNGD-all-to-one', 
                             'L-MBNGD-all-to-one-LRdecay']:
        args['matrix_name'] = 'Fisher-correct'
    elif args['algorithm'] in ['Fisher-BD',
                               'Fisher-BD-momentum-grad',
                               'kfac-correctFisher-warmStart-no-max-no-LM-momentum-grad-LRdecay', 
                               'kfac-correctFisher-warmStart-no-max-no-LM-momentum-grad']:
        args['matrix_name'] = 'Fisher-correct'
    elif args['algorithm'] in ['SGD-momentum-yura',
                               'SGD-momentum',
                               'SGD-LRdecay-momentum',
                               'SGD',
                               'shampoo-allVariables-filterFlattening-warmStart-momentum-grad-LRdecay', 
                               'shampoo-allVariables-filterFlattening-warmStart-momentum-grad',
                               'shampoo-allVariables-filterFlattening-warmStart-lessInverse-momentum-grad',
                               'shampoo-allVariables-filterFlattening-warmStart-lessInverse-momentum-grad-LRdecay',
                               'Adam-momentum-grad',
                               'Adam-noWarmStart-momentum-grad',
                               'Adam-noWarmStart-momentum-grad-LRdecay']:
        args['matrix_name'] = 'None'
    else:
        print('Error: undefined matrix name for ' + args['algorithm'])
        sys.exit()
    
    return args

def get_warm_start(data_, params):
    N1 = params['N1']
    assert N1 < params['num_train_data']
    # i.e. stochastic setting
    device = params['device']
    numlayers = params['numlayers']
    layers_params = params['layers_params']
    model = data_['model']
    i = 0 # position of training data
    j = 0 # position of mini-batch
    
    print('Begin warm start...')
    while i + N1 <= params['num_train_data']:
        X_mb, t_mb = data_['dataset'].train.next_batch(N1)

        if not params['if_dataset_onTheFly']:
            X_mb = torch.from_numpy(X_mb)
        X_mb = X_mb.to(device)
        z, a, h = model.forward(X_mb)
        if params['matrix_name'] in ['Fisher',
                                     'Fisher-correct']:
            params['N2_index'] = list(range(N1))
            t_mb_pred = sample_from_pred_dist(z, params)
            del params['N2_index']
            t_mb_used = t_mb_pred
        elif params['matrix_name'] == 'None':
            if not params['if_dataset_onTheFly']:
                t_mb = torch.from_numpy(t_mb)
            t_mb = t_mb.to(device)
            t_mb_used = t_mb
        else:
            print('params[matrix_name]')
            print(params['matrix_name'])
            sys.exit()
        loss = get_loss_from_z(model, z, t_mb_used, reduction='mean') # not regularized
        model.zero_grad()
        loss.backward()
        if params['if_model_grad_N2'] or\
        params['algorithm'] in ['shampoo-allVariables-warmStart',
                                'shampoo-allVariables-warmStart-lessInverse',
                                'shampoo-allVariables-filterFlattening-warmStart',
                                'shampoo-allVariables-filterFlattening-warmStart-lessInverse',
                                'matrix-normal-EF-same-trace-allVariables-filterFlattening-warmStart',]:
            model_grad_N2 = get_model_grad(model, params)
        if params['algorithm'] in ['MBNGD-all-to-one-LRdecay', 
                             'MBNGD-all-to-one', 
                             'MBNGD-all-to-one-Avg-LRdecay',
                             'MBNGD-all-to-one-Avg', 
                             'L-MBNGD-all-to-one', 
                            'L-MBNGD-all-to-one-LRdecay']:
            model_grad_N2 = get_model_grad(model, params)
        i += N1
        j += 1
        for l in range(numlayers):
            if params['algorithm'] in ['shampoo-allVariables-warmStart',
                                       'shampoo-allVariables-warmStart-lessInverse',
                                       'shampoo-allVariables-filterFlattening-warmStart',
                                       'shampoo-allVariables-filterFlattening-warmStart-lessInverse',]:
                
                for name_variable in data_['model'].layers_weight[l].keys():
                    shampoo_kron_matrices_warm_start_per_variable(j, model_grad_N2, l, name_variable, data_, params)
                    if params['if_Hessian_action'] and not i + N1 <= params['num_train_data']:
                        assert params['algorithm'] == 'matrix-normal-correctFisher-allVariables-KFACReshaping-warmStart-lessInverse'
                        epsilon = params['shampoo_epsilon']
                        H = data_['shampoo_H']
                        H_l_LM_minus_2k = []
                        for ii in range(len(H[l][name_variable])):
                            H_l_ii_LM = H[l][name_variable][ii] + epsilon * torch.eye(H[l][name_variable][ii].shape[0], device=device)
                            H_l_LM_minus_2k.append(H_l_ii_LM.inverse())
                        data_['shampoo_H_LM_minus_2k'][l][name_variable] = H_l_LM_minus_2k
                
            elif params['algorithm'] in ['kfac-no-max-no-LM',
                                         'kfac-warmStart-no-max-no-LM',
                                         'kfac-warmStart-lessInverse-no-max-no-LM',
                                         'kfac-warmStart-lessInverse-NoMaxNoSqrt-no-LM',
                                         'kfac-correctFisher-warmStart-no-max-no-LM',
                                         'kfac-correctFisher-warmStart-NoMaxNoSqrt-no-LM',
                                         'kfac-correctFisher-warmStart-lessInverse-no-max-no-LM',
                                         'kfac-correctFisher-warmStart-lessInverse-NoMaxNoSqrt-no-LM',]:
                
                if layers_params[l]['name'] in ['conv',
                                                'conv-no-activation',
                                                'conv-no-bias-no-activation',
                                                'fully-connected']:
                    A_j = get_A_A_T(h, l, data_, params)
                    data_['A'][l] *= (j-1)/j
                    data_['A'][l] += 1/j * A_j
                    G_j = get_g_g_T(a, l, params)
                    data_['G'][l] *= (j-1)/j
                    data_['G'][l] += 1/j * G_j
                    
                elif layers_params[l]['name'] == 'BN':
                    if params['kfac_if_update_BN'] and not params['kfac_if_BN_grad_direction']:
                        G_j = get_g_g_T_BN(model, l, N1)
                        data_['G'][l] *= (j-1)/j
                        data_['G'][l] += 1/j * G_j
                else:
                    print('error: need to check for ' + layers_params[l]['name'])
                    sys.exit()  
            elif params['algorithm'] in ['MBNGD-all-to-one-LRdecay', 
                             'MBNGD-all-to-one', 
                             'MBNGD-all-to-one-Avg-LRdecay',
                             'MBNGD-all-to-one-Avg', 'L-MBNGD-all-to-one', 'L-MBNGD-all-to-one-LRdecay']:
                if (layers_params[l]['name'] in ['fully-connected'])*(params['algorithm'] in ['MBNGD-all-to-one-Avg',
                             'MBNGD-all-to-one-Avg-LRdecay']):
                    Os, Is = model_grad_N2[l]['W'].shape
                    m1, m2 = data_['F_m'][l]['m1m2']
                    homo_model_grad_N2_l = torch.cat((model_grad_N2[l]['W'], model_grad_N2[l]['b'].unsqueeze(1)), dim=1)
                    grad_view = homo_model_grad_N2_l.view(m1*m2, ((Is+1)//m1)*(Os//m2))
                    matrix_dim = ((Is+1)//m1)*(Os//m2)
                    
                    if matrix_dim < -1:
                        data_['F_m'][l]['W'] *= (j-1)/j
                        data_['F_m'][l]['W'] += 1/j * torch.einsum('mi, mj-> mij', grad_view, grad_view)
                    else:
                        data_['F_m'][l]['W'] *= (j-1)/j
                        data_['F_m'][l]['W'] += 1/j * torch.einsum('mi, mj-> ij', grad_view, grad_view)/(m1*m2)
                    
                elif (layers_params[l]['name'] in ['fully-connected'])*(params['algorithm'] in ['L-MBNGD-all-to-one', 'L-MBNGD-all-to-one-LRdecay']):
                    Os, Is = model_grad_N2[l]['W'].shape
                    homo_model_grad_N2_l = torch.cat((model_grad_N2[l]['W'], model_grad_N2[l]['b'].unsqueeze(1)), dim=1)
                    grad_view = homo_model_grad_N2_l.view(Is+1, Os)
                    
                    if Os < -1:
                        data_['F_m'][l]['W'] *= (j-1)/j
                        data_['F_m'][l]['W'] += 1/j * torch.einsum('mi, mj-> mij', grad_view, grad_view)
                    else:
                        data_['F_m'][l]['Gt'][:, :, 1:].data = ((j-1)/j)*data_['F_m'][l]['Gt'][:, :, :-1].data
                        data_['F_m'][l]['Gt'][:, :, 0].data = (1/j)*grad_view.data

                elif layers_params[l]['name'] in ['conv','conv-no-activation','conv-no-bias-no-activation']:
                    flat_g = model_grad_N2[l]['W'].flatten(start_dim = 2)
                    ggT = torch.einsum('abi, abj-> abij', flat_g, flat_g)
                    data_['F_m'][l]['W'] *= (j-1)/j
                    data_['F_m'][l]['W'] += 1/j * ggT
                else:
                    1
            elif params['algorithm'] in ['Fisher-BD',]:
                print('i')
                print(i)
                G_j = get_block_Fisher(h, a, l, params)
                if j == 1:
                    data_['block_Fisher'][l] = G_j
                else:
                    data_['block_Fisher'][l] *= (j-1)/j
                    data_['block_Fisher'][l] += 1/j * G_j
            else:
                print('error: need to check for ' + params['algorithm'])
                sys.exit()
                                
def get_best_params(args, if_plot):
    
    result_path = args['home_path'] + 'result/'
    
    if 'algorithm_dict' in args:
        algorithm_dict = args['algorithm_dict']
    else:
        algorithm_dict = {}
        algorithm_dict['name'] = args['algorithm']
        algorithm_dict['params'] = {}
    
    
    # plot lr vs test accuracy
    if 'list_lr' in args:
        list_lr_try = args['list_lr']
    else:
        
        
        print('algorithm_dict[name]')
        print(algorithm_dict['name'])
        
        fake_params = {}
        fake_params['algorithm'] = algorithm_dict['name']
        fake_params['if_gpu'] = args['if_gpu']

        test_list_lr_try = os.listdir(result_path + args['dataset'] + '/' + get_name_algorithm(fake_params)[0] + '/')
        

        test_list_lr_try = [lr_ for lr_ in test_list_lr_try if lr_.startswith('alpha_')]

        test_list_lr_try = [lr_.replace('alpha_','') for lr_ in test_list_lr_try]

        list_lr_try = test_list_lr_try
        
        
        list_lr_try = sorted(list_lr_try, key=float)
        
        
        print('list_lr_try')
        print(list_lr_try)


    

    


    
    
    

    os.chdir(result_path)

    
    list_acc = []
    list_name_result_pkl = []
    list_lr = []
    for lr in list_lr_try:

        fake_params = {}
        fake_params['alpha'] = lr
        fake_params['N1'] = args['N1']
        fake_params['N2'] = args['N2']

        # fake_params['algorithm'] = args['algorithm']
        fake_params['algorithm'] = algorithm_dict['name']
        fake_params['if_gpu'] = args['if_gpu']

        name_algorithm_with_params = get_name_algorithm_with_params(fake_params)

        path_to_result = args['dataset'] + '/' + name_algorithm_with_params + '/'
        


        if os.path.isdir(path_to_result):
            onlyfiles = [f for f in os.listdir(
            path_to_result) if os.path.isfile(os.path.join(path_to_result, f))]
        else:
            continue
        

        for f_ in onlyfiles:
            with open(path_to_result + f_, 'rb') as handle:
                
                
                record_result = pickle.load(handle)


            if_candidate_result = True
            if 'params' in algorithm_dict:
                if 'params' in record_result:
                    for key in algorithm_dict['params']:
                        if key in record_result['params']:
                            if algorithm_dict['params'][key] != record_result['params'][key]:
                                if_candidate_result = False
                                break
                        else:
                            if_candidate_result = False
                            break
                else:
                    if algorithm_dict['params'] == {}:
                        # if ('params' in algorithm_dict) and ('params' not in record_result)
                        # and (algorithm_dict['params'] == {})
                        1
                    else:
                        if_candidate_result = False
            else:
                if 'params' in record_result:
                    if_candidate_result = False
                    


            if if_candidate_result == False:
                continue

            
            if args['tuning_criterion'] in ['test_acc',
                                            'train_acc',
                                            'train_minibatch_acc']:
                
                if args['tuning_criterion'] == 'train_acc':
                    
                    if 'train_acces' in record_result:
                        record_acc = record_result['train_acces']
                    else:
                        print('error: train_acces not in record_result')
                        sys.exit()
                elif args['tuning_criterion'] == 'train_minibatch_acc':
                    
                    assert 'train_minibatch_acces' in record_result
                    
                    record_acc = record_result['train_minibatch_acces']
                    
                elif args['tuning_criterion'] == 'test_acc':
                    if 'test_acces' in record_result:
                        record_acc = record_result['test_acces']
                    else:
                        record_acc = record_result['acces']
                else:
                    print('error: need to check for ' + args['tuning_criterion'])
                    sys.exit()
                
                    

                if args['name_loss'] in ['logistic-regression',
                                         'logistic-regression-sum-loss',
                                         'linear-regression',
                                         'linear-regression-half-MSE']:
                    list_acc.append(np.min(record_acc))
                elif args['name_loss'] in ['multi-class classification',
                                           'binary classification']:
                    list_acc.append(np.max(record_acc))
                else:
                    print('Error: unknown name loss.')
                    sys.exit()
                    
                    
            elif args['tuning_criterion'] == 'train_loss':
                list_acc.append(np.min(record_result['train_losses']))
            elif args['tuning_criterion'] == 'train_minibatch_loss':
                list_acc.append(np.min(record_result['train_unregularized_minibatch_losses']))
            else:
                print('error: unknown tuning criterion for ' + args['tuning_criterion'])
                sys.exit()

            list_name_result_pkl.append(f_)
            list_lr.append(lr)

            
    if list_acc == []:
        return None, None, None
    





    # save the best lr result
    os.chdir(args['home_path'] + 'result/')


    list_acc = np.asarray(list_acc)
    list_name_result_pkl = np.asarray(list_name_result_pkl)
    
    if args['tuning_criterion'] in ['test_acc',
                                    'train_acc',
                                    'train_minibatch_acc']:

        if args['name_loss'] in ['logistic-regression',
                                 'logistic-regression-sum-loss',
                                 'linear-regression',
                                 'linear-regression-half-MSE']:
            # max_indices = np.unravel_index(np.argmin(list_acc, axis=None), list_acc.shape)
            max_indices = np.argmin(list_acc, axis=None)
        elif args['name_loss'] in ['multi-class classification',
                                   'binary classification']:
            # max_indices = np.unravel_index(np.argmax(list_acc, axis=None), list_acc.shape)
            max_indices = np.argmax(list_acc, axis=None)
        else:
            print('Error: unknown name loss when max indices.')
            sys.exit()
            
    elif args['tuning_criterion'] in ['train_loss',
                                      'train_minibatch_loss']:
        
        max_indices = np.argmin(list_acc, axis=None)
            
    else:
        print('error: unknown tuning criterion 2 for ' + args['tuning_criterion'])
        sys.exit()

    print('list_acc[max_indices]')
    print(list_acc[max_indices])

    

    best_lr = list_lr[max_indices]
    best_name_result_pkl = list_name_result_pkl[max_indices]


    # save best params
    fake_params = {}

    # fake_params['algorithm'] = args['algorithm']
    fake_params['algorithm'] = algorithm_dict['name']

    fake_params['if_gpu'] = args['if_gpu']

    name_algorithm, _ = get_name_algorithm(fake_params)
    
    np.savez(
        args['dataset'] + '/' + name_algorithm + '/' + 'best_params' + '.npz',
            best_lr=best_lr
    )

    
    # visualize how to find the best
    if len(list_lr) == len(list_acc) and if_plot:
        

        plt.plot(list_lr, list_acc)
        plt.xlabel('learning rate')
        plt.ylabel('test accuracy')
        plt.xscale('log')
        # plt.title(name_result)
        plt.title(args['dataset'] + '/' + name_algorithm)

        if not os.path.exists(args['home_path'] + 'logs/plot_tune_lr/'):
            os.makedirs(args['home_path'] + 'logs/plot_tune_lr/')
        plt.savefig(args['home_path'] + 'logs/plot_tune_lr/' + str(datetime.datetime.now()) + '.pdf')
        plt.show()
        

    return best_lr, None, best_name_result_pkl

def get_name_algorithm_with_params(params):
    name_algorithm, _ = get_name_algorithm(params)
    
    if isinstance(params['alpha'], str):
        name_algorithm_with_params = name_algorithm + '/' +\
    'alpha_' + params['alpha'] + '/' +\
    'N1_' + str(params['N1']) + '/' +\
    'N2_' + str(params['N2'])
    else:
        name_algorithm_with_params = name_algorithm + '/' +\
    'alpha_' + str(params['alpha']) + '/' +\
    'N1_' + str(params['N1']) + '/' +\
    'N2_' + str(params['N2'])
    
    return name_algorithm_with_params

def get_name_algorithm(params):
    no_algorithm = 'if_gpu_' + str(params['if_gpu'])
    name_algorithm = params['algorithm'] + '/' + no_algorithm
    return name_algorithm, no_algorithm

def get_name_loss(dataset):
    if dataset in ['MNIST',
                   'MNIST-no-regularization',
                   'MNIST-N1-1000',
                   'MNIST-one-layer',
                   'DownScaledMNIST-no-regularization',
                   'DownScaledMNIST-N1-1000-no-regularization',
                   'CIFAR',
                   'CIFAR-deep',
                   'CIFAR-10-vgg16',
                   'CIFAR-10-vgg11',
                   'CIFAR-10-NoAugmentation-vgg11',
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
                   'CIFAR-10-AllCNNC',
                   'CIFAR-10-N1-128-AllCNNC',
                   'CIFAR-10-N1-512-AllCNNC',
                   'CIFAR-10-ConvPoolCNNC',
                   'CIFAR-100',
                   'CIFAR-100-NoAugmentation',
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
                   'CIFAR-100-onTheFly-AllCNNC',
                   'CIFAR-10-onTheFly-ResNet32-BNNoAffine',
                   'CIFAR-10-onTheFly-ResNet32-BN',
                   'CIFAR-10-onTheFly-ResNet32-BN-BNshortcut',
                   'CIFAR-10-onTheFly-ResNet32-BN-BNshortcutDownsampleOnly',
                   'CIFAR-10-onTheFly-ResNet32-BN-BNshortcutDownsampleOnly-NoBias',
                   'CIFAR-10-onTheFly-N1-128-ResNet32-BNNoAffine-PaddingShortcutDownsampleOnly-NoBias-no-regularization',
                   'CIFAR-10-onTheFly-N1-128-ResNet32-BN-BNshortcutDownsampleOnly-NoBias',
                   'CIFAR-10-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias',
                   'CIFAR-10-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias-no-regularization',
                   'CIFAR-10-onTheFly-ResNet32-BNNoAffine-NoBias',
                   'CIFAR-100-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias',
                   'CIFAR-100-onTheFly-ResNet34-BNNoAffine',
                   'CIFAR-100-onTheFly-ResNet34-BN',
                   'CIFAR-100-onTheFly-ResNet34-BN-BNshortcut',
                   'CIFAR-100-onTheFly-ResNet34-BN-BNshortcutDownsampleOnly',
                   'CIFAR-100-onTheFly-ResNet34-BN-BNshortcutDownsampleOnly-NoBias',
                   'CIFAR-100-onTheFly-N1-128-ResNet34-BN-BNshortcutDownsampleOnly-NoBias',
                   'CIFAR-100-onTheFly-N1-128-ResNet34-BN-PaddingShortcutDownsampleOnly-NoBias',
                   'Fashion-MNIST',
                   'Fashion-MNIST-N1-60',
                   'Fashion-MNIST-N1-60-no-regularization',
                   'Fashion-MNIST-N1-256-no-regularization',
                   'Fashion-MNIST-GAP-N1-60-no-regularization',
                   'STL-10-simple-CNN',
                   'Subsampled-ImageNet-simple-CNN',
                   'Subsampled-ImageNet-vgg16', 'SVHN-ResNet34', 'SVHN-vgg11']:
        return 'multi-class classification'
        
    elif dataset == 'webspam':
        return 'binary classification'
    
    elif dataset in ['MNIST-autoencoder',
                     'MNIST-autoencoder-no-regularization',
                     'MNIST-autoencoder-N1-1000',
                     'MNIST-autoencoder-N1-1000-no-regularization',
                     'CURVES-autoencoder-no-regularization',
                     'CURVES-autoencoder',
                     'CURVES-autoencoder-Botev',
                     'CURVES-autoencoder-shallow',
                     'FACES-autoencoder',
                     'FACES-autoencoder-no-regularization']:
        return 'logistic-regression'
    
    elif dataset in ['MNIST-autoencoder-N1-1000-sum-loss',
                     'MNIST-autoencoder-N1-1000-sum-loss-no-regularization',
                     'MNIST-autoencoder-relu-N1-1000-sum-loss-no-regularization',
                     'MNIST-autoencoder-relu-N1-1000-sum-loss',
                     'MNIST-autoencoder-relu-N1-100-sum-loss',
                     'MNIST-autoencoder-relu-N1-500-sum-loss',
                     'MNIST-autoencoder-relu-N1-1-sum-loss',
                     'MNIST-autoencoder-reluAll-N1-1-sum-loss',
                     'FACES-autoencoder-sum-loss-no-regularization',
                     'FACES-autoencoder-relu-sum-loss-no-regularization',
                     'FACES-autoencoder-relu-sum-loss',
                     'FACES-autoencoder-sum-loss',
                     'CURVES-autoencoder-sum-loss-no-regularization',
                     'CURVES-autoencoder-sum-loss',
                     'CURVES-autoencoder-relu-sum-loss-no-regularization',
                     'CURVES-autoencoder-relu-sum-loss',
                     'CURVES-autoencoder-relu-N1-100-sum-loss',
                     'CURVES-autoencoder-relu-N1-500-sum-loss',
                     'CURVES-autoencoder-Botev-sum-loss-no-regularization',]:
        
        return 'logistic-regression-sum-loss'
    
    elif dataset in ['sythetic-linear-regression',
                     'sythetic-linear-regression-N1-1']:
        return 'linear-regression'
    elif dataset in ['FacesMartens-autoencoder-relu',
                     'FacesMartens-autoencoder-relu-no-regularization',
                     'FacesMartens-autoencoder-relu-N1-500',
                     'FacesMartens-autoencoder-relu-N1-100']:
        return 'linear-regression-half-MSE'
    else:
        print('Error: Problem not specified.')
        sys.exit()
        
def from_dataset_to_N1_N2(args):
    
    if not 'tau' in args:
        args['tau'] = 10**(-5) # https://arxiv.org/pdf/1503.05671.pdf
    
    if 'N1' in args or 'N2' in args:
        print('error: N1, N2 not automated')
        sys.exit()
    else:
        if args['dataset'] == 'MNIST-N1-1000':
            args['N1'] = 1000
            args['N2'] = 1000
        elif args['dataset'] == 'Fashion-MNIST':
            args['N1'] = 1000
            args['N2'] = 1000
        elif args['dataset'] == 'Fashion-MNIST-N1-60':
            args['N1'] = 60
            args['N2'] = 60
        elif args['dataset'] == 'Fashion-MNIST-N1-60-no-regularization':
            args['N1'] = 60
            args['N2'] = 60
            args['tau'] = 0
        elif args['dataset'] == 'Fashion-MNIST-N1-256-no-regularization':
            # https://arxiv.org/pdf/1910.05446.pdf
            args['N1'] = 256
            args['N2'] = 256
            args['tau'] = 0
        elif args['dataset'] == 'Fashion-MNIST-GAP-N1-60-no-regularization':
            args['N1'] = 60
            args['N2'] = 60
            args['tau'] = 0
        elif args['dataset'] == 'webspam':
            args['N1'] = 1000
            args['N2'] = 1000
        elif args['dataset'] == 'MNIST':
            args['N1'] = 60
            args['N2'] = 60
        elif args['dataset'] == 'MNIST-no-regularization':
            args['N1'] = 60
            args['N2'] = 60
            args['tau'] = 0
        elif args['dataset'] == 'DownScaledMNIST-no-regularization':
            args['N1'] = 60
            args['N2'] = 60
            args['tau'] = 0
        elif args['dataset'] == 'DownScaledMNIST-N1-1000-no-regularization':
            args['N1'] = 1000
            args['N2'] = 1000
            args['tau'] = 0
        elif args['dataset'] == 'MNIST-autoencoder':
            args['N1'] = 60
            args['N2'] = 60
        elif args['dataset'] == 'MNIST-autoencoder-no-regularization':
            args['N1'] = 60
            args['N2'] = 60
            args['tau'] = 0
        elif args['dataset'] == 'MNIST-autoencoder-N1-1000':
            args['N1'] = 1000
            args['N2'] = 1000
        elif args['dataset'] == 'MNIST-autoencoder-N1-1000-sum-loss':
            args['N1'] = 1000
            args['N2'] = 1000
        elif args['dataset'] == 'MNIST-autoencoder-N1-1000-no-regularization':
            args['N1'] = 1000
            args['N2'] = 1000
            args['tau'] = 0
        elif args['dataset'] == 'MNIST-autoencoder-N1-1000-sum-loss-no-regularization':
            args['N1'] = 1000
            args['N2'] = 1000
            args['tau'] = 0
        elif args['dataset'] == 'MNIST-autoencoder-relu-N1-1000-sum-loss-no-regularization':
            args['N1'] = 1000
            args['N2'] = 1000
            args['tau'] = 0
        elif args['dataset'] == 'MNIST-autoencoder-relu-N1-1000-sum-loss':
            args['N1'] = 1000
            args['N2'] = 1000
        elif args['dataset'] == 'MNIST-autoencoder-relu-N1-100-sum-loss':
            args['N1'] = 100
            args['N2'] = 100
            args['tau'] = 10**(-5)
        elif args['dataset'] == 'MNIST-autoencoder-relu-N1-500-sum-loss':
            args['N1'] = 500
            args['N2'] = 500
        elif args['dataset'] == 'MNIST-autoencoder-relu-N1-1-sum-loss':
            args['N1'] = 1
            args['N2'] = 1
        elif args['dataset'] == 'MNIST-autoencoder-reluAll-N1-1-sum-loss':
            args['N1'] = 1
            args['N2'] = 1
        elif args['dataset'] == 'CURVES-autoencoder':
            args['N1'] = 1000
            args['N2'] = 1000
        elif args['dataset'] == 'CURVES-autoencoder-Botev':
            args['N1'] = 1000
            args['N2'] = 1000
        elif args['dataset'] == 'CURVES-autoencoder-Botev-sum-loss-no-regularization':
            args['N1'] = 1000
            args['N2'] = 1000
            args['tau'] = 0
        elif args['dataset'] == 'CURVES-autoencoder-no-regularization':
            args['N1'] = 1000
            args['N2'] = 1000
            args['tau'] = 0
        elif args['dataset'] == 'CURVES-autoencoder-sum-loss-no-regularization':
            args['N1'] = 1000
            args['N2'] = 1000
            args['tau'] = 0
        elif args['dataset'] == 'CURVES-autoencoder-relu-sum-loss-no-regularization':
            args['N1'] = 1000
            args['N2'] = 1000
            args['tau'] = 0
        elif args['dataset'] == 'CURVES-autoencoder-relu-sum-loss':
            args['N1'] = 1000
            args['N2'] = 1000
        elif args['dataset'] == 'CURVES-autoencoder-relu-N1-500-sum-loss':
            args['N1'] = 500
            args['N2'] = 500
        elif args['dataset'] == 'CURVES-autoencoder-relu-N1-100-sum-loss':
            args['N1'] = 100
            args['N2'] = 100
            args['tau'] = 10**(-5)
        elif args['dataset'] == 'CURVES-autoencoder-sum-loss':
            args['N1'] = 1000
            args['N2'] = 1000
        elif args['dataset'] == 'FACES-autoencoder':
            args['N1'] = 1000
            args['N2'] = 1000
        elif args['dataset'] == 'FACES-autoencoder-no-regularization':
            args['N1'] = 1000
            args['N2'] = 1000
            args['tau'] = 0
        elif args['dataset'] == 'FACES-autoencoder-sum-loss-no-regularization':
            args['N1'] = 1000
            args['N2'] = 1000
            args['tau'] = 0
        elif args['dataset'] == 'FACES-autoencoder-relu-sum-loss-no-regularization':
            args['N1'] = 1000
            args['N2'] = 1000
            args['tau'] = 0
        elif args['dataset'] == 'FACES-autoencoder-relu-sum-loss':
            args['N1'] = 1000
            args['N2'] = 1000
        elif args['dataset'] == 'FacesMartens-autoencoder-relu':
            args['N1'] = 1000
            args['N2'] = 1000
        elif args['dataset'] == 'FacesMartens-autoencoder-relu-no-regularization':
            args['N1'] = 1000
            args['N2'] = 1000
            args['tau'] = 0
        elif args['dataset'] == 'FacesMartens-autoencoder-relu-N1-500':
            args['N1'] = 500
            args['N2'] = 500
        elif args['dataset'] == 'FacesMartens-autoencoder-relu-N1-100':
            args['N1'] = 100
            args['N2'] = 100
            args['tau'] = 10**(-5)
        elif args['dataset'] == 'FACES-autoencoder-sum-loss':
            args['N1'] = 1000
            args['N2'] = 1000
        elif args['dataset'] == 'sythetic-linear-regression':
            args['N1'] = 900
            args['N2'] = 900
        elif args['dataset'] == 'sythetic-linear-regression-N1-1':
            args['N1'] = 1
            args['N2'] = 1
        elif args['dataset'] == 'MNIST-one-layer':
            args['N1'] = 60000
            args['N2'] = 60000
        elif args['dataset'] == 'UCI-HAR':
            args['N1'] = 32
            args['N2'] = 32
        elif args['dataset'] == 'CIFAR-100':
            
            print('error: data is augmented but not on the fly')
            sys.exit()
            
            args['N1'] = 1000
            args['N2'] = 1000
        elif args['dataset'] == 'CIFAR-100-NoAugmentation':
            
            args['N1'] = 1000
            args['N2'] = 1000
        elif args['dataset'] == 'CIFAR-100-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout':
            
            args['N1'] = 1000
            args['N2'] = 1000
            
            args['tau'] = 0.0005 # https://arxiv.org/pdf/1910.05446.pdf
            
        elif args['dataset'] == 'CIFAR-100-NoAugmentation-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout':
            
            args['N1'] = 256
            args['N2'] = 256
            
            args['tau'] = 0.0005 # https://arxiv.org/pdf/1910.05446.pdf
            
        elif args['dataset'] == 'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout':
            
            args['N1'] = 128
            args['N2'] = 128
            
            args['tau'] = 0.0005 # https://arxiv.org/pdf/1910.05446.pdf
            
        elif args['dataset'] == 'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine-no-regularization':
            
            args['N1'] = 128
            args['N2'] = 128
            
            args['tau'] = 0
            
        elif args['dataset'] == 'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN':
            
            args['N1'] = 128
            args['N2'] = 128
            
            args['tau'] = 0.0005 # https://arxiv.org/pdf/1910.05446.pdf
            
        elif args['dataset'] == 'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN-no-regularization':
            
            args['N1'] = 128
            args['N2'] = 128
            
            args['tau'] = 0
            
        elif args['dataset'] == 'SVHN-vgg11':
            args['N1'] = 128
            args['N2'] = 128
            
            args['tau'] = 0
            
        elif args['dataset'] == 'CIFAR-100-onTheFly-vgg16-NoLinear-BN-no-regularization':
            
            args['N1'] = 128
            args['N2'] = 128
            
            args['tau'] = 0
            
        elif args['dataset'] == 'CIFAR-100-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout':
            
            args['N1'] = 256
            args['N2'] = 256
            
            args['tau'] = 0.0005 # https://arxiv.org/pdf/1910.05446.pdf
            
        elif args['dataset'] == 'CIFAR-100-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout-no-regularization':
            
            args['N1'] = 256
            args['N2'] = 256
            
            args['tau'] = 0.0
            
        elif args['dataset'] == 'CIFAR-100-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine':
            
            args['N1'] = 256
            args['N2'] = 256
            
            args['tau'] = 0.0005 # https://arxiv.org/pdf/1910.05446.pdf
            
        elif args['dataset'] == 'CIFAR-100-onTheFly-AllCNNC':
            
            args['N1'] = 256
            args['N2'] = 256
            
            args['tau'] = 0.0005 # https://arxiv.org/pdf/1910.05446.pdf
            
        elif args['dataset'] == 'CIFAR-deep':
            args['N1'] = 128
            args['N2'] = 128
            args['tau'] = 0.0005
        elif args['dataset'] == 'CIFAR-10-vgg11':
            args['N1'] = 128
            args['N2'] = 128
            args['tau'] = 0.0005
        elif args['dataset'] == 'CIFAR-10-NoAugmentation-vgg11':
            args['N1'] = 128
            args['N2'] = 128
            args['tau'] = 0.0005
        elif args['dataset'] == 'CIFAR-10-vgg11-test':
            args['N1'] = 128
            args['N2'] = 128
            args['tau'] = 0.0005
        elif args['dataset'] == 'CIFAR-10-vgg16':
            # https://arxiv.org/pdf/1910.05446.pdf
            
            print('adaptive avg pool is not neede for CIFAR10 + vgg16')
            sys.exit()
            
            args['N1'] = 128
            args['N2'] = 128
            args['tau'] = 0.0005 # https://arxiv.org/pdf/1910.05446.pdf
        elif args['dataset'] == 'CIFAR-10-vgg16-NoAdaptiveAvgPoolNoDropout':
            
            print('error: should use CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout')
            sys.exit()
            
            # https://arxiv.org/pdf/1910.05446.pdf
            args['N1'] = 128
            args['N2'] = 128
            args['tau'] = 0.0005 # https://arxiv.org/pdf/1910.05446.pdf
            
        elif args['dataset'] == 'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout':
            
            # https://arxiv.org/pdf/1910.05446.pdf
            args['N1'] = 128
            args['N2'] = 128
            args['tau'] = 0.0005 # https://arxiv.org/pdf/1910.05446.pdf
            
        elif args['dataset'] == 'CIFAR-10-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout':
            # https://arxiv.org/pdf/1910.05446.pdf
            args['N1'] = 256
            args['N2'] = 256
            args['tau'] = 0.0005 # https://arxiv.org/pdf/1910.05446.pdf
            
        elif args['dataset'] == 'CIFAR-10-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout-no-regularization':
            # https://arxiv.org/pdf/1910.05446.pdf
            args['N1'] = 256
            args['N2'] = 256
            args['tau'] = 0.0
            
        elif args['dataset'] == 'CIFAR-10-onTheFly-N1-256-vgg16-NoAdaptiveAvgPool':
            # https://arxiv.org/pdf/1910.05446.pdf
            args['N1'] = 256
            args['N2'] = 256
            args['tau'] = 0.0005 # https://arxiv.org/pdf/1910.05446.pdf
            
        elif args['dataset'] == 'CIFAR-10-onTheFly-N1-512-vgg16-NoAdaptiveAvgPoolNoDropout':
            # https://arxiv.org/pdf/1910.05446.pdf
            args['N1'] = 512
            args['N2'] = 512
            args['tau'] = 0.0005 # https://arxiv.org/pdf/1910.05446.pdf
            
        elif args['dataset'] == 'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN':
            
            # https://arxiv.org/pdf/1910.05446.pdf
            args['N1'] = 128
            args['N2'] = 128
            args['tau'] = 0.0005 # https://arxiv.org/pdf/1910.05446.pdf
            
        elif args['dataset'] == 'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN-NoBias':
            
            # https://arxiv.org/pdf/1910.05446.pdf
            args['N1'] = 128
            args['N2'] = 128
            args['tau'] = 0.0005 # https://arxiv.org/pdf/1910.05446.pdf
            
        elif args['dataset'] == 'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine':
            
            # https://arxiv.org/pdf/1910.05446.pdf
            args['N1'] = 128
            args['N2'] = 128
            args['tau'] = 0.0005 # https://arxiv.org/pdf/1910.05446.pdf
            
        elif args['dataset'] == 'CIFAR-10-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout':
            # https://arxiv.org/pdf/1910.05446.pdf
            args['N1'] = 128
            args['N2'] = 128
            args['tau'] = 0.0005 # https://arxiv.org/pdf/1910.05446.pdf
        elif args['dataset'] == 'CIFAR-10-NoAugmentation-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout':
            # https://arxiv.org/pdf/1910.05446.pdf
            args['N1'] = 128
            args['N2'] = 128
            args['tau'] = 0.0005 # https://arxiv.org/pdf/1910.05446.pdf
        elif args['dataset'] == 'CIFAR-10-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout-BN':
            # https://arxiv.org/pdf/1910.05446.pdf
            args['N1'] = 128
            args['N2'] = 128
            args['tau'] = 0.0005 # https://arxiv.org/pdf/1910.05446.pdf
        elif args['dataset'] == 'CIFAR-10-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine':
            # https://arxiv.org/pdf/1910.05446.pdf
            args['N1'] = 128
            args['N2'] = 128
            args['tau'] = 0.0005 # https://arxiv.org/pdf/1910.05446.pdf
        elif args['dataset'] == 'CIFAR-10-vgg16-GAP':
            
            print('GAP is not needed for CIFAR10 + vgg16')
            sys.exit()
            
            args['N1'] = 128
            args['N2'] = 128
            args['tau'] = 0.0005
            
        elif args['dataset'] == 'CIFAR-10-onTheFly-ResNet32-BNNoAffine':
            # https://arxiv.org/pdf/1910.05446.pdf
            args['N1'] = 256
            args['N2'] = 256
            
            # sec 4.2 of https://arxiv.org/pdf/1512.03385.pdf
            args['tau'] = 0.0001 # inherit from VGG
            
        elif args['dataset'] == 'CIFAR-10-onTheFly-ResNet32-BN':
            # https://arxiv.org/pdf/1910.05446.pdf
            args['N1'] = 256
            args['N2'] = 256
            
            # sec 4.2 of https://arxiv.org/pdf/1512.03385.pdf
            args['tau'] = 0.0001 # inherit from VGG
            
        elif args['dataset'] == 'CIFAR-10-onTheFly-ResNet32-BN-BNshortcut':
            # https://arxiv.org/pdf/1910.05446.pdf
            args['N1'] = 256
            args['N2'] = 256
            
            # sec 4.2 of https://arxiv.org/pdf/1512.03385.pdf
            args['tau'] = 0.0001 # inherit from VGG
            
        elif args['dataset'] == 'CIFAR-10-onTheFly-ResNet32-BN-BNshortcutDownsampleOnly':
            # https://arxiv.org/pdf/1910.05446.pdf
            args['N1'] = 256
            args['N2'] = 256
            
            # sec 4.2 of https://arxiv.org/pdf/1512.03385.pdf
            args['tau'] = 0.0001 # inherit from VGG
            
        elif args['dataset'] == 'CIFAR-10-onTheFly-ResNet32-BN-BNshortcutDownsampleOnly-NoBias':
            # https://arxiv.org/pdf/1910.05446.pdf
            args['N1'] = 256
            args['N2'] = 256
            
            # sec 4.2 of https://arxiv.org/pdf/1512.03385.pdf
            args['tau'] = 0.0001 # inherit from VGG
            
        elif args['dataset'] == 'CIFAR-10-onTheFly-N1-128-ResNet32-BNNoAffine-PaddingShortcutDownsampleOnly-NoBias-no-regularization':
            # https://arxiv.org/pdf/1910.05446.pdf
            args['N1'] = 128
            args['N2'] = 128
            
            args['tau'] = 0
            
        elif args['dataset'] == 'CIFAR-10-onTheFly-N1-128-ResNet32-BN-BNshortcutDownsampleOnly-NoBias':
            # https://arxiv.org/pdf/1910.05446.pdf
            args['N1'] = 128
            args['N2'] = 128
            
            # sec 4.2 of https://arxiv.org/pdf/1512.03385.pdf
            args['tau'] = 0.0001 # inherit from VGG
            
        elif args['dataset'] == 'CIFAR-10-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias':
            # https://arxiv.org/pdf/1910.05446.pdf
            args['N1'] = 128
            args['N2'] = 128
            
            # sec 4.2 of https://arxiv.org/pdf/1512.03385.pdf
            args['tau'] = 0.0001 # inherit from VGG
            
        elif args['dataset'] == 'CIFAR-10-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias-no-regularization':
            # https://arxiv.org/pdf/1910.05446.pdf
            args['N1'] = 128
            args['N2'] = 128
            
            args['tau'] = 0
        
        elif args['dataset'] == 'SVHN-ResNet34':
            args['N1'] = 128
            args['N2'] = 128
            
            args['tau'] = 0
        elif args['dataset'] == 'CIFAR-100-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias':
            # https://arxiv.org/pdf/1910.05446.pdf
            args['N1'] = 128
            args['N2'] = 128
            
            # sec 4.2 of https://arxiv.org/pdf/1512.03385.pdf
            args['tau'] = 0.0001 # inherit from VGG
            
        elif args['dataset'] == 'CIFAR-10-onTheFly-ResNet32-BNNoAffine-NoBias':
            # https://arxiv.org/pdf/1910.05446.pdf
            args['N1'] = 256
            args['N2'] = 256
            
            # sec 4.2 of https://arxiv.org/pdf/1512.03385.pdf
            args['tau'] = 0.0001 # inherit from VGG
            
        elif args['dataset'] == 'CIFAR-100-onTheFly-ResNet34-BNNoAffine':
            # https://zhenye-na.github.io/2018/10/07/pytorch-resnet-cifar100.html
            
            args['N1'] = 256
            args['N2'] = 256
            args['tau'] = 1e-5
            
        elif args['dataset'] == 'CIFAR-100-onTheFly-ResNet34-BN':
            # https://zhenye-na.github.io/2018/10/07/pytorch-resnet-cifar100.html
            
            args['N1'] = 256
            args['N2'] = 256
            args['tau'] = 1e-5
            
        elif args['dataset'] == 'CIFAR-100-onTheFly-ResNet34-BN-BNshortcut':
            # https://zhenye-na.github.io/2018/10/07/pytorch-resnet-cifar100.html
            
            args['N1'] = 256
            args['N2'] = 256
            args['tau'] = 1e-5
            
        elif args['dataset'] == 'CIFAR-100-onTheFly-ResNet34-BN-BNshortcutDownsampleOnly':
            # https://zhenye-na.github.io/2018/10/07/pytorch-resnet-cifar100.html
            
            args['N1'] = 256
            args['N2'] = 256
            args['tau'] = 1e-5
            
        elif args['dataset'] == 'CIFAR-100-onTheFly-ResNet34-BN-BNshortcutDownsampleOnly-NoBias':
            # https://zhenye-na.github.io/2018/10/07/pytorch-resnet-cifar100.html
            
            args['N1'] = 256
            args['N2'] = 256
            args['tau'] = 1e-5
            
        elif args['dataset'] == 'CIFAR-100-onTheFly-N1-128-ResNet34-BN-BNshortcutDownsampleOnly-NoBias':
            # https://github.com/weiaicunzai/pytorch-cifar100/blob/master/train.py
            # https://github.com/bearpaw/pytorch-classification/blob/master/cifar.py
            
            args['N1'] = 128
            args['N2'] = 128
            args['tau'] = 5e-4
            
        elif args['dataset'] == 'CIFAR-100-onTheFly-N1-128-ResNet34-BN-PaddingShortcutDownsampleOnly-NoBias':
            # https://github.com/weiaicunzai/pytorch-cifar100/blob/master/train.py
            # https://github.com/bearpaw/pytorch-classification/blob/master/cifar.py
            
            args['N1'] = 128
            args['N2'] = 128
            args['tau'] = 5e-4
            
        elif args['dataset'] == 'CIFAR-10-AllCNNC':
            # https://arxiv.org/pdf/1910.05446.pdf
            args['N1'] = 256
            args['N2'] = 256
            args['tau'] = 0.0005
        elif args['dataset'] == 'CIFAR-10-N1-128-AllCNNC':
            # https://arxiv.org/pdf/1910.05446.pdf
            args['N1'] = 128
            args['N2'] = 128
            args['tau'] = 0.0005
        elif args['dataset'] == 'CIFAR-10-N1-512-AllCNNC':
            # https://arxiv.org/pdf/1910.05446.pdf
            args['N1'] = 512
            args['N2'] = 512
            args['tau'] = 0.0005
        elif args['dataset'] == 'CIFAR-10-ConvPoolCNNC':
            # https://arxiv.org/pdf/1910.05446.pdf
            args['N1'] = 256
            args['N2'] = 256
            args['tau'] = 0.0005
        elif args['dataset'] == 'STL-10-simple-CNN':
            args['N1'] = 1000
            args['N2'] = 1000
        elif args['dataset'] == 'Subsampled-ImageNet-simple-CNN':
            args['N1'] = 100
            args['N2'] = 100
        elif args['dataset'] == 'Subsampled-ImageNet-vgg16':
            args['N1'] = 10
            args['N2'] = 10
        else:
            print('error: unknown dataset for ' + args['dataset'])
            sys.exit()
    return args

def tune_lr(args):
    assert 'max_epoch/time' in args
    assert 'record_epoch' in args
    assert 'if_test_mode' in args
    
    assert 'if_grafting' in args
    assert 'weight_decay' in args

    args['name_loss'] = get_name_loss(args['dataset'])
    args = from_dataset_to_N1_N2(args)
    args = add_matrix_name_to_args(args)
    args = add_some_if_record_to_args(args)
    
    args['momentum_gradient_rho'] = 0.9
    args['lambda_'] = 1
    
    if args['if_auto_tune_lr']:
        
        assert len(args['list_lr']) == 2
        
        for learning_rate in args['list_lr']:
            args['alpha'] = learning_rate
            name_result, data_, params_saved = train(args)
            
            print_gpu_usage({'device': 'cuda:0'})
            
            data_ = None
            torch.cuda.empty_cache()
            
            print_gpu_usage({'device': 'cuda:0'})
            
        list_lr_tried = args['list_lr']
        
        while 1:
            fake_args = {}
            fake_args['home_path'] = args['home_path']
            fake_args['algorithm_dict'] = {}
            fake_args['algorithm_dict']['name'] = args['algorithm']
            fake_args['algorithm_dict']['params'] = params_saved
            
            fake_args['if_gpu'] = args['if_gpu']
            fake_args['dataset'] = args['dataset']
            fake_args['N1'] = args['N1']
            fake_args['N2'] = args['N2']
            fake_args['name_loss'] = args['name_loss']

            fake_args['tuning_criterion'] = args['tuning_criterion']
            fake_args['list_lr'] = list_lr_tried

            best_lr, _, best_name_result_pkl = get_best_params(fake_args, False)

            print('best_lr')
            print(best_lr)
            

            
            learning_rate = get_next_lr(list_lr_tried, best_lr)
            
            if learning_rate < 0:
                break
            else:
                args['alpha'] = learning_rate
                name_result, data_, params_saved = train(args)
                
                data_ = None
                torch.cuda.empty_cache()
                
                
                
                if learning_rate < min(list_lr_tried):
                    list_lr_tried = [learning_rate] + list_lr_tried
                elif learning_rate > max(list_lr_tried):
                    list_lr_tried = list_lr_tried + [learning_rate]
                else:
                    print('there is an error')
                    sys.exit()
                    
                    
        print('list_lr_tried, best_lr')
        print(list_lr_tried, best_lr)
    else:
        for learning_rate in args['list_lr']:
            args['alpha'] = learning_rate
            name_result, data_, _ = train(args)
            
            data_ = None
            torch.cuda.empty_cache()
    return data_

def get_if_stop(args, i, iter_per_epoch, timesCPU):
    if args['if_max_epoch']:
        if i < int(args['max_epoch/time'] * iter_per_epoch):
            return False
        else:
            return True
    else:
        
        
        if timesCPU[-1] < args['max_epoch/time']:
            return False
        else:
            return True

def update_parameter(p_torch, model, params):
    numlayers = params['numlayers']
    alpha = params['alpha_current']
    
    device = params['device']

    
    for l in range(numlayers):
        
        for name_variable in model.layers_weight[l].keys():
            
            if params['weight_decay'] != 0:
                model.layers_weight[l][name_variable].data *= (1 - alpha*params['weight_decay'])
            
            model.layers_weight[l][name_variable].data += alpha * p_torch[l][name_variable].data  
    return model

def get_model(params):
    model = Model_3(params)
    if params['if_gpu']:
        model.to(params['device'])
    return model

def train_initialization(data_, params, args):
    algorithm = params['algorithm']
    
    if params['N2'] > params['N1']:
        print('Error! 1432')
        sys.exit()
        
    params['if_grafting'] = args['if_grafting']
    
    params['weight_decay'] = args['weight_decay']
        
    if params['if_lr_decay']:
        params['num_epoch_to_decay'] = args['num_epoch_to_decay']
        params['lr_decay_rate'] = args['lr_decay_rate']
        
    if algorithm in ['kfac-correctFisher-warmStart-no-max-no-LM',
                 'kfac-correctFisher-warmStart-lessInverse-no-max-no-LM']:
        
        params['kfac_if_svd'] = args['kfac_if_svd']
        
        params['kfac_if_update_BN'] = args['kfac_if_update_BN']
        params['kfac_if_BN_grad_direction'] = args['kfac_if_BN_grad_direction']
        
        if params['kfac_if_update_BN'] == False and params['weight_decay'] != 0:
            print('error: only work if weight_decay == 0')
            sys.exit()
        
        if algorithm in ['kfac-no-max-no-LM',
                         'kfac-warmStart-no-max-no-LM',
                         'kfac-correctFisher-warmStart-no-max-no-LM',
                         'kfac-correctFisher-warmStart-NoMaxNoSqrt-no-LM',
                         'kfac-correctFisher-warmStart-lessInverse-no-max-no-LM',
                         'kfac-correctFisher-warmStart-lessInverse-NoMaxNoSqrt-no-LM',
                         'kfac-warmStart-lessInverse-no-max-no-LM',
                         'kfac-warmStart-lessInverse-NoMaxNoSqrt-no-LM']:
            params['Kron_BFGS_if_homo'] = True
        
        if algorithm in ['ekfac-EF-VA',
                         'ekfac-EF',
                         'kfac-EF']:
            print('error: need to check warm start')
            sys.exit()

        if params['algorithm'] in ['kfac-TR',
                                   'kfac-momentum-grad-TR']:
            params['TR_max_iter'] = args['TR_max_iter']
        if params['algorithm'] in ['kfac-CG',
                                   'kfac-momentum-grad-CG']:
            params['CG_max_iter'] = args['CG_max_iter']
            
        if params['algorithm'] in ['kfac-no-max-no-LM',
                                   'kfac-warmStart-no-max-no-LM',
                                   'kfac-correctFisher-warmStart-no-max-no-LM',
                                   'kfac-correctFisher-warmStart-NoMaxNoSqrt-no-LM',
                                   'kfac-correctFisher-warmStart-lessInverse-no-max-no-LM',
                                   'kfac-correctFisher-warmStart-lessInverse-NoMaxNoSqrt-no-LM',
                                   'kfac-warmStart-lessInverse-no-max-no-LM',
                                   'kfac-warmStart-lessInverse-NoMaxNoSqrt-no-LM',
                                   'kfac-NoMaxNoSqrt-no-LM']:
            params['kfac_damping_lambda'] = args['kfac_damping_lambda']
            
        if params['algorithm'] == 'kfac-no-max-epsilon-A-G-no-LM':
            params['kfac_A_epsilon'] = args['kfac_A_epsilon']
            params['kfac_G_epsilon'] = args['kfac_G_epsilon']
            
        

        device = params['device']
        layersizes = params['layersizes']
        numlayers = params['numlayers']
        
        layers_params = params['layers_params']

        A = []  # KFAC A
        G = []  # KFAC G


        for l in range(numlayers):
            if params['layers_params'][l]['name'] == 'fully-connected':
                
                input_size = params['layers_params'][l]['input_size']
                output_size = params['layers_params'][l]['output_size']

                
                A.append(torch.zeros(input_size + 1, input_size + 1, device=device))
                
                

                G.append(torch.zeros(output_size, output_size, device=device))
                
            elif params['layers_params'][l]['name'] in ['conv',
                                                        'conv-no-activation',
                                                        'conv-no-bias-no-activation',]:
            
                size_A = layers_params[l]['conv_in_channels'] *\
                layers_params[l]['conv_kernel_size']**2
                
                if params['layers_params'][l]['name'] in ['conv',
                                                          'conv-no-activation',]:
                
                    size_A += 1
            
                A.append(torch.zeros(size_A, size_A, device=device))
                
                size_G = layers_params[l]['conv_out_channels']
                
                G.append(torch.zeros(size_G, size_G, device=device))
            elif params['layers_params'][l]['name'] in ['BN']:

                
                A.append([])
                
                size_G = layers_params[l]['num_features'] * 2
                
                G.append(torch.zeros(size_G, size_G, device=device))
            else:
                print('Error: unsupported layer when initial cache for ' + params['layers_params'][l]['name'])
                sys.exit()

        data_['A'] = A
        data_['G'] = G


        if params['algorithm'] in ['kfac',
                                   'kfac-no-max',
                                   'kfac-NoMaxNoSqrt',
                                   'kfac-NoMaxNoSqrt-no-LM',
                                   'kfac-no-max-no-LM',
                                   'kfac-warmStart-no-max-no-LM',
                                   'kfac-correctFisher-warmStart-no-max-no-LM',
                                   'kfac-correctFisher-warmStart-NoMaxNoSqrt-no-LM',
                                   'kfac-correctFisher-warmStart-lessInverse-no-max-no-LM',
                                   'kfac-correctFisher-warmStart-lessInverse-NoMaxNoSqrt-no-LM',
                                   'kfac-warmStart-lessInverse-no-max-no-LM',
                                   'kfac-warmStart-lessInverse-NoMaxNoSqrt-no-LM',
                                   'kfac-no-max-epsilon-A-G-no-LM']:

            
            if params['kfac_if_svd']:
                U_A, U_G = numlayers * [0], numlayers * [0]
                
                data_['U_A'] = U_A
                data_['U_G'] = U_G
                
                s_A, s_G = numlayers * [0], numlayers * [0]
                
                data_['s_A'] = s_A
                data_['s_G'] = s_G
                
            else:
            
                A_inv, G_inv = numlayers * [0], numlayers * [0]

                data_['A_inv'] = A_inv
                data_['G_inv'] = G_inv


        params['kfac_inverse_update_freq'] = args['kfac_inverse_update_freq']
        params['kfac_cov_update_freq'] = args['kfac_cov_update_freq']
        params['kfac_rho'] = args['kfac_rho']
        
        get_warm_start(data_, params)
        
    elif algorithm in ['MBNGD-all-to-one-LRdecay', 
                       'MBNGD-all-to-one', 
                       'MBNGD-one-to-all', 
                       'MBNGD-m1m2', 
                       'MBNGD-all-to-one-Avg', 
                       'L-MBNGD-all-to-one', 
                       'L-MBNGD-all-to-one-LRdecay']:
        device = params['device']
        layersizes = params['layersizes']
        numlayers = params['numlayers']
        
        params['mbngd_damping_lambda'] = args['mbngd_damping_lambda']
        params['mbngd_damping_epsilon'] = args['mbngd_damping_epsilon']
        layers_params = params['layers_params']
        fbf_m = []  # fbf_m

        for l in range(numlayers):
            delta_l = {}
            
            if params['layers_params'][l]['name'] == 'fully-connected':
                if algorithm == 'MBNGD-m1m2':
                    delta_l['W'] = torch.zeros(layers_params[l]['output_size'], (layers_params[l]['input_size']+1), (layers_params[l]['input_size']+1), device=device)
                elif algorithm == 'MBNGD-one-to-all':
                    delta_l['W'] = torch.zeros(layers_params[l]['input_size']+1, layers_params[l]['output_size'], layers_params[l]['output_size'], device=device)
                elif algorithm == 'MBNGD-all-to-one':
                    I = layers_params[l]['input_size']
                    O = layers_params[l]['output_size']
                    m1 = I+1
                    m2 = 1
                    delta_l['m1m2'] = (m1, m2)
                    delta_l['W'] = torch.zeros(m1*m2, (O//m2)*((I+1)//m1), (O//m2)*((I+1)//m1), device=device)
                elif algorithm == 'MBNGD-all-to-one-Avg':
                    I = layers_params[l]['input_size']
                    O = layers_params[l]['output_size']
                    m1 = I+1
                    m2 = 1
                    delta_l['W'] = torch.zeros((O//m2)*((I+1)//m1), (O//m2)*((I+1)//m1), device=device)
                    delta_l['m1m2'] = (m1, m2)
                    # print('I,O = '+str((I, O)))
                    # print('F_shape = '+str(delta_l['W'].shape))
                elif algorithm == 'L-MBNGD-all-to-one':
                    I = layers_params[l]['input_size']
                    O = layers_params[l]['output_size']
                    if args['dataset'] == 'MNIST-autoencoder-relu-N1-1000-sum-loss-no-regularization':
                        r = args['window']
                        layersizes = [784, 1000, 500, 250, 30, 250, 500, 1000, 784]
                        m1 = I+1
                        m2 = 1
                    elif args['dataset'] == 'FacesMartens-autoencoder-relu-no-regularization':
                        r = args['window']
                        layersizes = [625, 2000, 1000, 500, 30, 500, 1000, 2000, 625]
                        m1 = I+1
                        m2 = 1
                    elif args['dataset'] == 'CURVES-autoencoder-relu-sum-loss-no-regularization':
                        layersizes = [784, 400, 200, 100, 50, 25, 6, 25, 50, 100, 200, 400, 784]
                        m1 = I+1
                        m2 = 1
                    else:
                        m1 = I+1
                        m2 = 1
                    print(I, O)
                    print(m1, m2)
                    if O < -1:
                        delta_l['W'] = torch.zeros(m1*m2, O//m2, O//m2, device=device)
                    else:
                        delta_l['Gt'] = torch.zeros((I+1, O, r), device = device, requires_grad=False) 
                        delta_l['Gt_m'] = torch.zeros((I+1, O, min(r,args['kfac_cov_update_freq'])), device = device, requires_grad=False) 
                else:
                    delta_l['W'] = torch.zeros(layers_params[l]['output_size'], layers_params[l]['input_size'], device=device)
                    delta_l['b'] = torch.zeros(layers_params[l]['output_size'], device=device)
                
            elif params['layers_params'][l]['name'] in ['conv',
                                                        'conv-no-activation',
                                                        'conv-no-bias-no-activation']:
                delta_l['W'] = torch.zeros(layers_params[l]['conv_out_channels'],
                                 layers_params[l]['conv_in_channels'],
                                 layers_params[l]['conv_kernel_size']**2,
                                 layers_params[l]['conv_kernel_size']**2, device=device)
                # delta_l['W2'] = torch.zeros_like(data_['model'].layers_weight[l]['W'], device=device)
                
                if params['layers_params'][l]['name'] in ['conv',
                                                          'conv-no-activation',]:
                    delta_l['b'] = torch.zeros(layers_params[l]['conv_out_channels'], device=device)
                    
            elif params['layers_params'][l]['name'] in ['BN']:
                delta_l['W'] = torch.zeros(layers_params[l]['num_features'], device=device)
                delta_l['b'] = torch.zeros(layers_params[l]['num_features'], device=device)
            else:
                print('Error: unsupported layer when initial cache for ' + params['layers_params'][l]['name'])
                sys.exit()
            fbf_m.append(delta_l)

        data_['F_m'] = fbf_m
        data_['F_inv'] = numlayers * [0]

        params['kfac_inverse_update_freq'] = args['kfac_inverse_update_freq']
        params['kfac_cov_update_freq'] = args['kfac_cov_update_freq']
        params['kfac_rho'] = args['kfac_rho']
        
        if algorithm in ['MBNGD-all-to-one-LRdecay', 
                       'MBNGD-all-to-one', 
                       'MBNGD-one-to-all', 
                       'MBNGD-m1m2', 
                       'MBNGD-all-to-one-Avg']:
            get_warm_start(data_, params)
            
        
        
    elif algorithm in ['RMSprop',
                       'RMSprop-warmStart',
                       'RMSprop-test',
                       'Adam',
                       'Adam-test',
                       'Adam-noWarmStart',
                       'RMSprop-no-sqrt',
                       'RMSprop-individual-grad',
                       'RMSprop-individual-grad-no-sqrt',
                       'RMSprop-individual-grad-no-sqrt-Fisher',
                       'RMSprop-individual-grad-no-sqrt-LM']:
        
        params['RMSprop_epsilon'] = args['RMSprop_epsilon']
        
        params['RMSprop_beta_2'] = args['RMSprop_beta_2']
        
        data_['RMSprop_momentum_2'] = get_zero_torch(params)
        
        if algorithm in ['RMSprop-warmStart',
                         'Adam',
                         'Adam-test']:
        
            N1 = params['N1']
            device = params['device']
            model = data_['model']
            if N1 < params['num_train_data']:
                # i.e. stochastic setting

                i = 0 # position of training data
                j = 0 # position of mini-batch

                while i + N1 <= params['num_train_data']:



                    X_mb, t_mb = data_['dataset'].train.next_batch(N1)
                    X_mb = torch.from_numpy(X_mb).to(device)
                    t_mb = torch.from_numpy(t_mb).to(device)



                    z, a, h = model.forward(X_mb)
                    loss = get_loss_from_z(model, z, t_mb, reduction='mean') # not regularized

                    model.zero_grad()
                    loss.backward()

                    model_grad = get_model_grad(model, params)

                    if params['if_regularized_grad'] and params['tau'] != 0:


                        model_grad = get_plus_torch(
                        model_grad,
                        get_multiply_scalar_no_grad(params['tau'], model.layers_weight)
                        )
                    else:
                        1

                    i += N1
                    j += 1



                    data_['RMSprop_momentum_2'] = get_multiply_scalar(
                            (j-1)/j, data_['RMSprop_momentum_2']
                        )

                    data_['RMSprop_momentum_2'] = get_plus_torch(
                            data_['RMSprop_momentum_2'],
                            get_multiply_scalar(1/j, get_square_torch(model_grad))
                        )

    
    elif params['algorithm'] in ['shampoo-allVariables-warmStart', 
    'shampoo-allVariables-warmStart-lessInverse',
    'shampoo-allVariables-filterFlattening-warmStart',
    'shampoo-allVariables-filterFlattening-warmStart-lessInverse']:
        
        params['shampoo_if_coupled_newton'] = args['shampoo_if_coupled_newton']
        
        params['if_Hessian_action'] = args['if_Hessian_action']

        params['shampoo_inverse_freq'] = args['shampoo_inverse_freq']
    
        params['shampoo_update_freq'] = args['shampoo_update_freq']
        
        params['shampoo_decay'] = args['shampoo_decay']
        params['shampoo_weight'] = args['shampoo_weight']
        
        
        if params['algorithm'] in ['shampoo-allVariables-warmStart',
                                     'shampoo-allVariables-warmStart-lessInverse',
                                     'shampoo-allVariables-filterFlattening-warmStart',
                                     'shampoo-allVariables-filterFlattening-warmStart-lessInverse']:
            params['shampoo_epsilon'] = args['shampoo_epsilon']
        else:
            print('params[algorithm]')
            print(params['algorithm'])
            sys.exit()
        
            
        
        numlayers = params['numlayers']
        
        data_['shampoo_H'] = []
        for l in range(numlayers):
            data_['shampoo_H'].append({})
    
        data_['shampoo_H_LM_minus_2k'] = []
        for l in range(numlayers):
            data_['shampoo_H_LM_minus_2k'].append({})
    
        data_['shampoo_H_trace'] = []
        for l in range(numlayers):
            data_['shampoo_H_trace'].append({})
            
        if params['algorithm'] in ['shampoo-allVariables-warmStart',
                                   'shampoo-allVariables-warmStart-lessInverse',
                                   'shampoo-allVariables-filterFlattening-warmStart',
                                   'shampoo-allVariables-filterFlattening-warmStart-lessInverse',]:
            params['if_warm_start'] = True
        elif params['algorithm'] in ['shampoo-allVariables']:
            params['if_warm_start'] = False
        else:
            print('error: need to check for ' + params['algorithm'])
            sys.exit()
            
        
        if params['if_warm_start']:
            get_warm_start(data_, params)
            
    elif algorithm in ['Fisher-BD']:
        
        params['Fisher_BD_damping'] = args['Fisher_BD_damping']
        
        data_['block_Fisher'] = []
        for l in range(params['numlayers']):
            data_['block_Fisher'].append([])
        
        get_warm_start(data_, params)
    else:
        pass
        
    return data_, params

def train(args):
    
    print('\n')
    print('learning rate = {}'.format(args['alpha']))
    
    assert os.path.isdir(args['home_path'])
    print('args')
    print(args)
    
    
    params = {}
    params['list_algorithm_shampoo'] = ['shampoo-allVariables-filterFlattening-warmStart', 
    'shampoo-allVariables-filterFlattening-warmStart-lessInverse']   
    params['list_algorithm_kfac'] = ['kfac-correctFisher-warmStart-no-max-no-LM']
    
    torch.cuda.empty_cache()
    
    params['initialization_pkg'] = args['initialization_pkg']
    
    seed_number = args['seed_number']
    
    
    params['seed_number'] = seed_number

    np.random.seed(seed_number)
    torch.manual_seed(seed_number)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    num_threads = args['num_threads']
    params['num_threads'] = num_threads
  
    
    if num_threads == float('inf'):
        1
    else:
    
        torch.set_num_threads(num_threads)
        assert torch.get_num_threads() == num_threads

    params['home_path'] = args['home_path']
    params['if_gpu'] = args['if_gpu']
    params['if_test_mode'] = args['if_test_mode']

    if params['if_gpu'] and torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu"  
    device = torch.device(dev)  
    params['device'] = device
    

    params['algorithm'] = args['algorithm']

    
    
    
    
    matrix_name = args['matrix_name']
    params['matrix_name'] = matrix_name
    
    params['if_records'] = {}

    params['if_record_sgd_norm'] = args['if_record_sgd_norm']
    params['if_record_p_norm'] = args['if_record_p_norm']
    params['if_record_kfac_p_norm'] = args['if_record_kfac_p_norm']
    params['if_record_kfac_p_cosine'] = args['if_record_kfac_p_cosine']
    params['if_record_res_grad_norm'] = args['if_record_res_grad_norm']
    params['if_record_res_grad_random_norm'] = args['if_record_res_grad_random_norm']
    params['if_record_res_grad_grad_norm'] = args['if_record_res_grad_grad_norm']
    params['if_record_res_grad_norm_per_iter'] = args['if_record_res_grad_norm_per_iter']
    params['if_record_sgn_norm'] = args['if_record_sgn_norm']
    
    if params['if_test_mode']:
        if 'if_record_kron_bfgs_update_status' in args:
            params['if_record_kron_bfgs_update_status'] =\
            args['if_record_kron_bfgs_update_status']
        if 'if_record_kron_bfgs_matrix_norm_per_iter' in args:
            params['if_record_kron_bfgs_matrix_norm_per_iter'] =\
            args['if_record_kron_bfgs_matrix_norm_per_iter']
        if 'if_record_kfac_G_inv_norm_per_iter' in args:
            params['if_record_kfac_G_inv_norm_per_iter'] =\
            args['if_record_kfac_G_inv_norm_per_iter']
        if 'if_record_kfac_G_inv_norm_per_epoch' in args:
            params['if_record_kfac_G_inv_norm_per_epoch'] =\
            args['if_record_kfac_G_inv_norm_per_epoch']
            
        if 'if_record_kfac_G_norm_per_epoch' in args:
            params['if_records']['if_record_kfac_G_norm_per_epoch'] =\
            args['if_record_kfac_G_norm_per_epoch']
        else:
            params['if_records']['if_record_kfac_G_norm_per_epoch'] = False
            
        if 'if_record_kfac_G_twoNorm_per_epoch' in args:
            params['if_records']['if_record_kfac_G_twoNorm_per_epoch'] =\
            args['if_record_kfac_G_twoNorm_per_epoch']
        else:
            params['if_records']['if_record_kfac_G_twoNorm_per_epoch'] = False
            
        if 'if_record_kfac_A_twoNorm_per_epoch' in args:
            params['if_records']['if_record_kfac_A_twoNorm_per_epoch'] =\
            args['if_record_kfac_A_twoNorm_per_epoch']
        else:
            params['if_records']['if_record_kfac_A_twoNorm_per_epoch'] = False
            
        if 'if_record_kron_bfgs_A_twoNorm_per_epoch' in args:
            params['if_records']['if_record_kron_bfgs_A_twoNorm_per_epoch'] =\
            args['if_record_kron_bfgs_A_twoNorm_per_epoch']
        else:
            params['if_records']['if_record_kron_bfgs_A_twoNorm_per_epoch'] = False
            
        if 'if_record_kron_bfgs_G_LM_twoNorm_per_epoch' in args:
            params['if_records']['if_record_kron_bfgs_G_LM_twoNorm_per_epoch'] =\
            args['if_record_kron_bfgs_G_LM_twoNorm_per_epoch']
        else:
            params['if_records']['if_record_kron_bfgs_G_LM_twoNorm_per_epoch'] = False
            
        if 'if_record_kron_bfgs_Hg_twoNorm_per_epoch' in args:
            params['if_records']['if_record_kron_bfgs_Hg_twoNorm_per_epoch'] =\
            args['if_record_kron_bfgs_Hg_twoNorm_per_epoch']
        else:
            params['if_records']['if_record_kron_bfgs_Hg_twoNorm_per_epoch'] = False
            
        if 'if_record_kron_bfgs_Ha_twoNorm_per_epoch' in args:
            params['if_records']['if_record_kron_bfgs_Ha_twoNorm_per_epoch'] =\
            args['if_record_kron_bfgs_Ha_twoNorm_per_epoch']
        else:
            params['if_records']['if_record_kron_bfgs_Ha_twoNorm_per_epoch'] = False
            
        if 'if_record_kron_bfgs_matrix_norm' in args:
            params['if_records']['if_record_kron_bfgs_matrix_norm'] =\
            args['if_record_kron_bfgs_matrix_norm']
        else:
            params['if_records']['if_record_kron_bfgs_matrix_norm'] = False
            
        if 'if_record_layerWiseHessian_twoNorm_per_epoch' in args:
            params['if_records']['if_record_layerWiseHessian_twoNorm_per_epoch'] =\
            args['if_record_layerWiseHessian_twoNorm_per_epoch']
        else:
            params['if_records']['if_record_layerWiseHessian_twoNorm_per_epoch'] = False
            
        if 'if_record_inverseLayerWiseHessian_twoNorm_per_epoch' in args:
            params['if_records']['if_record_inverseLayerWiseHessian_twoNorm_per_epoch'] =\
            args['if_record_inverseLayerWiseHessian_twoNorm_per_epoch']
        else:
            params['if_records']['if_record_inverseLayerWiseHessian_twoNorm_per_epoch'] = False
            
        if 'if_record_inverseLayerWiseHessian_LM_twoNorm_per_epoch' in args:
            params['if_records']['if_record_inverseLayerWiseHessian_LM_twoNorm_per_epoch'] =\
            args['if_record_inverseLayerWiseHessian_LM_twoNorm_per_epoch']
        else:
            params['if_records']['if_record_inverseLayerWiseHessian_LM_twoNorm_per_epoch'] = False
            
        if 'if_record_inverseLayerWiseHessian_LM_MA_twoNorm_per_epoch' in args:
            params['if_records']['if_record_inverseLayerWiseHessian_LM_MA_twoNorm_per_epoch'] =\
            args['if_record_inverseLayerWiseHessian_LM_MA_twoNorm_per_epoch']
        else:
            params['if_records']['if_record_inverseLayerWiseHessian_LM_MA_twoNorm_per_epoch'] = False
            
        if 'if_record_kfac_F_twoNorm_per_epoch' in args:
            params['if_record_kfac_F_twoNorm_per_epoch'] =\
            args['if_record_kfac_F_twoNorm_per_epoch']
        if 'if_record_kron_bfgs_norm_s_y_per_iter' in args:
            params['if_record_kron_bfgs_norm_s_y_per_iter'] =\
            args['if_record_kron_bfgs_norm_s_y_per_iter']
        if 'if_record_kron_bfgs_sTy_per_iter' in args:
            params['if_record_kron_bfgs_sTy_per_iter'] =\
            args['if_record_kron_bfgs_sTy_per_iter']
        if 'if_record_kron_bfgs_damping_status' in args:
            params['if_record_kron_bfgs_damping_status'] =\
            args['if_record_kron_bfgs_damping_status']
        if 'if_record_kron_bfgs_check_damping' in args:
            params['if_record_kron_bfgs_check_damping'] =\
            args['if_record_kron_bfgs_check_damping']
            
    
    params['if_max_epoch'] = args['if_max_epoch']
    params['max_epoch/time'] = args['max_epoch/time']

    if_max_epoch = args['if_max_epoch'] # 0 means max_time
    if if_max_epoch:
        max_epoch = args['max_epoch/time']
    else:
        max_time = args['max_epoch/time']
        
    

    record_epoch = args['record_epoch']
    
    
    name_dataset = args['dataset']
    params['name_dataset'] = name_dataset

    params['name_loss'] = args['name_loss']
    

    params['momentum_gradient_rho'] = args['momentum_gradient_rho']
    params['momentum_gradient_dampening'] = args['momentum_gradient_dampening']

    
    # Model
    model = get_model(params)
    params['name_model'] = model.name_model
    params['layersizes'] = model.layersizes

    print('name_loss:')
    print(model.name_loss)
    print('Model created.')


    params['layers_params'] = model.layers_params
    
    params['N1'] = args['N1']
    params['N2'] = args['N2']


    data_ = {}


    if name_dataset in ['CIFAR-10-NoAugmentation-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout',
                        'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout',
                        'CIFAR-10-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout',
                        'CIFAR-10-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout-no-regularization',
                        'CIFAR-10-onTheFly-N1-256-vgg16-NoAdaptiveAvgPool',
                        'CIFAR-10-onTheFly-N1-512-vgg16-NoAdaptiveAvgPoolNoDropout',
                        'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN',
                        'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN-NoBias',
                        'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine',
                        'CIFAR-10-onTheFly-ResNet32-BNNoAffine',
                        'CIFAR-10-onTheFly-ResNet32-BN',
                        'CIFAR-10-onTheFly-ResNet32-BN-BNshortcut',
                        'CIFAR-10-onTheFly-ResNet32-BN-BNshortcutDownsampleOnly',
                        'CIFAR-10-onTheFly-ResNet32-BN-BNshortcutDownsampleOnly-NoBias',
                        'CIFAR-10-onTheFly-N1-128-ResNet32-BNNoAffine-PaddingShortcutDownsampleOnly-NoBias-no-regularization',
                        'CIFAR-10-onTheFly-N1-128-ResNet32-BN-BNshortcutDownsampleOnly-NoBias',
                        'CIFAR-10-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias',
                        'CIFAR-10-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias-no-regularization',
                        'CIFAR-100-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias',
                        'CIFAR-100-onTheFly-ResNet34-BNNoAffine',
                        'CIFAR-100-onTheFly-ResNet34-BN',
                        'CIFAR-100-onTheFly-ResNet34-BN-BNshortcut',
                        'CIFAR-100-onTheFly-ResNet34-BN-BNshortcutDownsampleOnly',
                        'CIFAR-100-onTheFly-ResNet34-BN-BNshortcutDownsampleOnly-NoBias',
                        'CIFAR-100-onTheFly-N1-128-ResNet34-BN-BNshortcutDownsampleOnly-NoBias',
                        'CIFAR-100-onTheFly-N1-128-ResNet34-BN-PaddingShortcutDownsampleOnly-NoBias',
                        'CIFAR-10-AllCNNC',
                        'CIFAR-10-N1-128-AllCNNC',
                        'CIFAR-10-N1-512-AllCNNC',
                        'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout',
                        'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine-no-regularization',
                        'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN',
                        'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN-no-regularization',
                        'CIFAR-100-onTheFly-vgg16-NoLinear-BN-no-regularization',
                        'CIFAR-100-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout',
                        'CIFAR-100-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout-no-regularization',
                        'CIFAR-100-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine',
                        'CIFAR-100-onTheFly-AllCNNC', 'SVHN-ResNet34', 'SVHN-vgg11']:
        params['if_dataset_onTheFly'] = True
    elif name_dataset in ['Fashion-MNIST-N1-60-no-regularization',
                          'Fashion-MNIST-N1-256-no-regularization',
                          'CIFAR-100',
                          'CIFAR-100-NoAugmentation',
                          'CIFAR-100-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout',
                          'CIFAR-100-NoAugmentation-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout',
                          'CIFAR-10-NoAugmentation-vgg11',
                          'CIFAR-10-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout',
                          'CIFAR-10-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout-BN',
                          'CIFAR-10-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine',
                          'DownScaledMNIST-N1-1000-no-regularization',
                          'MNIST-autoencoder-relu-N1-1000-sum-loss',
                          'MNIST-autoencoder-relu-N1-1000-sum-loss-no-regularization',
                          'MNIST-autoencoder-relu-N1-100-sum-loss',
                          'CURVES-autoencoder-relu-sum-loss',
                          'CURVES-autoencoder-relu-sum-loss-no-regularization',
                          'CURVES-autoencoder-relu-N1-100-sum-loss',
                          'FacesMartens-autoencoder-relu',
                          'FacesMartens-autoencoder-relu-no-regularization',
                          'FacesMartens-autoencoder-relu-N1-100',]:
        params['if_dataset_onTheFly'] = False
    else:
        print('error: need to check if on the fly for ' + name_dataset)
        sys.exit()


    if not params['if_dataset_onTheFly']:
        
        dataset = read_data_sets(name_dataset, params['name_model'], params['home_path'], one_hot=False)
        X_train = dataset.train.images
        t_train = dataset.train.labels

        print('For X_train:')
        get_statistics(X_train)

        

        print('X_train.shape')
        print(X_train.shape)
        print('t_train.shape')
        print(t_train.shape)

        




        params['num_train_data'] = len(t_train)
        
        X_test = dataset.test.images
        t_test = dataset.test.labels

    
    if params['if_dataset_onTheFly']:
        dataset = read_data_sets_v2(name_dataset, params)
        params['num_train_data'] = dataset.num_train_data
    data_['dataset'] = dataset
    params['alpha'] = args['alpha']
    params['alpha_current'] = params['alpha']
    
    
    params['numlayers'] = model.numlayers
    data_['model'] = model

    params = get_params(params, args)
    params['tau'] = args['tau']
    data_, params = train_initialization(data_, params, args)
 

    if params['if_momentum_gradient']:
        data_['model_grad_momentum'] = get_zero_torch(params)
        data_['model_grad_beta1'] = get_zero_torch(params)
    
    if params['if_Adam']:
        
        print('error: should not reach here')
        sys.exit()

        params['Adam_beta_1'] = 0.9
        params['Adam_beta_2'] = 0.999
        params['Adam_epsilon'] = 10**(-8)

        data_['model_grad_Adam_momentum_1'] = get_zero(params)
        
        data_['model_grad_Adam_momentum_2'] = get_zero(params)


    if params['if_yura']:
        params['yura_lambda_0'] = 1

    if params['if_momentum_p']:
        data_['p_momentum_torch'] = get_zero_torch(params)

    if params['if_VA_p'] or\
    params['if_signVAsqrt'] or\
    params['if_signVA'] or\
    params['if_signVAerf']:
        data_['p_momentum_1_torch'] = get_zero_torch(params)
        data_['p_momentum_2_torch'] = get_zero_torch(params)

    if params['if_LM']:
            
        boost = 1.01
        drop = 1 / 1.01
        params['boost'] = boost
        params['drop'] = drop

    epochs = [0]
    timesCPU = [0]
    timesWallClock = [0]
    
    if not params['if_dataset_onTheFly']:
        train_losses = []
        train_unregularized_losses = []
        train_acces = []
    
    train_unregularized_minibatch_losses = []
    train_minibatch_acces = []

    test_acces = []
    test_losses = []
    reduction = 'mean'
    

    if params['if_dataset_onTheFly']:
            test_loss_0, _, test_acc_0 = get_regularized_loss_and_acc_from_x_whole_dataset_with_generator(
                model, dataset.test_generator, reduction, params
            )
    else:
        test_loss_0, _, test_acc_0 = get_regularized_loss_and_acc_from_x_whole_dataset(
            model, X_test, t_test, reduction, params
        )
    
    test_losses.append(test_loss_0)
    test_acces.append(test_acc_0)
    
    
    print('test_loss_0, test_acc_0')
    print(test_loss_0, test_acc_0)

    if params['if_LM']:
        lambdas = []
        lambdas.append(params['lambda_'])
    if params['if_yura']:
        yura_lambdas = []
        yura_lambdas.append(params['yura_lambda_0'])
    
    if params['if_test_mode']:
        if params['if_record_sgd_norm']:
            sgd_norms = []
        if params['if_record_p_norm']:
            p_norms = []
        if params['if_record_kfac_p_norm']:
            kfac_p_norms = []
            data_['kfac_p_norms'] = kfac_p_norms
        if params['if_record_kfac_p_cosine']:
            kfac_p_cosines = []
            data_['kfac_p_cosines'] = kfac_p_cosines
        if params['if_record_res_grad_norm']:
            res_grad_norms = []
            data_['res_grad_norms'] = res_grad_norms
        if params['if_record_res_grad_random_norm']:
            res_grad_random_norms = []
            data_['res_grad_random_norms'] = res_grad_random_norms
        if params['if_record_res_grad_grad_norm']:
            res_grad_grad_norms = []
            data_['res_grad_grad_norms'] = res_grad_grad_norms
        if params['if_record_res_grad_norm_per_iter']:
            res_grad_norms_per_iter = []
            data_['res_grad_norms_per_iter'] = res_grad_norms_per_iter
        if 'if_record_kron_bfgs_update_status' in params and\
        params['if_record_kron_bfgs_update_status']:
            data_['kron_bfgs_update_status'] = []
        if 'if_record_kron_bfgs_matrix_norm_per_iter' in params and\
        params['if_record_kron_bfgs_matrix_norm_per_iter']:
            data_['kron_bfgs_matrix_norms_per_iter'] = []

        if 'if_record_kfac_G_inv_norm_per_iter' in params and\
            params['if_record_kfac_G_inv_norm_per_iter'] == True:
            data_['kfac_G_inv_norms_per_iter'] = []
        if 'if_record_kfac_G_inv_norm_per_epoch' in params and\
            params['if_record_kfac_G_inv_norm_per_epoch'] == True:
            data_['kfac_G_inv_norms_per_epoch'] = []
        if params['if_records']['if_record_kfac_G_norm_per_epoch']:
            data_['kfac_G_norms_per_epoch'] = []
        if params['if_records']['if_record_kfac_G_twoNorm_per_epoch']:
            data_['kfac_G_twoNorms_per_epoch'] = []
        if params['if_records']['if_record_kfac_A_twoNorm_per_epoch']:
            data_['kfac_A_twoNorms_per_epoch'] = []
        if params['if_records']['if_record_kron_bfgs_A_twoNorm_per_epoch']:
            data_['kron_bfgs_A_twoNorms_per_epoch'] = []
        if params['if_records']['if_record_kron_bfgs_G_LM_twoNorm_per_epoch']:
            data_['kron_bfgs_G_LM_twoNorms_per_epoch'] = []
        if params['if_records']['if_record_kron_bfgs_Hg_twoNorm_per_epoch']:
            data_['kron_bfgs_Hg_twoNorms_per_epoch'] = []
        if params['if_records']['if_record_kron_bfgs_Ha_twoNorm_per_epoch']:
            data_['kron_bfgs_Ha_twoNorms_per_epoch'] = []
        if params['if_records']['if_record_layerWiseHessian_twoNorm_per_epoch'] == True:
            data_['layerWiseHessian_twoNorms_per_epoch'] = []
        if params['if_records']['if_record_inverseLayerWiseHessian_twoNorm_per_epoch'] == True:
            data_['inverseLayerWiseHessian_twoNorms_per_epoch'] = []
        if params['if_records']['if_record_inverseLayerWiseHessian_LM_twoNorm_per_epoch'] == True:
            data_['inverseLayerWiseHessian_LM_twoNorms_per_epoch'] = []
        if params['if_records']['if_record_inverseLayerWiseHessian_LM_MA_twoNorm_per_epoch'] == True:
            data_['inverseLayerWiseHessian_LM_MA_twoNorms_per_epoch'] = []
            
        if 'if_record_kfac_F_twoNorm_per_epoch' in params and\
            params['if_record_kfac_F_twoNorm_per_epoch'] == True:
            data_['kfac_F_twoNorms_per_epoch'] = []
        if 'if_record_kron_bfgs_norm_s_y_per_iter' in params and\
            params['if_record_kron_bfgs_norm_s_y_per_iter'] == True:
            data_['kron_bfgs_norms_s_y_per_iter'] = {}
            data_['kron_bfgs_norms_s_y_per_iter']['s'] = []
            data_['kron_bfgs_norms_s_y_per_iter']['y'] = []
        if 'if_record_kron_bfgs_sTy_per_iter' in params and\
            params['if_record_kron_bfgs_sTy_per_iter'] == True:
            data_['kron_bfgs_sTy_per_iter'] = []
        if 'if_record_kron_bfgs_damping_status' in params and\
            params['if_record_kron_bfgs_damping_status'] == True:
            data_['kron_bfgs_damping_statuses'] = {}
        if 'if_record_kron_bfgs_check_damping' in params and\
            params['if_record_kron_bfgs_check_damping'] == True:
            data_['kron_bfgs_check_dampings'] = []
            
        if params['if_records']['if_record_kron_bfgs_matrix_norm'] == True:
            data_['kron_bfgs_matrix_norms'] = []

    if params['if_record_sgn_norm']:
        sgn_norms = []
        data_['model_grad_full'] = get_full_grad(model, X_train, t_train, params)
        
    print('params[if_dataset_onTheFly]')
    print(params['if_dataset_onTheFly'])
    
    if not params['if_dataset_onTheFly']:
        
        reduction = 'mean'
        loss_0, unregularized_loss_0, acc_0 = get_regularized_loss_and_acc_from_x_whole_dataset(model, X_train, t_train, reduction, params)
        train_losses.append(loss_0)
        train_unregularized_losses.append(unregularized_loss_0)
        train_acces.append(acc_0)
    
        print('loss_0, unregularized_loss_0, acc_0')
        print(loss_0, unregularized_loss_0, acc_0)


    N1 = params['N1']
    iter_per_epoch = int(params['num_train_data'] / N1)
    
    params['iter_per_epoch'] = iter_per_epoch
    iter_per_record = int(np.floor(params['num_train_data'] * record_epoch / N1))

    # Training
    print('Begin training...')
    epoch = -1
    i = -1
 
    while not get_if_stop(args, i+1, iter_per_epoch, timesCPU):
        i += 1
        params['i'] = i

        if i % iter_per_record == 0:
            start_time_wall_clock = time.time()
            start_time_cpu = time.process_time()
            epoch += 1
            params['epoch'] = epoch

        # get minibatch
        X_mb, t_mb = dataset.train.next_batch(N1) 
        if not params['if_dataset_onTheFly']:
            X_mb = torch.from_numpy(X_mb)
        X_mb = X_mb.to(device)
        if not params['if_dataset_onTheFly']:
            t_mb = torch.from_numpy(t_mb)
        t_mb = t_mb.to(device)
        # Forward
        z, a, h = model.forward(X_mb)
  
        
        reduction = 'mean'
        loss = get_loss_from_z(
            model, z, t_mb, reduction)
    
        params['unregularized_minibatch_loss_i_no_MA'] = loss.item()
        
        if i == 0:
            
            unregularized_minibatch_loss_i = loss.item()
        
            train_unregularized_minibatch_losses.append(
            unregularized_minibatch_loss_i)
            
            minibatch_acc_i = get_acc_from_z(model, params, z, t_mb)
            
            train_minibatch_acces.append(minibatch_acc_i)
            
        else:
            
            minibatch_acc_i =\
            0.9 * minibatch_acc_i + 0.1 * get_acc_from_z(model, params, z, t_mb)
            
            unregularized_minibatch_loss_i =\
            0.9 * unregularized_minibatch_loss_i + 0.1 * loss.item()
 
        model.zero_grad()
        
        if params['if_second_order_algorithm'] and params['matrix_name'] == 'Fisher-correct':
            loss.backward(retain_graph=True)
        else:
            loss.backward()
        
        
        model_grad_torch = get_model_grad(model, params)
        
        data_['model_grad_torch_unregularized'] = model_grad_torch
        
        model_grad_torch =\
        from_unregularized_grad_to_regularized_grad(model_grad_torch, data_, params)
    

        data_['model_grad_torch'] = model_grad_torch # regularized


        if get_if_nan(model_grad_torch):
            print('Error: nan in model_grad_torch')
            for l in range(len(model_grad_torch)):
                for key in model_grad_torch[l]:
                    print('torch.max(model_grad_torch[l][key])')
                    print(torch.max(model_grad_torch[l][key]))
                    print('torch.min(model_grad_torch[l][key])')
                    print(torch.min(model_grad_torch[l][key]))
            for l in range(len(model.layers_weight)):
                for key in model.layers_weight[l]:
                    print('torch.max(model.layers_weight[l][key])')
                    print(torch.max(model.layers_weight[l][key]))
                    print('torch.min(model.layers_weight[l][key])')
                    print(torch.min(model.layers_weight[l][key]))

            break

        if params['if_test_mode']:
            if params['if_record_sgd_norm']:
                sgd_norms.append(
                    np.sqrt(get_dot_product_torch(model_regularized_grad_torch, model_regularized_grad_torch).item()))
        
        if params['if_record_sgn_norm']:
            sgn_ = get_subtract_torch(model_regularized_grad_torch, data_['model_grad_full'])
            sgn_norms.append(np.sqrt(get_dot_product_torch(sgn_, sgn_).item()))
            
        if params['if_momentum_gradient']:
            rho = params['momentum_gradient_rho']
            dampening = params['momentum_gradient_dampening']
            
            data_['model_grad_momentum'] = get_plus_torch(get_multiply_scalar(rho, data_['model_grad_momentum']), get_multiply_scalar(1 - dampening, model_grad_torch))
            data_['model_grad_beta1'] = get_plus_torch(get_multiply_scalar(rho, data_['model_grad_beta1']), get_multiply_scalar(1 - rho, model_grad_torch)) # regularized

        if params['if_momentum_gradient']:
            data_['model_grad_used_torch'] = data_['model_grad_momentum']
        else:
            data_['model_grad_used_torch'] = model_grad_torch



        # get second order caches
        if params['if_second_order_algorithm']:
            data_['X_mb'] = X_mb
            data_['t_mb'] = t_mb
            
            if (args['algorithm'] in ['MBNGD-all-to-one-LRdecay', 
                             'MBNGD-all-to-one', 
                             'MBNGD-all-to-one-Avg-LRdecay',
                             'MBNGD-all-to-one-Avg', 
                             'L-MBNGD-all-to-one', 
                             'L-MBNGD-all-to-one-LRdecay']) and (params['i'] % params['kfac_cov_update_freq'] == 0):
                data_ = get_second_order_caches(z, a, h, data_, params)
            elif (params['i'] % params['kfac_cov_update_freq'] != 0):
                1
            else:
                data_ = get_second_order_caches(z, a, h, data_, params)
        
        model = data_['model']

        if params['if_LM']:
            data_['regularized_loss'] = loss
            data_['t_mb_N1'] = t_mb
            lambda_minus_tau = params['lambda_']
            params['lambda_'] = params['lambda_'] + params['tau']

        
        if params['if_lr_decay']:
            params['alpha_current'] =\
            params['alpha'] *\
            (params['lr_decay_rate'] ** (params['epoch'] // params['num_epoch_to_decay']))
            
            

        algorithm = params['algorithm']

        if algorithm in ['ekfac-EF-VA',
                         'ekfac-EF',
                         'kfac-TR',
                         'kfac-momentum-grad-TR',
                         'kfac-CG',
                         'kfac-momentum-grad-CG',
                         'kfac',
                         'kfac-no-max',
                         'kfac-NoMaxNoSqrt',
                         'kfac-NoMaxNoSqrt-no-LM',
                         'kfac-no-max-no-LM',
                         'kfac-warmStart-no-max-no-LM',
                         'kfac-warmStart-lessInverse-no-max-no-LM',
                         'kfac-warmStart-lessInverse-NoMaxNoSqrt-no-LM',
                         'kfac-correctFisher-warmStart-no-max-no-LM',
                         'kfac-correctFisher-warmStart-NoMaxNoSqrt-no-LM',
                         'kfac-correctFisher-warmStart-lessInverse-no-max-no-LM',
                         'kfac-correctFisher-warmStart-lessInverse-NoMaxNoSqrt-no-LM',
                         'kfac-no-max-epsilon-A-G-no-LM',
                         'kfac-EF',
                         'Fisher-block']:    
            data_, params = kfac_update(data_, params)
            
            if params['kfac_svd_failed']:
                
                print('i')
                print(i)
                
                print('error: kfac_svd_failed')
                
                break
                
        elif algorithm in ['MBNGD-all-to-one-LRdecay', 
                           'MBNGD-all-to-one', 
                           'MBNGD-one-to-all', 
                           'MBNGD-m1m2', 
                          'MBNGD-all-to-one-Avg', 
                          'L-MBNGD-all-to-one', 
                          'L-MBNGD-all-to-one-LRdecay']:
            
            # start_up = time.time()
            data_, params = mbngd_update(data_, params)
            # print(time.time()-start_up)

        elif algorithm in ['shampoo',
                           'shampoo-allVariables',
                           'shampoo-allVariables-warmStart',
                           'shampoo-allVariables-warmStart-lessInverse',
                           'shampoo-allVariables-filterFlattening-warmStart',
                           'shampoo-allVariables-filterFlattening-warmStart-lessInverse']:
            data_, params = shampoo_update(data_, params)
        elif algorithm in ['Fisher-BD']:
            data_, params = Fisher_BD_update(data_, params)
        elif algorithm in ['SGD-VA',
                           'SGD-signVAsqrt',
                           'SGD-signVAerf',
                           'SGD-signVA',
                           'SGD-yura-BD',
                           'SGD-yura-old',
                           'SGD-yura',
                           'SGD-yura-MA',
                           'SGD-sign',
                           'SGD-momentum-yura',
                           'SGD-momentum',
                           'SGD',]:
            data_ = SGD_update(data_, params)
        elif algorithm in ['RMSprop',
                           'RMSprop-warmStart',
                           'RMSprop-test',
                           'Adam',
                           'Adam-test',
                           'Adam-noWarmStart',
                           'RMSprop-no-sqrt',
                           'RMSprop-individual-grad',
                           'RMSprop-individual-grad-no-sqrt',
                           'RMSprop-individual-grad-no-sqrt-Fisher',
                           'RMSprop-individual-grad-no-sqrt-LM']:
            data_ = RMSprop_update(data_, params)
        elif algorithm == 'GI-Fisher':
            data_, params = GI_Fisher_update(data_, params)
        else:
            print('Error: updating direction not defined for ' + algorithm)
            sys.exit()

        if params['if_LM']:
            params['lambda_'] = lambda_minus_tau

        p_torch = data_['p_torch']
        
        if params['if_LM']:
            lambda_ = update_lambda(p_torch, data_, params)
            params['lambda_'] = lambda_


        if params['if_momentum_p']:
            rho_momentum_p = 0.9
            data_['p_momentum_torch'] = get_plus_torch(\
                                           get_multiply_scalar(rho_momentum_p, data_['p_momentum_torch']),
                                           get_multiply_scalar(1 - rho_momentum_p, p_torch))
            # p = data_['p_momentum']
            p_torch = data_['p_momentum_torch']

        if params['if_VA_p']:
            rho_momentum_p = 0.9
            data_['p_momentum_1'] = get_plus(\
                                           get_multiply_scalar(rho_momentum_p, data_['p_momentum_1']),
                                           get_multiply_scalar(1 - rho_momentum_p, p))
            data_['p_momentum_2'] = get_plus(\
                                           get_multiply_scalar(rho_momentum_p, data_['p_momentum_2']),
                                           get_multiply_scalar(1 - rho_momentum_p, get_square(p)))
            p = get_divide(\
                           get_multiply(data_['p_momentum_1'], get_square(data_['p_momentum_1'])),
                           get_plus_scalar(10**(-8), data_['p_momentum_2']))
            
        
        if params['if_sign']:
            p = get_sign(p)
         
        model = update_parameter(p_torch, model, params)
 
        if get_if_nan(model.layers_weight):
            print('Error: nan in model.layers_weight')
            break

        if (i+1) % iter_per_record == 0:
        
            if params['if_test_mode']:
                if 'if_record_kfac_G_inv_norm_per_epoch' in params and\
                params['if_record_kfac_G_inv_norm_per_epoch']:
                    
                    data_['kfac_G_inv_norms_per_epoch'].append(
                        [torch.norm(G_inv_l).item() for G_inv_l in data_['G_inv']]
                    )
                    # torch.norm() is Fro-norm here
                    
            if params['if_test_mode']:
                if params['if_records']['if_record_kfac_G_norm_per_epoch']:
                    
                    # data_['G'] is without LM
                    data_['kfac_G_norms_per_epoch'].append(
                        [torch.norm(G_l).item() for G_l in data_['G']]
                    )
                    
                    
            if params['if_test_mode']:
                if params['if_records']['if_record_kfac_G_twoNorm_per_epoch']:
                    
                    # data_['G'] is without LM
                    data_['kfac_G_twoNorms_per_epoch'].append(
                        [np.linalg.norm(G_l.cpu().data.numpy(), ord=2) for G_l in data_['G']]
                    )
                    
            if params['if_test_mode']:
                if params['if_records']['if_record_kfac_A_twoNorm_per_epoch']:
                    
                    # data_['A'] is without LM
                    data_['kfac_A_twoNorms_per_epoch'].append(
                        [np.linalg.norm(A_l.cpu().data.numpy(), ord=2) for A_l in data_['A']]
                    )
                    
                    
            if params['if_test_mode']:
                if params['if_records']['if_record_kron_bfgs_A_twoNorm_per_epoch']:
                   
                    
                    # without LM
                    data_['kron_bfgs_A_twoNorms_per_epoch'].append(
                        [
                            np.linalg.norm(
                                Kron_BFGS_matrices_l['A'].cpu().data.numpy(), ord=2
                            ) for Kron_BFGS_matrices_l in data_['Kron_BFGS_matrices']
                        ]
                    )
                    
            
            if params['if_test_mode'] and\
        (params['if_records']['if_record_layerWiseHessian_twoNorm_per_epoch'] or\
                params['if_records']['if_record_inverseLayerWiseHessian_twoNorm_per_epoch'] or\
        params['if_records']['if_record_inverseLayerWiseHessian_LM_twoNorm_per_epoch'] or\
        params['if_records']['if_record_inverseLayerWiseHessian_LM_MA_twoNorm_per_epoch']):
                from utils_git.utils_hessian import compute_hessian

                true_layer_wise_hessian = compute_hessian(X_mb, t_mb, data_, params)
                
                if params['if_records']['if_record_inverseLayerWiseHessian_LM_MA_twoNorm_per_epoch']:
                    
                    print('torch.norm(X_mb)')
                    print(torch.norm(X_mb))
                    
                    if len(data_['inverseLayerWiseHessian_LM_MA_twoNorms_per_epoch']) == 0:
                        true_layer_wise_hessian_MA = true_layer_wise_hessian
                    else:

                        assert len(true_layer_wise_hessian_MA) == len(true_layer_wise_hessian)
                        
                        for l in range(len(true_layer_wise_hessian_MA)):
                            true_layer_wise_hessian_MA[l] =\
                        0.9 * true_layer_wise_hessian_MA[l] +\
                        0.1 * true_layer_wise_hessian[l]
                    
                        
                    data_['inverseLayerWiseHessian_LM_MA_twoNorms_per_epoch'].append([])
                        
                    for B_l in true_layer_wise_hessian_MA:
                        
                        lambda_hessian_LM =\
                        params['Kron_BFGS_A_LM_epsilon'] * params['Kron_BFGS_H_epsilon']
                        
                        B_l_LM = B_l + lambda_hessian_LM * np.eye(B_l.shape[0])
                        
                        data_['inverseLayerWiseHessian_LM_MA_twoNorms_per_epoch'][-1].append(
                        np.linalg.norm(np.linalg.inv(B_l_LM), ord=2)
                        )
                    
        
                

                if params['if_records']['if_record_layerWiseHessian_twoNorm_per_epoch']:
                    data_['layerWiseHessian_twoNorms_per_epoch'].append(
                    [np.linalg.norm(B_l, ord=2) for B_l in true_layer_wise_hessian]
                )
                
                if params['if_records']['if_record_inverseLayerWiseHessian_twoNorm_per_epoch']:
                    data_['inverseLayerWiseHessian_twoNorms_per_epoch'].append(
                    [np.linalg.norm(np.linalg.inv(B_l), ord=2) for B_l in true_layer_wise_hessian]
                )
                    
                if params['if_records']['if_record_inverseLayerWiseHessian_LM_twoNorm_per_epoch']:
                    data_['inverseLayerWiseHessian_LM_twoNorms_per_epoch'].append([])
                    
                    for B_l in true_layer_wise_hessian:
                        
                        lambda_hessian_LM =\
                        params['Kron_BFGS_A_LM_epsilon'] * params['Kron_BFGS_H_epsilon']
                        
                        B_l_LM = B_l + lambda_hessian_LM * np.eye(B_l.shape[0])
                        
                        data_['inverseLayerWiseHessian_LM_twoNorms_per_epoch'][-1].append(
                        np.linalg.norm(np.linalg.inv(B_l_LM), ord=2)
                        )

            
            

            import datetime
            import pytz
            my_date = datetime.datetime.now(pytz.timezone('US/Eastern'))
            my_date = my_date.strftime("%d/%m/%Y %H:%M:%S")
            print("date and time =", my_date)

            timesCPU_i = time.process_time() - start_time_cpu
            timesWallClock_i = time.time() - start_time_wall_clock
            
            


            
            if not params['if_dataset_onTheFly']:
                
                reduction = 'mean'
                loss_i, unregularized_loss_i, acc_i = get_regularized_loss_and_acc_from_x_whole_dataset(model, X_train, t_train, reduction, params)
            

            
            
            if params['if_dataset_onTheFly']:    
                if math.isnan(unregularized_minibatch_loss_i):
                    print('Warning: unregularized_minibatch_loss_i is NAN.')
                    break
            else:
                
                if math.isnan(loss_i):
                    print('Warning: loss_i is NAN.')
                    break

            timesCPU.append(timesCPU_i)
            timesWallClock.append(timesWallClock_i)
            if epoch > 0:
                timesCPU[-1] = timesCPU[-1] + timesCPU[-2]
                timesWallClock[-1] = timesWallClock[-1] + timesWallClock[-2]
            
            if not params['if_dataset_onTheFly']:
                train_losses.append(loss_i)
                train_unregularized_losses.append(unregularized_loss_i)
                train_acces.append(acc_i)
            
            train_unregularized_minibatch_losses.append(
                unregularized_minibatch_loss_i
            )
            train_minibatch_acces.append(minibatch_acc_i)
            
            

            reduction = 'mean'
            
            if params['if_dataset_onTheFly']:
                test_loss_i, test_unregularized_loss_i, test_acc_i =\
                get_regularized_loss_and_acc_from_x_whole_dataset_with_generator(model, dataset.test_generator, reduction, params)
            else:
                test_loss_i, test_unregularized_loss_i, test_acc_i =\
                get_regularized_loss_and_acc_from_x_whole_dataset(model, X_test, t_test, reduction, params)
            
            test_losses.append(test_loss_i)
            test_acces.append(test_acc_i)
            
            if params['if_LM']:
                lambdas.append(params['lambda_'])
                print('lambda = ', lambdas[-1])
            if params['if_yura']:
                yura_lambdas.append(params['yura_lambda'])
                print('yura-lambda = ', yura_lambdas[-1])
            epochs.append((epoch + 1) * record_epoch)
            
            print('Current learning rate: {0:.5f}'.format(params['alpha_current']))
            
            print('Iter-{0:.3f}'.format(epochs[-1]))
            
            if not params['if_dataset_onTheFly']:
                print('Training loss: {0:.3f}'.format(train_losses[-1]))
                print('Training unregularized loss: {0:.3f}'.format(train_unregularized_losses[-1]))
                print('Training accuracy: {0:.3f}'.format(train_acces[-1]))

            
            print('Training unregularized minibatch loss: {0:.3f}'.format(train_unregularized_minibatch_losses[-1]))
            print('Training minibatch acc: {0:.3f}'.format(train_minibatch_acces[-1]))
            
            

            print('Testing unregularized loss: {0:.3f}'.format(test_unregularized_loss_i))
            print('Testing accuracy: {0:.3f}'.format(test_acces[-1]))
            

            



            if epoch > 0:
                print('elapsed cpu time: ', timesCPU[-1] - timesCPU[-2])
                print('elapsed wall-clock time: ', timesWallClock[-1] - timesWallClock[-2])
            else:
                print('elapsed cpu time: ', timesCPU[-1])
                print('elapsed wall-clock time: ', timesWallClock[-1])

            

            import gc
            gc.collect()

            torch.cuda.empty_cache()

            
            values_virtual_memory = psutil.virtual_memory()

            print('total (GB): {}, available (GB): {}, percent (%): {}'.format(
                values_virtual_memory.total >> 30, values_virtual_memory.available >> 30,\
                values_virtual_memory.percent
            ))
            

            if params['device'] == 'cuda:0':
                
                print_gpu_usage(params)
                      
            print('\n')



    print('Begin saving results...')
    
    params['algorithm'] = params['true_algorithm']

    

    name_algorithm_with_params = get_name_algorithm_with_params(params)

    name_result = name_dataset + '/' + name_algorithm_with_params + '/'



    epochs = np.asarray(epochs)
    timesCPU = np.asarray(timesCPU)
    timesWallClock = np.asarray(timesWallClock)
    
    if not params['if_dataset_onTheFly']:
        train_losses = np.asarray(train_losses)
        train_unregularized_losses = np.asarray(train_unregularized_losses)
        train_acces = np.asarray(train_acces)
    
    train_unregularized_minibatch_losses = np.asarray(train_unregularized_minibatch_losses)
    train_minibatch_acces = np.asarray(train_minibatch_acces)
    
    
    test_losses = np.asarray(test_losses)
    test_acces = np.asarray(test_acces)
    dict_result = {'train_unregularized_minibatch_losses': train_unregularized_minibatch_losses,
                   'train_minibatch_acces': train_minibatch_acces,
                   'test_losses': test_losses,
                   'test_acces': test_acces,
                   'timesCPU': timesCPU,
                   'timesWallClock': timesWallClock,
                   'epochs': epochs}
    if not params['if_dataset_onTheFly']:
        dict_result.update(
        {'train_losses': train_losses,
         'train_unregularized_losses': train_unregularized_losses,
         'train_acces': train_acces}
    )
    if params['if_LM']:
        lambdas = np.asarray(lambdas)
        dict_result['lambdas'] = lambdas
    if params['if_yura']:
        yura_lambdas = np.asarray(yura_lambdas)
        dict_result['yura_lambdas'] = yura_lambdas
        

    

    if params['if_test_mode']:
        if params['if_record_sgd_norm']:
            sgd_norms = np.asarray(sgd_norms)
            dict_result['sgd_norms'] = sgd_norms
        if params['if_record_p_norm']:
            p_norms = np.asarray(p_norms)
            dict_result['p_norms'] = p_norms
        if params['if_record_kfac_p_norm']:
            kfac_p_norms = data_['kfac_p_norms']
            kfac_p_norms = np.asarray(kfac_p_norms)
            dict_result['kfac_p_norms'] = kfac_p_norms
        if params['if_record_kfac_p_cosine']:
            kfac_p_cosines = data_['kfac_p_cosines']
            kfac_p_cosines = np.asarray(kfac_p_cosines)
            dict_result['kfac_p_cosines'] = kfac_p_cosines
        if params['if_record_res_grad_norm']:
            res_grad_norms = data_['res_grad_norms']
            res_grad_norms = np.asarray(res_grad_norms)
            dict_result['res_grad_norms'] = res_grad_norms
        if params['if_record_res_grad_random_norm']:
            res_grad_random_norms = data_['res_grad_random_norms']
            res_grad_random_norms = np.asarray(res_grad_random_norms)
            dict_result['res_grad_random_norms'] = res_grad_random_norms
        if params['if_record_res_grad_grad_norm']:
            res_grad_grad_norms = data_['res_grad_grad_norms']
            res_grad_grad_norms = np.asarray(res_grad_grad_norms)
            dict_result['res_grad_grad_norms'] = res_grad_grad_norms
        if params['if_record_res_grad_norm_per_iter']:
            res_grad_norms_per_iter = data_['res_grad_norms_per_iter']
            res_grad_norms_per_iter = np.asarray(res_grad_norms_per_iter)
            dict_result['res_grad_norms_per_iter'] = res_grad_norms_per_iter
        if 'if_record_kron_bfgs_matrix_norm_per_iter' in params and\
            params['if_record_kron_bfgs_matrix_norm_per_iter'] == True:
            dict_result['kron_bfgs_matrix_norms_per_iter'] = data_['kron_bfgs_matrix_norms_per_iter']
            
        if 'if_record_kron_bfgs_damping_status' in params and\
            params['if_record_kron_bfgs_damping_status'] == True:
            dict_result['kron_bfgs_damping_statuses'] = data_['kron_bfgs_damping_statuses']
            
        if 'if_record_kron_bfgs_check_damping' in params and\
            params['if_record_kron_bfgs_check_damping'] == True:
            dict_result['kron_bfgs_check_dampings'] = data_['kron_bfgs_check_dampings']
        
        
            
        if 'if_record_kron_bfgs_update_status' in params and\
            params['if_record_kron_bfgs_update_status'] == True:
            dict_result['kron_bfgs_update_status'] = data_['kron_bfgs_update_status']
            
        if 'if_record_kfac_G_inv_norm_per_iter' in params and\
        params['if_record_kfac_G_inv_norm_per_iter']:
            dict_result['kfac_G_inv_norms_per_iter'] = data_['kfac_G_inv_norms_per_iter']
            
        if 'if_record_kfac_G_inv_norm_per_epoch' in params and\
        params['if_record_kfac_G_inv_norm_per_epoch']:
            dict_result['kfac_G_inv_norms_per_epoch'] = data_['kfac_G_inv_norms_per_epoch']
            

        if params['if_records']['if_record_kfac_G_norm_per_epoch']:
            dict_result['kfac_G_norms_per_epoch'] = data_['kfac_G_norms_per_epoch']
        if params['if_records']['if_record_kfac_G_twoNorm_per_epoch']:
            dict_result['kfac_G_twoNorms_per_epoch'] = data_['kfac_G_twoNorms_per_epoch']
        if params['if_records']['if_record_kfac_A_twoNorm_per_epoch']:
            dict_result['kfac_A_twoNorms_per_epoch'] = data_['kfac_A_twoNorms_per_epoch']
        if params['if_records']['if_record_kron_bfgs_A_twoNorm_per_epoch']:
            dict_result['kron_bfgs_A_twoNorms_per_epoch'] = data_['kron_bfgs_A_twoNorms_per_epoch']
        if params['if_records']['if_record_kron_bfgs_G_LM_twoNorm_per_epoch']:
            dict_result['kron_bfgs_G_LM_twoNorms_per_epoch'] = data_['kron_bfgs_G_LM_twoNorms_per_epoch']
        if params['if_records']['if_record_kron_bfgs_Hg_twoNorm_per_epoch']:
            dict_result['kron_bfgs_Hg_twoNorms_per_epoch'] = data_['kron_bfgs_Hg_twoNorms_per_epoch']
        if params['if_records']['if_record_kron_bfgs_Ha_twoNorm_per_epoch']:
            dict_result['kron_bfgs_Ha_twoNorms_per_epoch'] = data_['kron_bfgs_Ha_twoNorms_per_epoch']
        if params['if_records']['if_record_layerWiseHessian_twoNorm_per_epoch']:
            dict_result['layerWiseHessian_twoNorms_per_epoch'] = data_['layerWiseHessian_twoNorms_per_epoch']
        if params['if_records']['if_record_inverseLayerWiseHessian_twoNorm_per_epoch']:
            dict_result['inverseLayerWiseHessian_twoNorms_per_epoch'] = data_['inverseLayerWiseHessian_twoNorms_per_epoch']
        if params['if_records']['if_record_inverseLayerWiseHessian_LM_twoNorm_per_epoch']:
            dict_result['inverseLayerWiseHessian_LM_twoNorms_per_epoch'] = data_['inverseLayerWiseHessian_LM_twoNorms_per_epoch']
        if params['if_records']['if_record_inverseLayerWiseHessian_LM_MA_twoNorm_per_epoch']:
            dict_result['inverseLayerWiseHessian_LM_MA_twoNorms_per_epoch'] = data_['inverseLayerWiseHessian_LM_MA_twoNorms_per_epoch']
        if params['if_records']['if_record_kron_bfgs_matrix_norm'] == True:
            dict_result['kron_bfgs_matrix_norms'] = data_['kron_bfgs_matrix_norms']
            
        if 'if_record_kfac_F_twoNorm_per_epoch' in params and\
        params['if_record_kfac_F_twoNorm_per_epoch']:
            dict_result['kfac_F_twoNorms_per_epoch'] = data_['kfac_F_twoNorms_per_epoch']
            
        if 'if_record_kron_bfgs_norm_s_y_per_iter' in params and\
        params['if_record_kron_bfgs_norm_s_y_per_iter']:
            dict_result['kron_bfgs_norms_s_y_per_iter'] = data_['kron_bfgs_norms_s_y_per_iter']
            
        if 'if_record_kron_bfgs_sTy_per_iter' in params and\
        params['if_record_kron_bfgs_sTy_per_iter']:
            dict_result['kron_bfgs_sTy_per_iter'] = data_['kron_bfgs_sTy_per_iter']

    if params['if_record_sgn_norm']:
        sgn_norms = np.asarray(sgn_norms)
        dict_result['sgn_norms'] = sgn_norms
        

    params_saved = {}
    for key_ in params['keys_params_saved']:
        params_saved[key_] = params[key_]
    dict_result['params'] = params_saved

    path_to_goolge_drive_dir = params['home_path'] + 'result/'
    os.makedirs(path_to_goolge_drive_dir + name_result, exist_ok = True)

    fake_args = {}
    fake_args['algorithm_dict'] = {}
    fake_args['algorithm_dict']['name'] = params['algorithm']
    # for key in dict_result['params']:
    fake_args['algorithm_dict']['params'] = dict_result['params']
    fake_args['home_path'] = params['home_path']
    fake_args['N1'] = params['N1']
    fake_args['N2'] = params['N2']
    fake_args['if_gpu'] = params['if_gpu']
    fake_args['dataset'] = name_dataset
    fake_args['name_loss'] = params['name_loss']
    
    fake_args['list_lr'] = [params['alpha']]
    fake_args['tuning_criterion'] = 'test_acc'
    # does not matter because 
    # presumably, there will be at most 1 old pkl
    
    _, _, old_pkl_name = get_best_params(fake_args, if_plot=False)

    if old_pkl_name != None:
        print('Remove old result:')
        print(name_result + old_pkl_name)
        
        os.remove(path_to_goolge_drive_dir + name_result + old_pkl_name)

    import datetime        

    filename_result_with_time =\
    'result_' + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S") +'.pkl'
    
    print('dict_result.keys()')
    print(dict_result.keys())
    
    print('dict_result[params].keys()')
    print(dict_result['params'].keys())
    
    print('dict_result[params')
    print(dict_result['params'])
    
    with open(path_to_goolge_drive_dir + name_result + filename_result_with_time, 'wb') as output_result:
        pickle.dump(dict_result, output_result)

    print('Saved at ' + name_result + filename_result_with_time)

    return name_result, data_, dict_result['params']

def print_gpu_usage(params):
    device = params['device']
    
    gpu_total_memory = torch.cuda.get_device_properties(device).total_memory
    gpu_cached = torch.cuda.memory_reserved(device)
    gpu_allocated = torch.cuda.memory_allocated(device)
    # f = c-a  # free inside cache

    print('total GPU memory: {0:.3f} GB, cached: {1:.3f} GB, allocated: {2:.3f} GB'.format(
        gpu_total_memory * 1e-9, gpu_cached * 1e-9, gpu_allocated * 1e-9))

def get_sort_profile():
    filepath = 'lprof0.txt'
    with open(filepath) as fp:
        line = fp.readline()
        cnt = 1
        list_percent_time = []
        list_line_1 = []
        while line:
            if cnt <=9:
                print(line)
            
            line_1 = line.strip().split()
            if len(line_1) > 5 and\
            line_1[0].replace('.','',1).isdigit() and\
            line_1[1].replace('Error: check if need to save shampoo.','',1).isdigit() and\
            line_1[2].replace('.','',1).isdigit() and\
            line_1[3].replace('.','',1).isdigit() and\
            line_1[4].replace('.','',1).isdigit():
                list_percent_time.append(float(line_1[4]))
                list_line_1.append(line)
            
            # print(line_1[0])
            line = fp.readline()
            cnt += 1


    list_percent_time = np.asarray(list_percent_time)

    argsort_list_percent_time = np.argsort(-list_percent_time)
    for i in argsort_list_percent_time:
        print(list_line_1[i])

def SGD_update(data_, params):
    true_algorithm = params['algorithm']
    if params['algorithm'] in ['SGD-yura-MA',
                               'SGD-yura',
                               'SGD-momentum-yura',
                               'SGD-VA',
                               'SGD-signVAsqrt',
                               'SGD-signVAerf',
                               'SGD-signVA',
                               'SGD-sign']:
        params['algorithm'] = 'SGD'
    elif params['algorithm'] == 'SGD-yura-old':
        params['algorithm'] = 'SGD-yura'
    model_grad = data_['model_grad_used_torch']
        
    p = get_opposite(model_grad)
    # p_torch = get_opposite(model_grad_torch, params)

    if params['algorithm'] == 'SGD-yura' or\
    params['algorithm'] == 'SGD-yura-BD':
        # alpha = 2
        alpha = 1
        
        print('check whether we should use alpha or alpha_current')
        sys.exit()
        

        if params['i'] == 0:
            if params['algorithm'] == 'SGD-yura':
                lambda_0 = 1
                lambda_k = lambda_0
                theta_k = 10**10
            elif params['algorithm'] == 'SGD-yura-BD':
                lambda_0 = [1] * params['numlayers']
                lambda_k = lambda_0
                theta_k = [10**10] * params['numlayers']
                # print('test')
        else:
            lambda_k_minus_1 = params['yura_lambda']
            theta_k_minus_1 = params['yura_theta']
            weights_k_minus_1 = params['yura_weights']
            grad_k_minus_1 = params['yura_grad']

            
            # get previous grad
            model_new = copy.deepcopy(data_['model'])
            # model_new = get_model(params)


            device = params['device']
            for l in range(model_new.numlayers):
                for key in model_new.layers_weight[l]:

                    # model_new.layers_weight[l][key].data = torch.from_numpy(weights_k_minus_1[l][key]).float().to(device)
                    model_new.layers_weight[l][key].data = weights_k_minus_1[l][key].data
            

            reduction = 'mean'
            loss = get_regularized_loss_from_x(model_new, data_['X_mb'], data_['t_mb'], reduction)

            model_new.zero_grad()

            loss.backward()

            grad_k_minus_1_torch = get_model_grad(model_new, params)
            # diff_grad = get_subtract(model_grad, grad_k_minus_1, params)
            diff_grad_torch = get_subtract_torch(model_grad_torch, grad_k_minus_1_torch)

            if params['algorithm'] == 'SGD-yura':
                norm_sqaure_diff_weights =\
                get_dot_product_torch(data_['p_torch'], data_['p_torch']) * (params['alpha']**2)
                norm_sqaure_diff_weights_np = norm_sqaure_diff_weights.cpu().data.numpy()

                norm_sqaure_diff_grad_torch = get_dot_product_torch(diff_grad_torch, diff_grad_torch)
                norm_sqaure_diff_grad_np = norm_sqaure_diff_grad_torch.cpu().data.numpy()

                L_k_inv = np.sqrt(get_safe_division(norm_sqaure_diff_weights_np,
                    norm_sqaure_diff_grad_np))
                lambda_k = min(np.sqrt(1 + theta_k_minus_1 / 10) * lambda_k_minus_1, L_k_inv / alpha)

                if lambda_k == 0:
                    print('Warning: lambda_k == 0')
                    lambda_k = lambda_k_minus_1
                theta_k = lambda_k / lambda_k_minus_1

            elif params['algorithm'] == 'SGD-yura-BD':
                lambda_k = []
                theta_k = []

                norm_sqaure_diff_weights =\
                get_dot_product_blockwise_torch(data_['p_torch'], data_['p_torch']) * (params['alpha']**2)
                norm_sqaure_diff_weights = [element_.cpu().data.numpy() for element_ in norm_sqaure_diff_weights]

                norm_sqaure_diff_grad = get_dot_product_blockwise(diff_grad, diff_grad)

                for l in range(params['numlayers']):
                    L_k_inv = np.sqrt(get_safe_division(norm_sqaure_diff_weights[l], norm_sqaure_diff_grad[l]))
                    lambda_k_l = min(
                        np.sqrt(1 + theta_k_minus_1[l] / 10) * lambda_k_minus_1[l], L_k_inv / alpha)
                    theta_k_l = lambda_k_l / lambda_k_minus_1[l]
                    lambda_k.append(lambda_k_l)
                    theta_k.append(theta_k_l)

        if params['algorithm'] == 'SGD-yura':    
            p = get_multiply_scalar(lambda_k, p)
        elif params['algorithm'] == 'SGD-yura-BD':
            p = get_multiply_scalar_blockwise(lambda_k, p)

        params['yura_lambda'] = lambda_k
        params['yura_theta'] = theta_k
        params['yura_weights'] = copy.deepcopy(data_['model'].layers_weight)
        params['yura_grad'] = copy.deepcopy(model_grad)
    elif params['algorithm'] in ['SGD']:
        1
    else:
        print('Error: unkown algo when yura')
        sys.exit()
    if params['algorithm'] in ['SGD-LRdecay']:
        print('error: should not reach here')
        
        sys.exit()
        params['alpha_current'] =\
    params['alpha'] *\
    (params['lr_decay_rate'] ** (params['epoch'] // params['num_epoch_to_decay']))
    elif params['algorithm'] in ['SGD']:
        pass
    else:
        print('params[algorithm]')
        print(params['algorithm'])
        sys.exit()


    data_['p_torch'] = p
    if true_algorithm in ['SGD-yura', 
                          'SGD-yura-MA', 
                          'SGD-momentum-yura',
                          'SGD-momentum',
                          'SGD-VA',
                          'SGD-signVA',
                          'SGD-signVAerf',
                          'SGD-signVAsqrt',
                          'SGD-sign']:
        params['algorithm'] = true_algorithm
    elif true_algorithm == 'SGD-yura-old':
        params['algorithm'] = true_algorithm
    return data_

def RMSprop_update(data_, params):
    model_grad = data_['model_grad_used_torch']

    algorithm = params['algorithm']
    
    if algorithm in ['Adam',
                     'Adam-noWarmStart']:
        beta_1 = params['momentum_gradient_rho']
        
        assert params['momentum_gradient_rho'] == params['momentum_gradient_dampening']
        
        i = params['i']
        
        model_grad = get_multiply_scalar(1 / (1 - beta_1**(i+1)), model_grad)
        
    elif algorithm in ['RMSprop',
                       'RMSprop-warmStart']:
        1
    else:
        print('error: check if bias correction for grad for ' + algorithm)
        sys.exit()

    if algorithm == 'RMSprop-individual-grad-no-sqrt-LM':
        epsilon = params['lambda_']
    elif algorithm in ['RMSprop-individual-grad-no-sqrt-Fisher',
                       'RMSprop-individual-grad-no-sqrt',
                       'RMSprop-individual-grad',
                       'RMSprop-no-sqrt',
                       'RMSprop',
                       'RMSprop-warmStart',
                       'RMSprop-test',
                       'Adam',
                       'Adam-test',
                       'Adam-noWarmStart']:
        epsilon = params['RMSprop_epsilon']
    else:
        print('Error: undefined epsilon.')
        sys.exit()
    beta_2 = params['RMSprop_beta_2']
    if algorithm in ['RMSprop',
                     'RMSprop-warmStart',
                     'RMSprop-test',
                     'Adam',
                     'Adam-test',
                     'Adam-noWarmStart',
                     'RMSprop-no-sqrt']:
        data_['RMSprop_momentum_2'] =\
        get_plus_torch(
            get_multiply_scalar(beta_2, data_['RMSprop_momentum_2']), 
            get_multiply_scalar(1-beta_2, get_square_torch(model_grad)))
    elif algorithm == 'RMSprop-individual-grad' or\
    algorithm == 'RMSprop-individual-grad-no-sqrt' or\
    algorithm == 'RMSprop-individual-grad-no-sqrt-Fisher' or\
    algorithm == 'RMSprop-individual-grad-no-sqrt-LM':
        a_grad_N2 = data_['a_grad_N2']
        h_N2 = data_['h_N2']

        model = data_['model']

        N2 = params['N2']

        for l in range(model.numlayers):
            if params['layers_params'][l]['name'] == 'fully-connected':

                h_l_square = torch.mul(h_N2[l], h_N2[l])
                a_grad_l_square = torch.mul(a_grad_N2[l], a_grad_N2[l]) # N2 * m_l

                W_l_square = torch.mm(h_l_square.t(), a_grad_l_square) / N2

                data_['RMSprop_momentum_2'][l]['W'] =\
                beta_2 * data_['RMSprop_momentum_2'][l]['W'] +\
                (1-beta_2) * W_l_square.t().cpu().data.numpy()

                data_['RMSprop_momentum_2'][l]['b'] =\
                beta_2 * data_['RMSprop_momentum_2'][l]['b'] +\
                (1-beta_2) * torch.mean(a_grad_l_square, dim=0).cpu().data.numpy()
            elif params['layers_params'][l]['name'] == 'conv':
                print('h_N2[l].size')
                print(h_N2[l].size())
                print('a_grad_N2[l].size')
                print(a_grad_N2[l].size())
                print('model_grad[l][W].shape()')
                print(model_grad[l]['W'].shape)

                h_N2_l_pad = F.pad(h_N2[l], (2,2,2,2))

                print('h_N2_l_pad.size()')
                print(h_N2_l_pad.size())
                for i in range(model_grad[l]['W'].shape[0]):
                    for j in range(model_grad[l]['W'].shape[1]):
                        for test_h in range(model_grad[l]['W'].shape[2]):
                            for test_w in range(model_grad[l]['W'].shape[3]):
                                print('i, j, test_h, test_w')
                                print(i, j, test_h, test_w)


                                print('torch.sum(torch.mean(torch.mul(a_grad_N2[l][:, i], h_N2_l_pad[:, j, test_h: test_h+28, test_w: test_w+28]), dim=0)) -\
                                      torch.from_numpy(model_grad[l][W][i, j, test_h, test_w]).float().cuda()')
                                print(torch.sum(torch.mean(torch.mul(a_grad_N2[l][:, i], h_N2_l_pad[:, j, test_h: test_h+28, test_w: test_w+28]), dim=0)) -\
                                      model_grad[l]['W'][i, j, test_h, test_w])

                for i in range(len(h_N2[l])):
                    print('h_N2_l_pad[i].size()')
                    print(h_N2_l_pad[i].size())
                    print('a_grad_N2[l][i].size()')
                    print(a_grad_N2[l][i].size())

                    h_N2_l_pad_i_expand = torch.unsqueeze(h_N2_l_pad[i], 0)

                    print('h_N2_l_pad_i_expand.size()')
                    print(h_N2_l_pad_i_expand.size())

                    sys.exit()
            else:
                print('Error: unknown layer when update rmsprop')
                sys.exit()
    else:
        print('Error: unsupported algorithm.')
        sys.exit()
        
    if algorithm in ['Adam',
                     'Adam-test',
                     'Adam-noWarmStart']:
        
        i = params['i']
        
        model_grad_second_moment = get_multiply_scalar(1 / (1 - beta_2**(i+1)), data_['RMSprop_momentum_2'])
        
    elif algorithm in ['RMSprop',
                       'RMSprop-warmStart']:
        model_grad_second_moment = data_['RMSprop_momentum_2']
    else:
        print('error: check if bias correction for grad for ' + algorithm)
        sys.exit()
    
        


    if algorithm in ['RMSprop',
                     'RMSprop-warmStart',
                     'RMSprop-test',
                     'Adam',
                     'Adam-test',
                     'Adam-noWarmStart',
                     'RMSprop-individual-grad']:
        p = get_divide_torch(
            model_grad, 
            get_plus_scalar(epsilon, get_sqrt_torch(model_grad_second_moment)))
    elif algorithm == 'RMSprop-individual-grad-no-sqrt' or\
    algorithm == 'RMSprop-no-sqrt' or\
    algorithm == 'RMSprop-individual-grad-no-sqrt-Fisher' or\
    algorithm == 'RMSprop-individual-grad-no-sqrt-LM':
        p = get_divide(
            model_grad, 
            get_plus_scalar(epsilon, model_grad_second_moment))

    else:
        print('Error: unsupported algorithm 2.')
        sys.exit()
    
    p = get_opposite(p)

    data_['p_torch'] = p
    return data_

def get_Adam_direction(p, data_, params):
    beta_1 = params['Adam_beta_1']
    beta_2 = params['Adam_beta_2']
    epsilon = params['Adam_epsilon']

    i = params['i'] + 1
    data_['model_grad_Adam_momentum_1'] = get_plus(
    get_multiply_scalar(beta_1, data_['model_grad_Adam_momentum_1']),
    get_multiply_scalar(1 - beta_1, p))

    data_['model_grad_Adam_momentum_2'] = get_plus(
    get_multiply_scalar(beta_2, data_['model_grad_Adam_momentum_2']),
    get_multiply_scalar(1 - beta_2, 
                 get_square(p)))

    hat_m = get_multiply_scalar(1 / (1 - beta_1 ** i), data_['model_grad_Adam_momentum_1'])
    hat_v = get_multiply_scalar(1 / (1 - beta_2 ** i), data_['model_grad_Adam_momentum_2'])


    p_Adam = get_divide(
        hat_m,
        get_plus_scalar(epsilon,
                        get_sqrt(hat_v)))

    return p_Adam, data_

def symsqrt(matrix):
    """Compute the square root of a positive definite matrix."""
    _, s, v = matrix.svd()
    good = s > s.max(-1, True).values * s.size(-1) * torch.finfo(s.dtype).eps
    components = good.sum(-1)
    common = components.max()
    unbalanced = common != components.min()
    if common < s.size(-1):
        s = s[..., :common]
        v = v[..., :common]
        if unbalanced:
            good = good[..., :common]
    if unbalanced:
        s = s.where(good, torch.zeros((), device=s.device, dtype=s.dtype))
    return (v * s.sqrt().unsqueeze(-2)) @ v.transpose(-2, -1)

def mbngd_if_inverse(params):  
    i = params['i']
    inverse_update_freq = params['kfac_inverse_update_freq']
    cov_update_freq = params['kfac_cov_update_freq']
    
    if (i <= inverse_update_freq and i % cov_update_freq == 0) or i % inverse_update_freq == 0:
        return True
    else:
        return False
           
def mbngd_update(data_, params):
    algorithm = params['algorithm']
    F_m = data_['F_m']
    F_inv = data_['F_inv']
    model = data_['model']
    
    N1 = params['N1']
    i = params['i']
    beta_1 = params['momentum_gradient_rho']
    dampening = params['momentum_gradient_dampening']

    model_grad = data_['model_grad_used_torch']
    # model_grad_beta1 = get_multiply_scalar(1 / (1 - beta_1**(i+1)), data_['model_grad_beta1'])
    model_grad_beta1 = data_['model_grad_beta1']
    model_grad_N2 = data_['model_grad_torch'] 
    # model_grad_N2 = data_['model_grad_beta1']
    # model_grad_N2 = data_['model_grad_N2']
    
    # N2 = params['N2']

    inverse_update_freq = params['kfac_inverse_update_freq']
    cov_update_freq = params['kfac_cov_update_freq']
    
    lambda_ = params['mbngd_damping_lambda']
    lambda_epsilon = params['mbngd_damping_epsilon']

    numlayers = params['numlayers']
    kfac_rho = params['kfac_rho']
    beta_2 = 0.999
    rho = kfac_rho

    device = params['device']
    
    # Step
    delta = []
    for l in range(numlayers):
        delta_l = {}
        
        if params['layers_params'][l]['name'] in ['fully-connected', 'BN']:
            if (algorithm == 'MBNGD-m1m2')*(params['layers_params'][l]['name'] == 'fully-connected'):
                if i % cov_update_freq == 0:
                    homo_model_grad_N2_l = torch.cat((model_grad_N2[l]['W'], model_grad_N2[l]['b'].unsqueeze(1)), dim=1)
                    ggT = torch.einsum('ai, aj-> aij', homo_model_grad_N2_l, homo_model_grad_N2_l)
                    F_m[l]['W'] = kfac_rho*F_m[l]['W'].data + (1-kfac_rho)*ggT

                if mbngd_if_inverse(params):
                    F_m_LM = F_m[l]['W'] + lambda_*torch.eye(F_m[l]['W'].size()[2], device=device).reshape((1, F_m[l]['W'].size()[2], F_m[l]['W'].size()[2])).repeat(F_m[l]['W'].size()[0], 1, 1)
                    F_inv[l] = F_m_LM.inverse()

                homo_model_grad_N1_l = torch.cat((model_grad[l]['W'], model_grad[l]['b'].unsqueeze(1)), dim=1)
                dW_M = torch.matmul(F_inv[l], homo_model_grad_N1_l.unsqueeze(-1)).reshape(homo_model_grad_N1_l.shape)

                delta_l['W'] = dW_M[:, :-1]
                delta_l['b'] = dW_M[:, -1]
            elif (algorithm == 'MBNGD-one-to-all')*(params['layers_params'][l]['name'] == 'fully-connected'):
                if i % cov_update_freq == 0:
                    homo_model_grad_N2_l = torch.cat((model_grad_N2[l]['W'], model_grad_N2[l]['b'].unsqueeze(1)), dim=1)
                    ggT = torch.einsum('ia, ja-> aij', homo_model_grad_N2_l, homo_model_grad_N2_l)
                    F_m[l]['W'] = kfac_rho*F_m[l]['W'].data + (1-kfac_rho)*ggT

                if mbngd_if_inverse(params):
                    F_m_LM = F_m[l]['W'] + lambda_*torch.eye(F_m[l]['W'].size()[1], device=device).reshape((1, F_m[l]['W'].size()[1], F_m[l]['W'].size()[1])).repeat(F_m[l]['W'].size()[0],1,1)
                    F_inv[l] = F_m_LM.inverse()

                homo_model_grad_N1_l = torch.cat((model_grad[l]['W'], model_grad[l]['b'].unsqueeze(1)), dim=1)
                dW_M = torch.matmul(F_inv[l], homo_model_grad_N1_l.view(homo_model_grad_N1_l.shape[1], homo_model_grad_N1_l.shape[0]).unsqueeze(-1)).reshape(homo_model_grad_N1_l.shape)

                delta_l['W'] = dW_M[:, :-1]
                delta_l['b'] = dW_M[:, -1]
                
            elif (algorithm == 'MBNGD-all-to-one')*(params['layers_params'][l]['name'] == 'fully-connected'):
                Os, Is = model_grad_N2[l]['W'].shape
                m1m2, Ooverm2, _ = F_m[l]['W'].shape
                
                m2 = Os//Ooverm2
                m1 = m1m2//m2
                
                if i % cov_update_freq == 0:
                    homo_model_grad_N2_l = torch.cat((model_grad_N2[l]['W'], model_grad_N2[l]['b'].unsqueeze(1)), dim=1)
                    #grad_view = homo_model_grad_N2_l.view(m1*m2, ((Is+1)//m1)*(Os//m2))
                    #ggT = torch.einsum('mi, mj-> mij', grad_view, grad_view)
                    #F_m[l]['W'] = kfac_rho*F_m[l]['W'].data + (1-kfac_rho)*ggT
                    F_m[l]['W'] = kfac_rho*F_m[l]['W'].data + (1-kfac_rho)*torch.einsum('mi, mj-> mij', homo_model_grad_N2_l.view(m1*m2, ((Is+1)//m1)*(Os//m2)), 
                        homo_model_grad_N2_l.view(m1*m2, ((Is+1)//m1)*(Os//m2)))

                if mbngd_if_inverse(params):
                    n_m, s_m, _ = F_m[l]['W'].shape
                    F_m_LM = F_m[l]['W'] + lambda_*torch.eye(s_m, device=device).reshape((1, s_m, s_m)).repeat(n_m, 1, 1)
                    F_inv[l] = F_m_LM.inverse()

                homo_model_grad_N1_l = torch.cat((model_grad[l]['W'], model_grad[l]['b'].unsqueeze(1)), dim=1)
                dW_M = torch.matmul(F_inv[l], homo_model_grad_N1_l.view(F_inv[l].shape[0], F_inv[l].shape[1]).unsqueeze(-1)).reshape(homo_model_grad_N1_l.shape)
                
                delta_l['W'] = dW_M[:, :-1]
                delta_l['b'] = dW_M[:, -1]
                
            elif (algorithm == 'MBNGD-all-to-one-Avg')*(params['layers_params'][l]['name'] == 'fully-connected'):
                Os, Is = model_grad_N2[l]['W'].shape
                m1, m2 = F_m[l]['m1m2']
                matrix_size = (Os//m2)*((Is+1)//m1)
                
                if matrix_size < -1:
                    if i % cov_update_freq == 0:
                        homo_model_grad_N2_l = torch.cat((model_grad_N2[l]['W'], model_grad_N2[l]['b'].unsqueeze(1)), dim=1)
                        grad_view = homo_model_grad_N2_l.view(m1*m2, ((Is+1)//m1)*(Os//m2))
                        F_m[l]['W'] = kfac_rho*F_m[l]['W'].data + (1-kfac_rho)*torch.einsum('mi, mj-> mij', grad_view, grad_view)
                    if mbngd_if_inverse(params):
                        n_m, s_m, _ = F_m[l]['W'].shape
                        F_m_LM = F_m[l]['W'] + lambda_*torch.eye(s_m, device=device).reshape((1, s_m, s_m)).repeat(n_m, 1, 1)
                        F_inv[l] = F_m_LM.inverse()
                    homo_model_grad_N1_l = torch.cat((model_grad[l]['W'], model_grad[l]['b'].unsqueeze(1)), dim=1)
                    dW_M = torch.matmul(F_inv[l], homo_model_grad_N1_l.view(F_inv[l].shape[0], F_inv[l].shape[1]).unsqueeze(-1)).reshape(homo_model_grad_N1_l.shape)
                else:
                    if i % cov_update_freq == 0:
                        homo_model_grad_N2_l = torch.cat((model_grad_N2[l]['W'], model_grad_N2[l]['b'].unsqueeze(1)), dim=1)
                        grad_view = homo_model_grad_N2_l.view(m1*m2, ((Is+1)//m1)*(Os//m2))
                        F_m[l]['W'] = kfac_rho*F_m[l]['W'].data + (1-kfac_rho)*torch.einsum('mi, mj-> ij', grad_view, grad_view)/(m1*m2)
                    if mbngd_if_inverse(params):
                        F_m_LM = F_m[l]['W'] + lambda_*torch.eye(((Is+1)//m1)*(Os//m2), device=device)
                        F_inv[l] = F_m_LM.inverse()
                    homo_model_grad_N1_l = torch.cat((model_grad[l]['W'], model_grad[l]['b'].unsqueeze(1)), dim=1)
                    dW_M = torch.matmul(F_inv[l], homo_model_grad_N1_l.view(m1*m2, ((Is+1)//m1)*(Os//m2)).unsqueeze(-1)).reshape(homo_model_grad_N1_l.shape)
                delta_l['W'] = dW_M[:, :-1]
                delta_l['b'] = dW_M[:, -1]
            elif (algorithm == 'L-MBNGD-all-to-one')*(params['layers_params'][l]['name'] == 'fully-connected'):
                Os, Is = model_grad_N2[l]['W'].shape
                r = F_m[0]['Gt'].shape[-1]
                mem = F_m[0]['Gt_m'].shape[-1]
                m1 = Is+1
                m2 = 1
                cov_update_freq = params['kfac_cov_update_freq']
                Beta_ef = 0.9
                
                if Os < -1:
                    Os, Is = model_grad_N2[l]['W'].shape
                    m1m2, Ooverm2, _ = F_m[l]['W'].shape
                    m2 = Os//Ooverm2
                    m1 = m1m2//m2

                    if i % cov_update_freq == 0:
                        homo_model_grad_N2_l = torch.cat((model_grad_N2[l]['W'], model_grad_N2[l]['b'].unsqueeze(1)), dim=1)
                        grad_view = homo_model_grad_N2_l.view(m1*m2, ((Is+1)//m1)*(Os//m2))
                        F_m[l]['W'] = kfac_rho*F_m[l]['W'].data + (1-kfac_rho)*torch.einsum('mi, mj-> mij', grad_view, grad_view)

                    if mbngd_if_inverse(params):
                        n_m, s_m, _ = F_m[l]['W'].shape
                        F_m_LM = F_m[l]['W'] + lambda_*torch.eye(s_m, device=device).reshape((1, s_m, s_m)).repeat(n_m, 1, 1)
                        F_inv[l] = F_m_LM.inverse()
                        
                    homo_model_grad_N1_l = torch.cat((model_grad[l]['W'], model_grad[l]['b'].unsqueeze(1)), dim=1)
                    dW_M = torch.matmul(F_inv[l], homo_model_grad_N1_l.view(F_inv[l].shape[0], F_inv[l].shape[1]).unsqueeze(-1)).reshape(homo_model_grad_N1_l.shape)
                else:  
                    homo_model_grad_N2_l = torch.cat((model_grad_N2[l]['W'], model_grad_N2[l]['b'].unsqueeze(1)), dim=1)
                    grad_view = homo_model_grad_N2_l.view(m1*m2, ((Is+1)//m1)*(Os//m2))
                    F_m[l]['Gt_m'][:, :, 1:] = Beta_ef*F_m[l]['Gt_m'][:, :, :-1]
                    F_m[l]['Gt_m'][:, :, 0] = grad_view.data
                    
                    if i % cov_update_freq == 0: 
                        if mem < r:
                            F_m[l]['Gt'][:, :, mem:] = (Beta_ef**cov_update_freq)*F_m[l]['Gt'][:, :, :-mem]
                        F_m[l]['Gt'][:, :, :mem] = F_m[l]['Gt_m']

                    if mbngd_if_inverse(params):
                        # mE, mV = torch.symeig(torch.einsum('mij, mib -> mjb', F_m[l]['Gt'], F_m[l]['Gt']) + lambda_*torch.eye(r, device = device).reshape((1, r, r)).repeat(Is+1, 1, 1), eigenvectors=True)
                        # sigma_inv = torch.pow(mE, -3)
                        # sigma_sqrt_min = torch.min(torch.sqrt(mE))
                        # F_inv[l] = {'mV': mV, 'sigma_inv':sigma_inv}
                        F_inv[l] = torch.inverse(torch.einsum('mij, mib -> mjb', F_m[l]['Gt'], F_m[l]['Gt']) + lambda_*torch.eye(r, device = device).reshape((1, r, r)).repeat(Is+1, 1, 1))
                        # F_inv[l] = torch.inverse(torch.einsum('mij, mib -> mjb', F_m[l]['Gt']/np.sqrt(min((i+1), r)), F_m[l]['Gt']/np.sqrt(min((i+1), r))) + lambda_*torch.eye(r, device = device).reshape((1, r, r)).repeat(Is+1, 1, 1))

                    homo_model_grad_N1_l = torch.cat((model_grad[l]['W'], model_grad[l]['b'].unsqueeze(1)), dim=1)
                    grad_view_N1 = homo_model_grad_N1_l.view(m1*m2, ((Is+1)//m1)*(Os//m2)).unsqueeze(-1)

                    # batch_diag = F_inv[l]['sigma_inv'].unsqueeze(2).expand(*F_inv[l]['sigma_inv'].size(), F_inv[l]['sigma_inv'].size(1)) * torch.eye(F_inv[l]['sigma_inv'].size(1), device = device)
                    # dW_M = torch.matmul(F_m[l]['Gt'], torch.matmul(F_inv[l]['mV'], 
                    #        torch.matmul(batch_diag, torch.matmul(torch.transpose(F_inv[l]['mV'], 1, 2), 
                    #        torch.matmul(torch.transpose(F_m[l]['Gt'], 1, 2), homo_model_grad_N1_l.view(m1*m2, ((Is+1)//m1)*(Os//m2)).unsqueeze(-1)))))).reshape(homo_model_grad_N1_l.shape)

                    dW_M = ((grad_view_N1/lambda_) - torch.matmul(F_m[l]['Gt'], torch.matmul(F_inv[l], torch.matmul(torch.transpose(F_m[l]['Gt'], 1, 2), grad_view_N1)))).reshape(homo_model_grad_N1_l.shape)
                    # dW_M = (torch.matmul(F_m[l]['Gt'], torch.matmul(F_inv[l], torch.matmul(torch.transpose(F_m[l]['Gt'], 1, 2), grad_view.unsqueeze(-1))))).reshape(homo_model_grad_N1_l.shape)
                
                delta_l['W'] = dW_M[:, :-1]
                delta_l['b'] = dW_M[:, -1]
                
            else:
                F_m[l]['W'] = beta_2*F_m[l]['W'].data + (1-beta_2)*((model_grad_beta1[l]['W'].data)**2)
                F_m[l]['b'] = beta_2*F_m[l]['b'].data + (1-beta_2)*((model_grad_beta1[l]['b'].data)**2)
                dW_M = model_grad_beta1[l]['W'].data/(lambda_epsilon + torch.sqrt(F_m[l]['W'].data/(1 - beta_2**(i+1))))
                delta_l['W'] = dW_M
                db_M = model_grad_beta1[l]['b'].data/(lambda_epsilon + torch.sqrt(F_m[l]['b'].data/(1 - beta_2**(i+1))))
                delta_l['b'] = db_M
        elif params['layers_params'][l]['name'] in ['conv',
                                                        'conv-no-activation',
                                                        'conv-no-bias-no-activation']:
            if i % cov_update_freq == 0:
                flat_g = model_grad_N2[l]['W'].flatten(start_dim = 2)
                ggT = torch.einsum('abi, abj-> abij', flat_g, flat_g)
                F_m[l]['W'] = kfac_rho*F_m[l]['W'].data + (1-kfac_rho)*ggT
            if mbngd_if_inverse(params):
                F_m_LM = F_m[l]['W'] + lambda_*torch.eye(F_m[l]['W'].size()[2], device=device).reshape((1, F_m[l]['W'].size()[2], F_m[l]['W'].size()[2])).repeat(F_m[l]['W'].size()[0], F_m[l]['W'].size()[1], 1, 1)
                F_inv[l] = F_m_LM.inverse()
            dW_M = torch.matmul(F_inv[l], model_grad[l]['W'].flatten(start_dim = 2).unsqueeze(-1)).reshape(model_grad[l]['W'].shape)
            delta_l['W'] = dW_M
            if params['layers_params'][l]['name'] in ['conv', 'conv-no-activation']:
                F_m[l]['b'] = beta_2*F_m[l]['b'].data + (1-beta_2)*((model_grad_beta1[l]['b'].data)**2)
                db_M = model_grad_beta1[l]['b'].data/(lambda_epsilon + torch.sqrt(F_m[l]['b'].data/(1 - beta_2**(i+1))))
                delta_l['b'] = db_M
        else:
            print('Error: unknown layer when compute A')
            sys.exit()
        delta.append(delta_l)

    ##############
    algorithm = params['algorithm']  
    p = get_opposite(delta)
    data_['F_m'] = F_m 
    data_['F_inv'] = F_inv
    data_['p_torch'] = p
    
    
    return data_, params

# Kfac
def get_g_g_T_BN(model, l, batch_size):
    # In kfac code (https://github.com/tensorflow/kfac), they use a "sum the squares estimator"
    # (see the class ScaleAndShiftFullFB in https://github.com/tensorflow/kfac/blob/master/kfac/python/ops/fisher_blocks.py)
    # For more detail, see ScaleAndShiftFactor in https://github.com/tensorflow/kfac/blob/master/kfac/python/ops/fisher_factors.py
    
    # However, it is difficult to cache the intermediate variable of BN layer in pytorch.
    # Hence, we decide to use a "square the sum estimator", for simplicity
    
    g = torch.cat((model.layers_weight[l]['W'].grad.data, model.layers_weight[l]['b'].grad.data))
                    
    # g is averaged over minibatch
    # should first * batch_size, take outer product, then / batch_size 
    # i.e. "square the sum estimator" in kfac code
    # which is equivalent to batch_size * (g g^T)
    G_j = batch_size * torch.outer(g, g)
    return G_j

def kfac_if_inverse(params):
    i = params['i']
    inverse_update_freq = params['kfac_inverse_update_freq']
    cov_update_freq = params['kfac_cov_update_freq']
    
    if (i <= inverse_update_freq and i % cov_update_freq == 0) or i % inverse_update_freq == 0:
        return True
    else:
        return False
        
def kfac_update(data_, params):
    true_algorithm = params['algorithm']    
    i = params['i']
        
    if i == 0:
        params['kfac_svd_failed'] = False

    A = data_['A']
    G = data_['G']
    model = data_['model']
    model_grad = data_['model_grad_used_torch']
    
    model_grad_N1 = data_['model_grad_torch']
    if params['kfac_if_svd']:
        U_A = data_['U_A']
        U_G = data_['U_G']
        s_A = data_['s_A']
        s_G = data_['s_G']
    else:
        A_inv = data_['A_inv']
        G_inv = data_['G_inv']
   
    
    N1 = params['N1']
    N2 = params['N2']
    
    lambda_ = params['kfac_damping_lambda']
    lambda_A = math.sqrt(lambda_)
    lambda_G = math.sqrt(lambda_)

    numlayers = params['numlayers']
    kfac_rho = params['kfac_rho']
    device = params['device']
    
    h_N2 = data_['h_N2']
    
    # h denotes the bar_a in kfac paper, a_grad denotes the g
    G_ = []
    A_ = []
    
    rho = kfac_rho
    # used in ekfac
    homo_model_grad_N1 = get_homo_grad(model_grad_N1, params)
    
    if params['if_momentum_gradient']:
        homo_model_grad = get_homo_grad(model_grad, params)
    else:
        homo_model_grad = homo_model_grad_N1

        
    cov_update_freq = params['kfac_cov_update_freq']
        
    # Step
    delta = []
    for l in range(numlayers):
        
        if i % cov_update_freq == 0:
            if params['layers_params'][l]['name'] in ['fully-connected',
                                                      'conv',
                                                      'conv-no-activation',
                                                      'conv-no-bias-no-activation']:
                G_.append(get_g_g_T(data_['a_N2'], l, params))

                # no need to save A_, can be improved
                A_.append(get_A_A_T(data_['h_N2'], l, data_, params))

                # Update running estimates of KFAC
                A[l].data = rho*A[l].data + (1-rho)*A_[l].data
                G[l].data = rho*G[l].data + (1-rho)*G_[l].data

            elif params['layers_params'][l]['name'] in ['BN']:

                A_.append([])
                
                if params['kfac_if_update_BN'] and not params['kfac_if_BN_grad_direction']:

                    G_.append(get_g_g_T_BN(model, l, N2))

                    G[l].data = rho*G[l].data + (1-rho)*G_[l].data
                    
                else:
                    G_.append([])
            else:
                print('Error: unknown layer when compute A: ' + params['layers_params'][l]['name'])
                sys.exit()

            
        

        
        # Amortize the inverse. Only update inverses every now and then
        if kfac_if_inverse(params): 
            phi_ = 1
             
            if not params['kfac_if_svd']:

                if params['layers_params'][l]['name'] == 'BN':
                    if params['kfac_if_update_BN'] and not params['kfac_if_BN_grad_direction']:
                        G_l_LM = G[l] + (1 / lambda_) * torch.eye(G[l].size()[0], device=device)
                else:
                    A_l_LM = A[l] + (phi_ * lambda_A) * torch.eye(A[l].size()[0], device=device)
                    G_l_LM = G[l] + (1 / phi_ * lambda_G) * torch.eye(G[l].size()[0], device=device)

            if params['kfac_if_svd']:
                if params['layers_params'][l]['name'] == 'BN':
                    if params['kfac_if_update_BN'] and not params['kfac_if_BN_grad_direction']:
                        try:
                            s_G[l], U_G[l] = torch.symeig(G[l].data, eigenvectors=True)
                        except:
                            print('svd faild in G')
                            params['kfac_svd_failed'] = True
                else:
                    try:
                        s_A[l], U_A[l] = torch.symeig(A[l].data, eigenvectors=True)
                    except:
                        print('l')
                        print(l)
                        print('svd faild in A')
                        params['kfac_svd_failed'] = True
                    try:
                        s_G[l], U_G[l] = torch.symeig(G[l].data, eigenvectors=True)
                    except:
                        print('svd faild in G')
                        params['kfac_svd_failed'] = True
            else:
                if params['layers_params'][l]['name'] == 'BN':
                    if params['kfac_if_update_BN'] and not params['kfac_if_BN_grad_direction']:
                        G_inv[l] = G_l_LM.inverse()
                else:
                    A_inv[l] = A_l_LM.inverse()
                    G_inv[l] = G_l_LM.inverse()
    
        # compute kfac direction  
        if params['kfac_if_svd']:
            
            if params['layers_params'][l]['name'] == 'BN':
                
                if params['kfac_if_update_BN']:
            
                    if params['kfac_if_BN_grad_direction']:
                        homo_delta_l = copy.deepcopy(homo_model_grad[l])
                    else:
                        print('error: not implemented')
                        sys.exit()
                    
                else:
                    homo_delta_l = torch.zeros(homo_model_grad[l].size(), device=device)
                    
                
            else:
                homo_delta_l = torch.mm(
                    torch.mm(U_G[l].t(), homo_model_grad[l]),
                    U_A[l]
                )

                homo_delta_l = homo_delta_l / (torch.outer(s_G[l], s_A[l]) + params['kfac_damping_lambda'])

                homo_delta_l = torch.mm(
                    torch.mm(U_G[l], homo_delta_l),
                    U_A[l].t()
                )
        else:
        
            if params['layers_params'][l]['name'] == 'BN':

                if params['kfac_if_update_BN']:
                    
                    if params['kfac_if_BN_grad_direction']:
                        homo_delta_l = copy.deepcopy(homo_model_grad[l])
                    else:

                        homo_delta_l = torch.mv(G_inv[l], homo_model_grad[l])
                else:
                    homo_delta_l = torch.zeros(homo_model_grad[l].size(), device=device)
            else:
                homo_delta_l = torch.mm(torch.mm(G_inv[l], homo_model_grad[l]), A_inv[l])

            if params['if_test_mode']:
                if 'if_record_kfac_G_inv_norm_per_iter' in params and\
                params['if_record_kfac_G_inv_norm_per_iter']:
                    if l == 0:
                        data_['kfac_G_inv_norms_per_iter'].append([])

                    data_['kfac_G_inv_norms_per_iter'][-1].append(torch.norm(G_inv[l]).item())
            

        delta_l = from_homo_to_weight_and_bias(homo_delta_l, l, params)
        delta.append(delta_l)
        


    ##############
    algorithm = params['algorithm']
    p = get_opposite(delta)
    
    data_['A'] = A
    data_['G'] = G 
    # A, G are without LM
    if params['kfac_if_svd']:
        data_['U_A'] = U_A
        data_['U_G'] = U_G
        data_['s_A'] = s_A
        data_['s_G'] = s_G
    else:
    
        data_['A_inv'] = A_inv
        data_['G_inv'] = G_inv
   
    
    data_['p_torch'] = p
        
    data_['a_grad_N2'] = None
    data_['a_N2'] = None
    data_['h_N2'] = None
   
    return data_, params

def get_g_g_T(a, l, params):
    # returns the AVERAGED g_g_T across a minibatch
    
    layers_params = params['layers_params']
    
    if layers_params[l]['name'] == 'fully-connected':
        
        size_minibatch = a[l].size(0)
        
        
        # we use "size_minibatch *", instead of "1/size_minibatch *"
        # because a[l].grad is actually "1/size_minibatch * a[l].grad"
        G_j = size_minibatch * torch.mm(a[l].grad.t(), a[l].grad).data
        
    elif layers_params[l]['name'] in ['conv',
                                      'conv-no-activation',
                                      'conv-no-bias-no-activation']:
        # take Fashion-MNIST as an example:
        # a[l]: 1000 * 32 * 28 * 28
        # 1000: size of minibatch
        # 32: # out-channel
        # 28 * 28: size of image
        
        # return 1 / |T| / size_minibatch * g_g_T
        

        
        size_minibatch = a[l].size(0)
        
        
        a_l_grad = size_minibatch * a[l].grad
        
        a_l_grad_permuted = a_l_grad.permute(1, 0, 2, 3)
        
        a_l_grad_flattened = torch.flatten(a_l_grad_permuted, start_dim=1)
        
        G_j = torch.mm(a_l_grad_flattened, a_l_grad_flattened.t()) / a_l_grad_flattened.size(1)
    else:
        print('error: not implemented for ' + layers_params[l]['name'])
        sys.exit()
        
    return G_j

# Shampoo
def get_tensor_reshape_back(delta_l, l, name_variable, params):
    if get_tensor_reshape_option(l, name_variable, params) == 'filter-flattening':
        kernel_size = params['layers_params'][l]['conv_kernel_size']
        delta_l = delta_l.view(delta_l.size(0), delta_l.size(1), kernel_size, kernel_size)
        
    elif get_tensor_reshape_option(l, name_variable, params) == 'KFAC-reshaping':
        
        kernel_size = params['layers_params'][l]['conv_kernel_size']
        conv_in_channels = params['layers_params'][l]['conv_in_channels']
        
        delta_l = delta_l.view(delta_l.size(0), conv_in_channels, kernel_size, kernel_size)
        
    elif get_tensor_reshape_option(l, name_variable, params) == 'None':
        pass
        
    else:
        print('get_tensor_reshape_option(l, name_variable, params)')
        print(get_tensor_reshape_option(l, name_variable, params))
        sys.exit()
    
    return delta_l

def get_tensor_reshape_option(l, name_variable, params):
    if params['algorithm'] in ['shampoo-allVariables-filterFlattening-warmStart',
                               'shampoo-allVariables-filterFlattening-warmStart-lessInverse']:
        
        if params['layers_params'][l]['name'] in ['conv',
                                                  'conv-no-activation',
                                                  'conv-no-bias-no-activation']:
        
            if name_variable == 'W': 
                if params['algorithm'] in ['shampoo-allVariables-filterFlattening-warmStart',
                                             'shampoo-allVariables-filterFlattening-warmStart-lessInverse']:
                    return 'filter-flattening'
                else:
                    print('params[algorithm]')
                    print(params['algorithm'])
                    sys.exit()

            elif name_variable == 'b':
                return 'None'
                
            else:
                print('name_variable')
                print(name_variable)
                sys.exit()
            
            
        elif params['layers_params'][l]['name'] in ['fully-connected',
                                                    'BN']:
            return 'None'
            
        else:
            print('params[layers_params][l]')
            print(params['layers_params'][l])
            
            sys.exit()
            
    elif params['algorithm'] in ['shampoo-allVariables-warmStart',
                                 'shampoo-allVariables-warmStart-lessInverse',]:
        return 'None'
    else:
        
        print('params[algorithm]')
        print(params['algorithm'])
        sys.exit()

def get_tensor_reshape(g_W, l, name_variable, params):
    if get_tensor_reshape_option(l, name_variable, params) == 'filter-flattening':
        g_W = g_W.view(g_W.size(0), g_W.size(1), g_W.size(2) * g_W.size(3))
    elif get_tensor_reshape_option(l, name_variable, params) == 'KFAC-reshaping':
        g_W = g_W.view(g_W.size(0), g_W.size(1) * g_W.size(2) * g_W.size(3))
        
    elif get_tensor_reshape_option(l, name_variable, params) == 'None':
        1
    else:
        print('get_tensor_reshape_option(l, name_variable, params)')
        print(get_tensor_reshape_option(l, name_variable, params))
        sys.exit()
        
    return g_W
        
    
    
    '''
    if params['algorithm'] == 'matrix-normal-same-trace-allVariables-filterFlattening-warmStart':
        
        
        
        
        
        if params['layers_params'][l]['name'] == 'conv':
        
            if name_variable == 'W': 
                g_W = g_W.view(g_W.size(0), g_W.size(1), g_W.size(2) * g_W.size(3))

            elif name_variable == 'b':
        
                1
                
            else:
                
                print('name_variable')
                print(name_variable)
        
                sys.exit()
            
            
        elif params['layers_params'][l]['name'] == 'fully-connected':
            1
            
        else:
            print('params[layers_params][l]')
            print(params['layers_params'][l])
            
            sys.exit()
            
        
    else:
        
        print('params[algorithm]')
        print(params['algorithm'])
    
        sys.exit()
    
    
    
    return g_W
    '''

def get_if_shampoo_update(name_variable, params):
    if name_variable == 'W' or\
    (
        name_variable == 'b'):
        return True
    else:
        print('error: not implemented for ' + name_variable)
        sys.exit()
    
def shampoo_kron_matrices_warm_start_per_variable(j, model_grad_N1, l, name_variable, data_, params):
    if not get_if_shampoo_update(name_variable, params):
        return
    g_W = model_grad_N1[l][name_variable]
    g_W = get_tensor_reshape(g_W, l, name_variable, params)

    test_H_l = shampoo_get_list_of_contractions(g_W)

    if j == 1:
        data_['shampoo_H'][l][name_variable] = test_H_l
    else:



        for ii in range(len(data_['shampoo_H'][l][name_variable])):
            data_['shampoo_H'][l][name_variable][ii] *= (j-1)/j
            data_['shampoo_H'][l][name_variable][ii] += 1/j * test_H_l[ii]

def shampoo_get_list_of_contractions(g_W):
    
    test_H_l = []
        
    for ii in range(len(g_W.size())):
        axes = list(range(len(g_W.size())))

        axes.remove(ii)

        test_H_l.append(
            torch.tensordot(g_W, g_W, dims=(axes, axes)).data
        )        
    return test_H_l
    
def shampoo_kron_matrices_per_variable(model_grad_N1, l, name_variable, data_, params):
    # model_grad_N1: used for 2nd-order estimate
    
    if not get_if_shampoo_update(name_variable, params):
        return
        
    i = params['i']
    
    if i % params['shampoo_update_freq'] != 0:
        return

    decay_ = params['shampoo_decay']
    weight_ = params['shampoo_weight']

    if not params['if_warm_start']:
        weight_ = max(1/(i+1), weight_)




    g_W = model_grad_N1[l][name_variable]

    g_W = get_tensor_reshape(g_W, l, name_variable, params)






    test_H_l = shampoo_get_list_of_contractions(g_W)



    if i == 0:


        if not params['if_warm_start']:
            data_['shampoo_H'][l][name_variable] = test_H_l
    else:
        for ii in range(len(data_['shampoo_H'][l][name_variable])):
            data_['shampoo_H'][l][name_variable][ii] = decay_ * data_['shampoo_H'][l][name_variable][ii].data + weight_ * test_H_l[ii]
        
def shampoo_inversion_per_variable(model_grad_N1, l, name_variable, data_, params):
    if not get_if_shampoo_update(name_variable, params):
        return
        
    device = params['device']
    i = params['i']
    inverse_freq = params['shampoo_inverse_freq']

    if params['algorithm'] in ['shampoo-allVariables-warmStart-lessInverse',
                               'shampoo-allVariables-filterFlattening-warmStart-lessInverse']:
        pass
    elif params['algorithm'] in ['shampoo-allVariables-filterFlattening-warmStart']:
        if i < inverse_freq:
            inverse_freq = params['shampoo_update_freq']
    else:
        print('params[algorithm]')
        print(params['algorithm'])
        sys.exit()

        

    if i % inverse_freq == 0:
        if params['if_LM']:
            epsilon = params['lambda_']
        else:
            if params['algorithm'] in ['shampoo-allVariables-warmStart',
                                         'shampoo-allVariables-warmStart-lessInverse',
                                         'shampoo-allVariables-filterFlattening-warmStart',
                                         'shampoo-allVariables-filterFlattening-warmStart-lessInverse']:
                epsilon = params['shampoo_epsilon']
            else:
                print('params[algorithm]')
                print(params['algorithm'])
                sys.exit()

        if params['algorithm'] in ['shampoo-no-sqrt',
                                   'shampoo-no-sqrt-Fisher']:
            power_preconditioner = 1
        elif params['algorithm'] in ['shampoo',
                                     'shampoo-allVariables',
                                     'shampoo-allVariables-warmStart',
                                     'shampoo-allVariables-warmStart-lessInverse',
                                     'shampoo-allVariables-filterFlattening-warmStart',
                                     'shampoo-allVariables-filterFlattening-warmStart-lessInverse',]:
            power_preconditioner = 0.5
        else:
            print('Error: unkown algo for power_preconditioner for ' + params['algorithm'])
            sys.exit()


        H_l_LM_minus_2k = []
        H_l_trace = []

        H = data_['shampoo_H']

        if params['algorithm'] in ['shampoo',
                                     'shampoo-allVariables',
                                     'shampoo-allVariables-warmStart',
                                     'shampoo-allVariables-warmStart-lessInverse',
                                     'shampoo-allVariables-filterFlattening-warmStart',
                                     'shampoo-allVariables-filterFlattening-warmStart-lessInverse',
                                     'shampoo-no-sqrt',
                                     'shampoo-no-sqrt-Fisher']:

            for ii in range(len(H[l][name_variable])):
            
                if params['shampoo_if_coupled_newton']:
            
                    H_l_LM_minus_2k.append(
                        coupled_newton(
                            H[l][name_variable][ii], 
                            len(H[l][name_variable]) / power_preconditioner,
                            epsilon, 
                            device,
                        )
                    )
 
                else:
                    # this is the default of params['shampoo_if_coupled_newton']

                    H_l_ii_LM = H[l][name_variable][ii] + epsilon * torch.eye(H[l][name_variable][ii].shape[0], device=device)

                    try:
                        H_l_U, H_l_S, H_l_V = torch.svd(H_l_ii_LM)

                        if torch.sum(H_l_S != H_l_S) or\
                        torch.sum(H_l_U != H_l_U) or\
                        torch.sum(H_l_V != H_l_V):
                            H_l_U, H_l_S, H_l_V = get_svd_by_cpu(H_l_ii_LM, params)
                    except:
                        H_l_U, H_l_S, H_l_V = get_svd_by_cpu(H_l_ii_LM, params)



                    power_H_l_LM_minus_2k = power_preconditioner / len(H[l][name_variable])

                    H_l_LM_minus_2k.append(
                        torch.mm(
                            torch.mm(
                                H_l_U, 
                                torch.diag(
                                    1/(H_l_S**power_H_l_LM_minus_2k)
                                )
                            ), 
                            H_l_V.t()
                        )
                    )

        else:
            print('Error: unkown algo in svd for ' + params['algorithm'])
            sys.exit()

        data_['shampoo_H_LM_minus_2k'][l][name_variable] = H_l_LM_minus_2k
        data_['shampoo_H_trace'][l][name_variable] = H_l_trace

def shampoo_compute_direction_per_variable(model_grad, l, name_variable, data_, params):
    # model_grad: 1st-order estimate
    
    if get_if_shampoo_update(name_variable, params):
        
        H_l_LM_minus_2k = data_['shampoo_H_LM_minus_2k'][l][name_variable]
        H_l_trace = data_['shampoo_H_trace'][l][name_variable]
        
        delta_l = model_grad[l][name_variable]
        
        delta_l = get_tensor_reshape(delta_l, l, name_variable, params)
        
        
        for ii in range(len(H_l_LM_minus_2k)):
            
            delta_l = torch.tensordot(delta_l, H_l_LM_minus_2k[ii], dims=([0], [0]))
        
        
        delta_l = get_tensor_reshape_back(delta_l, l, name_variable, params)
        
        if params['algorithm'] in ['shampoo',
                                     'shampoo-allVariables',
                                     'shampoo-allVariables-warmStart',
                                     'shampoo-allVariables-warmStart-lessInverse',
                                     'shampoo-allVariables-filterFlattening-warmStart',
                                     'shampoo-allVariables-filterFlattening-warmStart-lessInverse',
                                     'shampoo-no-sqrt',
                                     'shampoo-no-sqrt-Fisher',
                                     'matrix-normal-same-trace',
                                     'matrix-normal-same-trace-warmStart',
                                     'matrix-normal-same-trace-warmStart-noPerDimDamping',
                                     'matrix-normal-same-trace-allVariables',
                                     'matrix-normal-same-trace-allVariables-warmStart',
                                     'matrix-normal-same-trace-allVariables-warmStart-AvgEigDamping',
                                     'matrix-normal-same-trace-allVariables-warmStart-MaxEigDamping',
                                     'matrix-normal-same-trace-allVariables-filterFlattening-warmStart',
                                     'matrix-normal-same-trace-allVariables-KFACReshaping-warmStart',
                                     'matrix-normal-same-trace-allVariables-warmStart-noPerDimDamping',
                                     'matrix-normal-correctFisher-same-trace-allVariables-filterFlattening-warmStart',
                                     'matrix-normal-correctFisher-same-trace-allVariables-filterFlattening-warmStart-lessInverse',
                                     'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping',
                                     'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping-warmStart',
                                     'matrix-normal-correctFisher-same-trace-allVariables-KFACReshaping-warmStart-lessInverse',
                                     'matrix-normal-EF-same-trace-allVariables-filterFlattening-warmStart',]:
            1
        else:
            print('Error: unknown algo in tensor dot for ' + params['algorithm'])
            sys.exit()
            
        return delta_l
    else:
        return model_grad[l][name_variable]

def get_svd_by_cpu(H_l_ii_LM, params):
    device = params['device']

    if_cpu_svd = True
    H_l_cpu = H_l_ii_LM.cpu()

    try:
        H_l_U_cpu, H_l_S_cpu, H_l_V_cpu = torch.svd(H_l_cpu)
    except:
        

        
        if_np_svd = True
        np_H_l_U_cpu, np_H_l_S_cpu, np_H_l_V_cpu = np.linalg.svd(H_l_cpu.data.numpy())
        np_H_l_V_cpu = np.transpose(np_H_l_V_cpu)
        H_l_U_cpu, H_l_S_cpu, H_l_V_cpu = \
        torch.from_numpy(np_H_l_U_cpu),\
        torch.from_numpy(np_H_l_S_cpu), torch.from_numpy(np_H_l_V_cpu)

    H_l_U, H_l_S, H_l_V =\
    H_l_U_cpu.to(device), H_l_S_cpu.to(device), H_l_V_cpu.to(device)

    return H_l_U, H_l_S, H_l_V

def shampoo_update(data_, params):
    
    true_algorithm = params['algorithm']
    model_grad = data_['model_grad_used_torch']
    
    
    
    if params['matrix_name'] in ['Fisher',
                                 'Fisher-correct']:
        model_grad_N1 = data_['model_grad_N2'] # unregularized
    elif params['matrix_name'] == 'None':
        # None is also EF
        model_grad_N1 = data_['model_grad_torch']
    else:
        print('params[matrix_name]')
        print(params['matrix_name'])
        sys.exit()
        

    i = params['i']

    

    # alpha = params['alpha']
    numlayers = params['numlayers']

    device = params['device']
    

    
        
        
    # Step
    delta = []
    for l in range(numlayers):

        for name_variable in data_['model'].layers_weight[l].keys():
            shampoo_kron_matrices_per_variable(model_grad_N1, l, name_variable, data_, params)
        

    




        for name_variable in data_['model'].layers_weight[l].keys():
            shampoo_inversion_per_variable(model_grad_N1, l, name_variable, data_, params)
            


        
        
        
        

        # store the delta
        dict_delta_l = {}
        
        for name_variable in data_['model'].layers_weight[l].keys():
            dict_delta_l[name_variable] = shampoo_compute_direction_per_variable(model_grad, l, name_variable, data_, params)

            
        delta.append(dict_delta_l)



    p = get_opposite(delta)


    
    
    
    data_['p_torch'] = p

    if true_algorithm == 'matrix-normal-LM-momentum-grad':
        params['algorithm'] = true_algorithm

    return data_, params

