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

import warnings
warnings.filterwarnings('error')
# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings("ignore", message="numpy.dtype size changed")


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def coupled_newton(mat_g, p, ridge_epsilon, device):
    # revised from 
    # https://github.com/google-research/google-research/blob/master/scalable_shampoo/jax/shampoo.py#L340
    iter_count=100
    error_tolerance=1e-6
    def coupled_newton_while_loop(cond_fun, body_fun, init_val):
        val = init_val
        while cond_fun(val):
            val = body_fun(val)
        return val
    def coupled_newton_iter_condition(state):
        (i, unused_mat_m, unused_mat_h, unused_old_mat_h, error,
         run_step) = state
        error_above_threshold = ((error > error_tolerance) and run_step)
        return (i < iter_count) and error_above_threshold
    
    def coupled_newton_iter_body(state):
        (i, mat_m, mat_h, unused_old_mat_h, error, unused_run_step) = state
        mat_m_i = (1 - alpha) * identity + alpha * mat_m
        
        # slight different from jax code
        if int(p) == 2:
            mat_pow_2 = torch.matmul(mat_m_i, mat_m_i)
            new_mat_m = torch.matmul(
                mat_pow_2, mat_m,
            )
        elif int(p) == 4:
            mat_pow_2 = torch.matmul(mat_m_i, mat_m_i)
            mat_pow_4 = torch.matmul(mat_pow_2, mat_pow_2)
            new_mat_m = torch.matmul(
                mat_pow_4, mat_m,
            )
        elif int(p) == 6:
            mat_pow_2 = torch.matmul(mat_m_i, mat_m_i)
            mat_pow_4 = torch.matmul(mat_pow_2, mat_pow_2)
            mat_pow_6 = torch.matmul(mat_pow_2, mat_pow_4)
            new_mat_m = torch.matmul(
                mat_pow_6, mat_m,
            )
        else:
            print('p')
            print(p)
            sys.exit()
        new_mat_h = torch.matmul(mat_h, mat_m_i,)
        new_error = torch.max(torch.abs(new_mat_m - identity)).item()
        return (i + 1, new_mat_m, new_mat_h, mat_h, new_error,
                new_error < error * 1.2)
    
    mat_g_size = mat_g.shape[0]
    alpha = -1.0 / p
    identity = torch.eye(mat_g_size, device=device)
        
    
    assert mat_g_size > 1
    damped_mat_g = mat_g + ridge_epsilon * identity
    z = (1 + p) / (2 * torch.linalg.norm(damped_mat_g))
    
    new_mat_m_0 = damped_mat_g * z
    new_error = torch.max(torch.abs(new_mat_m_0 - identity)).item()
    new_mat_h_0 = identity * torch.pow(z, 1.0 / p)
    
    init_state = tuple(
        [0, new_mat_m_0, new_mat_h_0, new_mat_h_0, new_error, True]
    )
    _, mat_m, mat_h, old_mat_h, error, convergence = coupled_newton_while_loop(
        coupled_newton_iter_condition, coupled_newton_iter_body, init_state)
    error = torch.max(torch.abs(mat_m - identity)).item()
    is_converged = convergence
    
    resultant_mat_h = is_converged * mat_h + (1 - is_converged) * old_mat_h
    return resultant_mat_h

def get_block_Fisher(h, a, l, params):
    device = params['device']
    size_minibatch = h[l].size(0)
                
    homo_h_l = torch.cat(
        (h[l].data, torch.ones(size_minibatch, 1, device=device)),
        dim=1
    )
    a_l_grad = size_minibatch * a[l].grad.data
    a_a_T = torch.einsum('ij,ik->ijk', homo_h_l, homo_h_l)
    g_g_T = torch.einsum('ij,ik->ijk', a_l_grad, a_l_grad)

    G_j = torch.zeros(homo_h_l.size(1)*a_l_grad.size(1), homo_h_l.size(1)*a_l_grad.size(1), device=device)
    for dp in range(size_minibatch):
        G_j += torch.kron(a_a_T[dp], g_g_T[dp])
    G_j /= size_minibatch
    return G_j

def Fisher_BD_update(data_, params):
    i = params['i']
    device = params['device']
    model_grad = data_['model_grad_used_torch']
    homo_model_grad = get_homo_grad(model_grad, params)
    delta = []
    for l in range(params['numlayers']):
        if i == 0:
            pass
        else:
            F_l = get_block_Fisher(data_['h_N2'], data_['a_N2'], l, params)
            data_['block_Fisher'][l] *= 0.9
            data_['block_Fisher'][l] += 0.1 * F_l
        
        
        homo_grad_l_vec = torch.reshape(homo_model_grad[l].t(), (-1, 1))
        F_l_LM = data_['block_Fisher'][l] + params['Fisher_BD_damping'] * torch.eye(data_['block_Fisher'][l].size(0), device=device)
        
        homo_delta_l, _ = torch.solve(homo_grad_l_vec, F_l_LM)
        homo_delta_l = torch.reshape(homo_delta_l, homo_model_grad[l].t().size()).t()

        delta_l = from_homo_to_weight_and_bias(homo_delta_l, l, params)
    
        delta.append(delta_l)
        
    p = get_opposite(delta)
    data_['p_torch'] = p
    return data_, params

def get_BFGS_formula_v2(H, s, y, g_k, if_test_mode):
    s = s.data
    y = y.data

    # ger(a, b) = a b^T
    rho_inv = torch.dot(s, y)

    if rho_inv <= 0:
        return H, 1

    rho = 1 / rho_inv
    Hy = torch.mv(H, y)
    
    H_new = H.data +\
    torch.ger((rho**2 * torch.dot(y, Hy) + rho)*s, s) -\
    (torch.ger(rho*s, Hy) + torch.ger(Hy, rho*s))
    
    H = H_new
    return H, 0

def from_unregularized_grad_to_regularized_grad(model_grad_torch, data_, params):
    
    if params['if_regularized_grad']:
        # if you want unregularized grad, you should NOT
        # backward the regularized grad and then subtract the 
        # regularization term. Because the accurate value will be
        # overwhelmed by the noise (numerical error?). 

        # however, if you want regularized grad, you can backward 
        # the unregularized grad and then add the regularization,
        # which will gives you the same value as backward
        # the regularized grad

        if params['tau'] == 0:
            1
        else:
            
            model = data_['model']
            model_grad_torch = get_plus_torch(
        model_grad_torch,
        get_multiply_scalar_no_grad(params['tau'], model.layers_weight)
        )
    else:
        1
        
    return model_grad_torch
    

def get_h_l_unfolded_noHomo_noFlatten(h, l, params):
    # for the use of stride, see _extract_patches in
    # https://github.com/gpauloski/kfac_pytorch/blob/master/kfac/utils.py
    layers_params = params['layers_params']
    assert layers_params[l]['name'] in ['conv', 'conv-no-activation']
     
    padding = layers_params[l]['conv_padding']
    kernel_size = layers_params[l]['conv_kernel_size']
    device = params['device']
    
    stride = layers_params[l]['conv_stride']
    
    assert 2 * padding + 1 == kernel_size
    
    # 2d-conv: a[l]: M * I * |T|, where |T| has two dimensions
        
    # (Take Fashion-MNIST as an example)
    # h[l]: 1000 * 1 * 28 * 28
    # 1000: size of minibatch
    # 1: conv_in_channels
    # 28 * 28: size of input
    # h_l_padded_unfolded: 1000 * 1 * 32 * 32
    # 32 * 32: size of padded input
    h_l_padded = F.pad(
        h[l].data, (padding, padding, padding, padding), "constant", 0
    )
    h_l_padded_unfolded = h_l_padded.unfold(2, kernel_size, stride)
    h_l_padded_unfolded = h_l_padded_unfolded.unfold(3, kernel_size, stride)
    h_l_padded_unfolded = h_l_padded_unfolded.permute(0, 2, 3, 1, 4, 5)
    
    return h_l_padded_unfolded
                                
def get_h_l_unfolded_noHomo(h, l, params):
    # for the use of stride, see _extract_patches in
    # https://github.com/gpauloski/kfac_pytorch/blob/master/kfac/utils.py
    
    layers_params = params['layers_params']
    
    assert layers_params[l]['name'] in ['conv', 'conv-no-activation']
    
    padding = layers_params[l]['conv_padding']
    kernel_size = layers_params[l]['conv_kernel_size']
    device = params['device']
    stride = layers_params[l]['conv_stride']
    
    assert 2 * padding + 1 == kernel_size
    
    # 2d-conv: a[l]: M * I * |T|, where |T| has two dimensions
        
    # (Take Fashion-MNIST as an example)
    # h[l]: 1000 * 1 * 28 * 28
    # 1000: size of minibatch
    # 1: conv_in_channels
    # 28 * 28: size of input
    # h_l_padded_unfolded: 1000 * 1 * 32 * 32
    # 32 * 32: size of padded input
    h_l_padded = F.pad(
        h[l].data, (padding, padding, padding, padding), "constant", 0
    )

    h_l_padded_unfolded = h_l_padded.unfold(2, kernel_size, stride)
    h_l_padded_unfolded = h_l_padded_unfolded.unfold(3, kernel_size, stride)

    h_l_padded_unfolded = h_l_padded_unfolded.permute(0, 2, 3, 1, 4, 5)
    # h_l_padded_unfolded: 1000 * 28 * 28 * 1 * 5 * 5

    h_l_padded_unfolded = h_l_padded_unfolded.flatten(start_dim=3)
    # h_l_padded_unfolded: 1000 * 28 * 28 * 25
    
    return h_l_padded_unfolded
                
def get_h_l_unfolded(h, l, data_, params):
    layers_params = params['layers_params']
    
    assert layers_params[l]['name'] in ['conv',
                                        'conv-no-activation',
                                        'conv-no-bias-no-activation']
    
    padding = layers_params[l]['conv_padding']
    kernel_size = layers_params[l]['conv_kernel_size']
    device = params['device']
    
    stride = layers_params[l]['conv_stride']
    
    assert 2 * padding + 1 == kernel_size
    
    # 2d-conv: a[l]: M * I * |T|, where |T| has two dimensions
        
    # (Take Fashion-MNIST as an example)
    # h[l]: 1000 * 1 * 28 * 28
    # 1000: size of minibatch
    # 1: conv_in_channels
    # 28 * 28: size of input
    # h_l_padded_unfolded: 1000 * 1 * 32 * 32
    # 32 * 32: size of padded input
    h_l_padded = F.pad(
        h[l].data, (padding, padding, padding, padding), "constant", 0
    )

    # h_l_padded_unfolded: 1000 * 1 * 28 * 32 * 5
    # 5: conv_kernel_size
    h_l_padded_unfolded = h_l_padded.unfold(2, kernel_size, stride)


    h_l_padded_unfolded = h_l_padded_unfolded.unfold(3, kernel_size, stride)

    h_l_padded_unfolded = h_l_padded_unfolded.permute(0, 2, 3, 1, 4, 5)
    # h_l_padded_unfolded: 1000 * 28 * 28 * 1 * 5 * 5

    h_l_padded_unfolded = h_l_padded_unfolded.flatten(start_dim=3)
    # h_l_padded_unfolded: 1000 * 28 * 28 * 25


    if 'b' not in data_['model'].layers_weight[l].keys():
        
        pass

    elif params['Kron_BFGS_if_homo']:

        h_homo_ones = torch.ones(
        h_l_padded_unfolded.size(0), h_l_padded_unfolded.size(1), h_l_padded_unfolded.size(2), 1, device=device
    )

        h_l_padded_unfolded = torch.cat(
            (h_l_padded_unfolded, h_homo_ones), 
            dim=3
        )
        # h_l_padded_unfolded: 1000 * 28 * 28 * 26

    else:
        print('error: need to check')
        sys.exit()
    
    
    return h_l_padded_unfolded

def get_A_A_T_v_kfac_v2(v, h, l, params, data_):
    layers_params = params['layers_params']
    device = params['device']
    
    kernel_size = layers_params[l]['conv_kernel_size']
    padding = layers_params[l]['conv_padding']

    if layers_params[l]['name'] == '1d-conv':
        
        print('error: need to change so that it is averaged on minibatch')
        
        sys.exit()
        
        
        
        # 1d-conv: a[l]: M * I * |T|
        h_l_padded = F.pad(h[l].data, (padding, padding), "constant", 0)
        
        # M * J * |T| * |Delta|
        h_l_padded_unfolded = h_l_padded.unfold(2, kernel_size, 1)
        
        h_l_padded_unfolded = h_l_padded_unfolded.permute(0, 2, 1, 3)
        
        h_l_padded_unfolded = h_l_padded_unfolded.flatten(start_dim=2)
        
        
        print('following wrong for kfac?')
        sys.exit()
        
        if params['Kron_BFGS_if_homo']:
            
            h_homo_ones = torch.ones(
            h_l_padded_unfolded.size(0), h_l_padded_unfolded.size(1), 1, device=device
        )
            
            h_l_padded_unfolded = torch.cat(
                (h_l_padded_unfolded, h_homo_ones), 
                dim=2
            )
            
            
        
        test_2_A_j = torch.einsum('sti,stj->ij', h_l_padded_unfolded, h_l_padded_unfolded)
    elif layers_params[l]['name'] in ['conv',
                                      'conv-no-activation']:
        h_l_padded_unfolded_noHomo = data_['h_N2_unfolded_noHomo'][l]
        size_minibatch = h_l_padded_unfolded_noHomo.size(0)
        h_l_padded_unfolded_noHomo_viewed = h_l_padded_unfolded_noHomo.view(-1, h_l_padded_unfolded_noHomo.size(-1))

    
        if params['Kron_BFGS_if_homo']:
            Av = torch.mv(h_l_padded_unfolded_noHomo_viewed, v[:-1]) + v[-1].item()
            Av = torch.cat((torch.mv(h_l_padded_unfolded_noHomo_viewed.t(), Av), torch.sum(Av).unsqueeze(dim=0)))
        else:
            print('error: need to check')
            sys.exit()
        
        Av = Av / size_minibatch
    else:
        print('error: not implemented')
        sys.exit()
    return Av

def get_A_A_T_kfac_v2(h, l, data_, params):
    layers_params = params['layers_params']
    device = params['device']
    
    kernel_size = layers_params[l]['conv_kernel_size']
    padding = layers_params[l]['conv_padding']

    if layers_params[l]['name'] == '1d-conv':
        print('need to make it averaged over minibatch')
        
        # 1d-conv: a[l]: M * I * |T|
        h_l_padded = F.pad(h[l].data, (padding, padding), "constant", 0)
        
        # M * J * |T| * |Delta|
        h_l_padded_unfolded = h_l_padded.unfold(2, kernel_size, 1)
        
        h_l_padded_unfolded = h_l_padded_unfolded.permute(0, 2, 1, 3)
        
        h_l_padded_unfolded = h_l_padded_unfolded.flatten(start_dim=2)
        print('following wrong for kfac?')
        sys.exit()
        
        if params['Kron_BFGS_if_homo']:
            
            h_homo_ones = torch.ones(
            h_l_padded_unfolded.size(0), h_l_padded_unfolded.size(1), 1, device=device
        )
            
            h_l_padded_unfolded = torch.cat(
                (h_l_padded_unfolded, h_homo_ones), 
                dim=2
            )
            
        test_2_A_j = torch.einsum('sti,stj->ij', h_l_padded_unfolded, h_l_padded_unfolded)
  
    elif layers_params[l]['name'] in ['conv',
                                      'conv-no-activation',
                                      'conv-no-bias-no-activation']:
        
        h_l_padded_unfolded = get_h_l_unfolded(h, l, data_, params)
        size_minibatch = h_l_padded_unfolded.size(0)
        
        test_2_A_j = torch.einsum('stli,stlj->ij', h_l_padded_unfolded, h_l_padded_unfolded)
        test_2_A_j = test_2_A_j / size_minibatch

    else:
        print('error: not implemented for ' + layers_params[l]['name'])
        sys.exit()
    
                
    return test_2_A_j

def get_A_A_T_v(v, h, l, params, data_):
    layers_params = params['layers_params']
    device = params['device']
    
    if layers_params[l]['name'] == 'fully-connected':
        size_minibatch = h[l].size(0)
    
        if params['algorithm'] in ['kfac-no-max-no-LM',
                                   'kfac-warmStart-no-max-no-LM',
                                   'kfac-warmStart-lessInverse-no-max-no-LM'] or\
        params['Kron_BFGS_if_homo']:
            
            homo_h_l = torch.cat(
                (h[l], torch.ones(size_minibatch, 1, device=device)),
                dim=1
            )
        elif algorithm in ['Kron-BFGS',
                           'Kron-BFGS-no-norm-gate',
                           'Kron-BFGS-no-norm-gate-momentum-s-y',
                           'Kron-BFGS-no-norm-gate-momentum-s-y-damping',
                           'Kron-BFGS-no-norm-gate-damping',
                           'Kron-BFGS-no-norm-gate-Shiqian-damping',
                           'Kron-BFGS-wrong',
                           'Kron-BFGS-Hessian-action',
                           'Kron-BFGS-wrong-Hessian-action',
                           'Kron-BFGS-LM',
                           'Kron-BFGS-LM-sqrt']:
            homo_h_l = h[l]
        else:
            print('error: not implemented')
            sys.exit()
        Av = torch.mv(homo_h_l.t(), torch.mv(homo_h_l, v)) /size_minibatch

    elif layers_params[l]['name'] in ['1d-conv',
                                      'conv',
                                      'conv-no-activation',
                                      'conv-no-bias-no-activation',]:
        Av = get_A_A_T_v_kfac_v2(v, h, l, params, data_)
    else:
        print('error in get_A_A_T unknown: ' + layers_params[l]['name'])
        sys.exit()
    return Av

def get_A_A_T(h, l, data_, params):
    # return the AVERAGED A_A_T over a minibatch
    layers_params = params['layers_params']
    
    device = params['device']
    
    if layers_params[l]['name'] == 'fully-connected':
        size_minibatch = h[l].size(0)
        if params['Kron_BFGS_if_homo']:
            homo_h_l = torch.cat(
                (h[l], torch.ones(size_minibatch, 1, device=device)),
                dim=1
            )
        elif algorithm in ['Kron-BFGS',
                           'Kron-BFGS-no-norm-gate',
                           'Kron-BFGS-no-norm-gate-momentum-s-y',
                           'Kron-BFGS-no-norm-gate-momentum-s-y-damping',
                           'Kron-BFGS-no-norm-gate-damping',
                           'Kron-BFGS-no-norm-gate-Shiqian-damping',
                           'Kron-BFGS-wrong',
                           'Kron-BFGS-Hessian-action',
                           'Kron-BFGS-wrong-Hessian-action',
                           'Kron-BFGS-LM',
                           'Kron-BFGS-LM-sqrt']:
            homo_h_l = h[l]
        else:
            print('error: not implemented')
            sys.exit()

        A_j = 1/size_minibatch * torch.mm(homo_h_l.t(), homo_h_l).data

    elif layers_params[l]['name'] in ['1d-conv',
                                      'conv',
                                      'conv-no-activation',
                                      'conv-no-bias-no-activation',]:
        test_2_A_j = get_A_A_T_kfac_v2(h, l, data_, params)
        A_j = test_2_A_j
    else:
        print('error in get_A_A_T unknown: ' + layers_params[l]['name'])
        sys.exit()
    return A_j

def get_model_grad(model, params):
    model_grad_torch = []
    for l in range(model.numlayers):
        # model_grad_l = {}
        model_grad_torch_l = {}
        for key in model.layers_weight[l]:
            model_grad_torch_l[key] = copy.deepcopy(model.layers_weight[l][key].grad)
        model_grad_torch.append(model_grad_torch_l)
    return model_grad_torch

def get_statistics(X_train):
    print('\n')
    print('max value:')
    print(np.max(X_train))
    print('min value:')
    print(np.min(X_train))

    print('max of per feature mean:')
    print(np.max(np.mean(X_train, axis=0)))
    print('min of per feature mean:')
    print(np.min(np.mean(X_train, axis=0)))

    print('max of per feature std:')
    print(np.max(np.std(X_train, axis=0)))
    print('min of per feature std:')
    print(np.min(np.std(X_train, axis=0)))
    print('\n')
      
def sample_from_pred_dist(z, params):
    name_loss = params['name_loss']
    N2_index = params['N2_index']

    if name_loss == 'multi-class classification':
        from torch.utils.data import WeightedRandomSampler
        pred_dist_N2 = F.softmax(z[N2_index], dim=1)

        t_mb_pred_N2 = list(WeightedRandomSampler(pred_dist_N2, 1))
        
        
        t_mb_pred_N2 = torch.tensor(t_mb_pred_N2)
        t_mb_pred_N2 = t_mb_pred_N2.squeeze(dim=1)


    elif name_loss == 'binary classification':

        pred_dist_N2 = torch.sigmoid(a[-1][N2_index]).cpu().data.numpy()

        t_mb_pred_N2 = np.random.binomial(n=1, p=pred_dist_N2)

        t_mb_pred_N2 = np.squeeze(t_mb_pred_N2, axis=1)

        print('check if need long')
        sys.exit()

        t_mb_pred_N2 = torch.from_numpy(t_mb_pred_N2).long()



    elif name_loss in ['logistic-regression',
                       'logistic-regression-sum-loss']:
        pred_dist_N2 = torch.sigmoid(z[N2_index]).data

        t_mb_pred_N2 = torch.distributions.Bernoulli(pred_dist_N2).sample()
        t_mb_pred_N2 = t_mb_pred_N2
    
    elif name_loss == 'linear-regression':
        t_mb_pred_N2 = torch.distributions.Normal(loc=z[N2_index], scale=1/2).sample()
    
    elif name_loss == 'linear-regression-half-MSE':
        t_mb_pred_N2 = torch.distributions.Normal(loc=z[N2_index], scale=1).sample()
    else:
        print('Error: sampling not supported.')
        sys.exit()
        
    t_mb_pred_N2 = t_mb_pred_N2.to(params['device'])
    return t_mb_pred_N2

def get_second_order_caches(z, a, h, data_, params):
        
    matrix_name = params['matrix_name']
    model = data_['model']
        
    N1 = params['N1']
    N2 = params['N2']
    
    assert N1 == N2
    
    if matrix_name in ['Fisher',
                       'EF']:
        N2_index = np.random.permutation(N1)
    elif matrix_name == 'Fisher-correct':
        N2_index = list(range(N1))
    else:
        print('matrix_name')
        print(matrix_name)
        
        sys.exit()
    
    X_mb = data_['X_mb']
    assert params['if_different_minibatch'] == False
    if params['if_different_minibatch']:
        print('error: should not reach here')
        sys.exit()
        X_mb_N2, _ = data_['dataset'].train.next_batch(N2)
        X_mb_N2 = torch.from_numpy(X_mb_N2).to(params['device'])
        # if name_dataset == 'MNIST-autoencoder':
            # t_mb = X_mb
    else:
        
        if matrix_name in ['Fisher',
                           'EF']:
        
            X_mb_N2 = X_mb[N2_index]
        elif matrix_name == 'Fisher-correct':
            X_mb_N2 = X_mb
        else:
            print('matrix_name')
            print(matrix_name)
            sys.exit()
            
        
    params['N2_index'] = N2_index
    
    data_['X_mb_N1'] = X_mb
    data_['X_mb_N2'] = X_mb_N2
    if matrix_name == 'EF':
        if params['if_different_minibatch']:
            print('error: need to check for different minibatch when EF')
            sys.exit()
        else:
            t_mb = data_['t_mb']
            data_['t_mb_pred_N2'] = t_mb[N2_index]
            data_['mean_a_grad_N2'] = [torch.mean(N2 * (a_l.grad)[N2_index], dim=0).data for a_l in a]
            data_['h_N2'] = [h_l[N2_index].data if len(h_l) else [] for h_l in h]
            data_['mean_a_N2'] = [torch.mean(a_l[N2_index], dim=0).data for a_l in a]
    elif matrix_name == 'Fisher':
        
        if params['i'] % params['shampoo_update_freq'] != 0:
            return data_

        t_mb_pred_N2 = sample_from_pred_dist(z, params)
        data_['t_mb_pred_N2'] = t_mb_pred_N2
        
        z, a_N2, h_N2 = model.forward(X_mb_N2)
        reduction = 'mean'
        loss = get_loss_from_z(model, z, t_mb_pred_N2, reduction) # this is unregularized loss
        model.zero_grad()
        loss.backward()

        l = -1
        for a_l in a_N2:
            l += 1

        data_['a_grad_N2'] = [N2 * (a_l.grad) for a_l in a_N2]
        data_['h_N2'] = h_N2
        data_['a_N2'] = a_N2

        if params['if_model_grad_N2']:
            # this is unregularized grad
            data_['model_grad_N2'] = get_model_grad(model, params)
            
    elif matrix_name == 'Fisher-correct':        
        if params['algorithm'] in params['list_algorithm_shampoo'] and params['i'] % params['shampoo_update_freq'] != 0:
            return data_
        
        if params['algorithm'] in params['list_algorithm_kfac'] and params['i'] % params['kfac_cov_update_freq'] != 0:
            return data_

        a_N2 = a
        h_N2 = h

        t_mb_pred_N2 = sample_from_pred_dist(z, params)
        data_['t_mb_pred_N2'] = t_mb_pred_N2
        reduction = 'mean'
        loss = get_loss_from_z(model, z, t_mb_pred_N2, reduction) # this is unregularized loss
        model.zero_grad()
        loss.backward()
        #loss.backward(retain_graph=True)
        if params['algorithm'] in ['Fisher-BD',
                                   'kfac-correctFisher-warmStart-no-max-no-LM',
                                   'kfac-correctFisher-warmStart-NoMaxNoSqrt-no-LM',
                                   'kfac-correctFisher-warmStart-lessInverse-no-max-no-LM',
                                   'kfac-correctFisher-warmStart-lessInverse-NoMaxNoSqrt-no-LM',]:
            data_['h_N2'] = h_N2
            data_['a_N2'] = a_N2

        if params['if_model_grad_N2']:
            # this is unregularized grad
            data_['model_grad_N2'] = get_model_grad(model, params)

    elif matrix_name == 'GN':
        data_['z_N2'] = a[-1][N2_index]

        m_L = data_['model'].layersizes[-1]
        params['m_L'] = m_L
        N2 = params['N2']

        list_a = [ [] for i in range(m_L) ]

        a_grad_momentum = []
        for l in range(model.numlayers):
            a_grad_momentum.append(
                torch.zeros(m_L, N2, data_['model'].layersizes[l+1], device=params['device']))

        for i in range(m_L):
            z, a, h = model.forward(X_mb[N2_index])
            fake_loss = torch.sum(z[:, i])

            test_time_zero = time.time()
            model.zero_grad()
            fake_loss.backward()
            for l in range(model.numlayers):
                a_grad_momentum[l][i] = a[l].grad



        h_momentum = [hi.data for hi in h]

        for l in range(model.numlayers):
            a_grad_momentum[l] = a_grad_momentum[l].permute(1, 0, 2)

        data_['a_grad'] = a_grad_momentum
        data_['h'] = h_momentum
    else:
        print('Error: unknown matrix name for ' + matrix_name)
        sys.exit()
            
    return data_

def get_erf(p):
    sign_ = []
    for l in range(len(p)):
        sign_l = {}
        for key in p[l]:
            sign_l[key] = torch.erf(p[l][key])
        sign_.append(sign_l)
    return sign_

def get_erf_approx(p):
    sign_ = []
    for l in range(len(p)):
        sign_l = {}
        for key in p[l]:
            sign_l[key] = 1 / (1 + p[l][key]**2)
        sign_.append(sign_l)
    return sign_

def get_reciprocal(p):
    sign_ = []
    for l in range(len(p)):
        sign_l = {}
        for key in p[l]:
            sign_l[key] = 1 / p[l][key]
        sign_.append(sign_l)
    return sign_

def get_sign(p):
    sign_ = []
    for l in range(len(p)):
        sign_l = {}
        for key in p[l]:
            sign_l[key] = np.sign(p[l][key])
        sign_.append(sign_l)
    return sign_

def get_sign_torch(p):
    sign_ = []
    for l in range(len(p)):
        sign_l = {}
        for key in p[l]:
            sign_l[key] = torch.sign(p[l][key])
        sign_.append(sign_l)
    return sign_

def get_zero_torch(params):
    layers_params = params['layers_params']
    device = params['device']
    
    delta = []
    for l in range(len(layers_params)):
        delta_l = {}
        if layers_params[l]['name'] == 'fully-connected':
            delta_l['W'] = torch.zeros(layers_params[l]['output_size'], layers_params[l]['input_size'], device=device)
            delta_l['b'] = torch.zeros(layers_params[l]['output_size'], device=device)
        elif layers_params[l]['name'] in ['conv',
                                          'conv-no-activation',
                                          'conv-no-bias-no-activation']:
            delta_l['W'] = torch.zeros(layers_params[l]['conv_out_channels'],
                                     layers_params[l]['conv_in_channels'],
                                     layers_params[l]['conv_kernel_size'],
                                     layers_params[l]['conv_kernel_size'], device=device)
            if layers_params[l]['name'] in ['conv',
                                            'conv-no-activation',]:
                delta_l['b'] = torch.zeros(layers_params[l]['conv_out_channels'], device=device)
        elif layers_params[l]['name'] == '1d-conv':
            delta_l['W'] = torch.zeros(layers_params[l]['conv_out_channels'],
                                     layers_params[l]['conv_in_channels'],
                                     layers_params[l]['conv_kernel_size'],
                                       device=device)
            delta_l['b'] = torch.zeros(layers_params[l]['conv_out_channels'], device=device)
        elif layers_params[l]['name'] == 'BN':
            delta_l['W'] = torch.zeros(layers_params[l]['num_features'], device=device)
            
            delta_l['b'] = torch.zeros(layers_params[l]['num_features'], device=device)
            
        else:
            print('Error: layers unsupported when get zero for ' + layers_params[l]['name'])
            sys.exit()
        delta.append(delta_l)
        
    return delta

def get_full_grad(model, x, t, params):
    N1 = params['N1']
    reduction = 'mean'
    i = 0
    while (i+1) * N1 <= len(x):
        loss = get_regularized_loss_from_x(model, x[i*N1: (i+1)*N1], t[i*N1: (i+1)*N1], reduction)
        model.zero_grad()
        loss.backward()
        grad_i = get_model_grad(model, params)
        if i == 0:
            full_grad = grad_i
        else:
            full_grad = get_plus_torch(full_grad, grad_i)
        i += 1
    full_grad = get_multiply_scalar(1./i, full_grad)

    
    
    return full_grad

def get_regularized_loss_from_x_no_grad(model, x, t, reduction, tau):
    with torch.no_grad():
        z, _, _ = model.forward(x)
    return get_regularized_loss_from_z(model, z, t, reduction, tau)

def get_acc_whole_dataset(model, params, x, np_t):
    N1 = params['N1']
    N1 = np.minimum(N1, len(x))
    
    i = 0
    list_acc = []
    model.eval()
    
    while i + N1 <= len(x):
        with torch.no_grad():
            list_acc.append(get_acc_from_x(model, params, x[i: i+N1], np_t[i: i+N1]))
            # z, _, _ = model.forward(x[i: i+N1])

        i += N1
    model.train()
    return sum(list_acc) / len(list_acc)

def get_regularized_loss_from_x_whole_dataset(model, x, t, reduction, params):
    N1 = params['N1']
    device = params['device']
    
    
    i = 0
    list_loss = []
    model.eval()
    
    
    while i + N1 <= len(x):
        with torch.no_grad():
            z, _, _ = model.forward(torch.from_numpy(x[i: i+N1]).to(device))
        
        list_loss.append(
            get_regularized_loss_from_z(model, z, torch.from_numpy(t[i: i+N1]).to(device), reduction).item())
        

        
        i += N1
        

        
        
    model.train()
    return sum(list_loss) / len(list_loss)

def get_regularized_loss_and_acc_from_x_whole_dataset_with_generator(model, generator, reduction, params):
    N1 = params['N1']    
    
    device = params['device']
    
    list_loss = []
    list_unregularized_loss = []
    list_acc = []
    
    model.eval()
    
    for (X_mb, t_mb) in generator:
        X_mb, t_mb = X_mb.to(device), t_mb.to(device)
        
        if len(X_mb) != N1:
            break
        z, _, _ = model.forward(X_mb)
            
        
        loss_i, unregularized_loss_i =\
        get_regularized_loss_from_z(model, z, t_mb, reduction, params['tau'])
        
        list_loss.append(loss_i.item())
        
        list_unregularized_loss.append(unregularized_loss_i.item())
        
        list_acc.append(
            get_acc_from_z(model, params, z, t_mb))
        
    model.train()
    
    return sum(list_loss) / len(list_loss), sum(list_unregularized_loss) / len(list_unregularized_loss), sum(list_acc) / len(list_acc)

def get_regularized_loss_and_acc_from_x_whole_dataset(model, x, t, reduction, params):
    N1 = params['N1']
    N1 = np.minimum(N1, len(x))
    
    
    i = 0
    device = params['device']
    
    list_loss = []
    list_unregularized_loss = []
    list_acc = []
    
    model.eval()
    
    while i + N1 <= len(x):
        
        X_mb = torch.from_numpy(x[i: i+N1]).to(device)
        t_mb = torch.from_numpy(t[i: i+N1]).to(device)
        
        z, _, _ = model.forward(X_mb)
            
        
        
        loss_i, unregularized_loss_i =\
        get_regularized_loss_from_z(model, z, t_mb, reduction, params['tau'])
        
        list_loss.append(loss_i.item())
        
        list_unregularized_loss.append(unregularized_loss_i.item())
        
        list_acc.append(
            get_acc_from_z(model, params, z, t_mb))
        
        i += N1
    model.train()
    
    return sum(list_loss) / len(list_loss), sum(list_unregularized_loss) / len(list_unregularized_loss), sum(list_acc) / len(list_acc)

def get_loss_from_x(model, x, t, reduction):
    # with torch.no_grad():
    z, _, _ = model.forward(x)
    return get_loss_from_z(model, z, t, reduction)

def get_acc_from_x(model, params, x, np_t):
    
    z, _ , _= model.forward(x)

    return get_acc_from_z(model, params, z, np_t)

def get_regularized_loss_from_z(model, z, t, reduction, tau):
    unregularized_loss = get_loss_from_z(model, z, t, reduction)
    
    loss = unregularized_loss + 0.5 * tau *\
    get_dot_product_torch(model.layers_weight, model.layers_weight)
    
    return loss, unregularized_loss

def get_loss_from_z(model, z, t, reduction):
    if model.name_loss == 'multi-class classification':
        # since this is multi-calss cross entropy loss,
        # there is no ambiguity between "average" and "sum"
        
        # common bug: size of z does not match max of t
        loss = F.cross_entropy(z, t, reduction = reduction)


    elif model.name_loss == 'binary classification':
        # since this is binary-calss cross entropy loss,
        # there is no ambiguity between "average" and "sum"
        

        loss = torch.nn.BCEWithLogitsLoss(reduction = reduction)(z, t.float().unsqueeze_(1))


        if reduction == 'none':
            loss = loss.squeeze(1)

    elif model.name_loss == 'logistic-regression':
        # use of cross-entropy endorsed by Hinton and Salakhutdinov 2006 and 
        # Hessian-free code
        
        # if reduction == 'mean', the following gives the sum of loss / #data / #feature
        # i.e. for a single data point, the loss on all pixels are averaged      
        

        if reduction == 'none':
            loss = torch.nn.BCEWithLogitsLoss(reduction = reduction)(z, t.float())
            loss = torch.sum(loss, dim=1)
        elif reduction == 'mean':
            loss = torch.nn.BCEWithLogitsLoss(reduction = 'sum')(z, t.float())
            
            loss = loss / z.size(0) / z.size(1)
            
        elif reduction == 'sum':
            loss = torch.nn.BCEWithLogitsLoss(reduction = reduction)(z, t.float())
            
    elif model.name_loss == 'logistic-regression-sum-loss':
        # use of cross-entropy endorsed by Hinton and Salakhutdinov 2006 and 
        # Hessian-free code
        
        # if reduction == 'mean', the following gives the sum of loss / #data
        # i.e. for a single data point, the loss on all pixels are summed      
        

        if reduction == 'none':
            loss = torch.nn.BCEWithLogitsLoss(reduction = reduction)(z, t.float())
            loss = torch.sum(loss, dim=1)
        elif reduction == 'mean':
            loss = torch.nn.BCEWithLogitsLoss(reduction = 'sum')(z, t.float())
            loss = loss / z.size(0)
            
        elif reduction == 'sum':
            loss = torch.nn.BCEWithLogitsLoss(reduction = reduction)(z, t.float())

    elif model.name_loss == 'linear-regression-half-MSE':
        
        if reduction == 'mean':
            loss = torch.nn.MSELoss(reduction = 'sum')(z, t) / 2
            
            
            # only average by mini-batch, not by #feature
            loss = loss / z.size(0)
        elif reduction == 'none':
            loss = torch.nn.MSELoss(reduction = 'none')(z, t) / 2
            loss = torch.sum(loss, dim=1)
        else:
            print('reduction')
            print(reduction)
            print('error: unknown reduction')
            sys.exit()
            
            sys.exit()
            
    elif model.name_loss == 'linear-regression':
        
        if reduction == 'mean':
            loss = torch.nn.MSELoss(reduction = 'sum')(z, t)
            loss = loss / z.size(0)
            
        elif reduction == 'none':
            loss = torch.nn.MSELoss(reduction = 'none')(z, t)
            loss = torch.sum(loss, dim=1)
        else:
            print('reduction')
            print(reduction)
            print('error: unknown reduction')
            sys.exit()
            
            sys.exit()
    
    else:
        print('Error: loss function not specified.')
        sys.exit()
    
    return loss

def get_acc_from_z(model, params, z, torch_t):
    
    if model.name_loss == 'multi-class classification':
        y = z.argmax(dim=1)
        acc = torch.mean((y == torch_t).float())
    
    elif model.name_loss == 'binary classification':
        print('np_t should be tensor')
        sys.exit()
        z_1 = torch.sigmoid(z)
        y = (z_1 > 0.5)
        y = y[:, 0]
        acc = np.mean(y.cpu().data.numpy() == np_t)
    elif model.name_loss in ['logistic-regression',
                             'logistic-regression-sum-loss']:

        z_sigmoid = torch.sigmoid(z)
    
        criterion = nn.MSELoss(reduction = 'mean')
        acc = criterion(z_sigmoid, torch_t)
        
    elif model.name_loss in ['linear-regression',
                             'linear-regression-half-MSE']:
        acc = nn.MSELoss(reduction = 'mean')(z, torch_t)

    else:
        print('Error: unkwoen name_loss')
        sys.exit()
    acc = acc.item()
    
    return acc
    
def compute_sum_J_transpose_V_backp(v, data_, params):
    # use backpropagation
    algorithm = params['algorithm']
    N2 = params['N2']
    numlayers = params['numlayers']
    
    model = data_['model']
    X_mb_N2 = data_['X_mb_N2']
    
    
    
    z, _, _ = model.forward(X_mb_N2)
    
    if params['matrix_name'] == 'Fisher':
                         
                         
        t_mb_N2 = data_['t_mb_pred_N2'] # note that t_mb will be correspond to either EF or Fisher
        reduction = 'none'
        loss = get_loss_from_z(model, z, t_mb_N2, reduction)
        loss = torch.dot(loss, v)
        
    elif algorithm == 'SMW-GN':
        
        m_L = params['m_L']
        
        v = v.view(N2, m_L)
        
        loss = torch.sum(z * v.data)
        
    else:
        print('Error! 1500')
        sys.exit()


    model.zero_grad()

    loss.backward()

    delta = get_model_grad(model, params)
    model.zero_grad()
    
    
    return delta
 
def get_D_t(data_, params):
    algorithm = params['algorithm'] 
    N2 = params['N2']
    numlayers = params['numlayers']
    
    if algorithm == 'SMW-Fisher-different-minibatch' or\
    algorithm == 'SMW-Fisher':
        a_grad_momentum = data_['a_grad_N2']
        h_momentum = data_['h_N2']
    
        lambda_ = params['lambda_']
        
        print('error: should change to device')
        sys.exit()

        # compute D_t 
        D_t = lambda_ * torch.eye(N2).to(params['device'])

        for l in range(numlayers):
            # @ == torch.mm in this case, speed also similar
            # D_t += 1 / N2 * (a_grad_momentum[l] @ a_grad_momentum[l].t()) * (h_momentum[l] @ h_momentum[l].t() + 1)
            D_t += 1 / N2 * torch.mm(a_grad_momentum[l], a_grad_momentum[l].t()) *\
            (torch.mm(h_momentum[l], h_momentum[l].t()) + 1)

        # D_t = D_t.cpu().data.numpy()
        torch_D_t = D_t
    elif algorithm == 'SMW-Fisher-momentum' or\
    algorithm == 'SMW-Fisher-D_t-momentum' or\
    algorithm == 'GI-Fisher':
        a_grad_momentum = data_['a_grad_momentum']
        h_momentum = data_['h_momentum']
    
        lambda_ = params['lambda_']
        
        

        # compute D_t 
        D_t = lambda_ * torch.eye(N2, device=params['device'])

        for l in range(numlayers):
            # @ == torch.mm in this case, speed also similar
            # D_t += 1 / N2 * (a_grad_momentum[l] @ a_grad_momentum[l].t()) * (h_momentum[l] @ h_momentum[l].t() + 1)
            D_t += 1 / N2 * torch.mm(a_grad_momentum[l], a_grad_momentum[l].t()) *\
            (torch.mm(h_momentum[l], h_momentum[l].t()) + 1)

        # D_t = D_t.cpu().data.numpy()
        torch_D_t = D_t


    elif algorithm == 'SMW-Fisher-momentum-D_t-momentum':

        a_grad_momentum = data_['a_grad_for_D_t']
        h_momentum = data_['h_for_D_t']
    
        lambda_ = params['lambda_']
        
        

        # compute D_t 
        D_t = lambda_ * torch.eye(N2)
    
        for l in range(numlayers):
        
            D_t += 1 / N2 * torch.mm(a_grad_momentum[l], a_grad_momentum[l].t()) *\
            (torch.mm(h_momentum[l], h_momentum[l].t()) + 1)
        
        D_t = D_t.data.numpy()
    elif algorithm == 'SMW-GN':
        # a_grad[l]: N2, m_L, m_l
        
    
        GN_cache = data_['GN_cache']
        h = GN_cache['h']
        a_grad = GN_cache['a_grad']
        
        m_L = params['m_L']
        lambda_ = params['lambda_']
        
        
        # D_t = np.zeros((m_L * N2, m_L * N2))
        torch_D_t = torch.zeros(m_L * N2, m_L * N2, device=params['device'])
        
        
        
        model = data_['model']
        
        for l in range(numlayers):
            torch_a_grad_1 = a_grad[l]
            torch_h_l = h[l]
            
            torch_permuted_a_grad_l =\
            torch_a_grad_1.permute(1,0,2).view(m_L * N2, data_['model'].layersizes[l+1])
            
            

            
            # h_l_h_l_t = np.matmul(h_l, np.transpose(h_l)) + 1 # + 1 for b
            torch_h_l_h_l_t = torch.mm(torch_h_l, torch_h_l.t()) + 1 # + 1 for b
            
            # h_kron = np.kron(h_l_h_l_t, np.ones((m_L, m_L)))
            torch_h_kron = get_kronecker_torch(
                torch_h_l_h_l_t, torch.ones(m_L, m_L, device=params['device']))
        
            torch_D_t += torch.mul(
                torch_h_kron, torch.mm(torch_permuted_a_grad_l, torch_permuted_a_grad_l.t()))
        
    
        if model.name_loss == 'binary classification':
            # print('need to check')
            # sys.exit()

            torch_D_t = 1 / N2 * torch_D_t

            torch_D_t = torch_D_t + lambda_ * torch.eye(m_L * N2, device=params['device'])
        elif model.name_loss == 'multi-class classification':
            # D_t = get_JH(D_t, data_, params)
            torch_D_t = get_JH(torch_D_t, data_, params)

            torch_D_t = 1 / N2 * torch_D_t

            torch_D_t = torch_D_t + lambda_ * torch.eye(m_L * N2, device=params['device'])
        else:
            
            print('Error: unknown loss')
            sys.exit() 
    else:
        print('Error! 1501')
        sys.exit()
    return torch_D_t

def get_kronecker_torch(A, B):
    return torch.einsum("ab,cd->acbd", A, B).view(A.size(0)*B.size(0),  A.size(1)*B.size(1))

def get_JH(torch_D_t, data_, params):
    y = data_['y']
    N2 = params['N2']
    m_L = params['m_L']
    diag_y = y.view(m_L * N2)
    
    diag_y = diag_y.repeat(N2 * m_L, 1)
    
    # D_t_1 = D_t * diag_y.cpu().data.numpy()
    torch_D_t_1 = torch_D_t * diag_y
    
    # D_t_2
    
    # D_t_3 = np.zeros((m_L * N2, m_L * N2))
    torch_D_t_3 = torch.zeros(m_L * N2, m_L * N2, device=params['device'])
    for i in range(N2):
        torch_y_i = torch.unsqueeze(y[i], -1)
        torch_D_t_3[:, i * m_L : (i+1) * m_L] =\
        torch.mm(torch.mm(torch_D_t[:, i * m_L : (i+1) * m_L], torch_y_i), torch_y_i.t())
    
    torch_D_t = torch_D_t_1 - torch_D_t_3

    return torch_D_t
    

def get_H(data_, params):
    model = data_['model']
    
    if model.name_loss == 'multi-class classification':
        print('wrong')
        sys.exit()
        
        # N2_index = params['N2_index']
        m_L = params['m_L']
        N2 = params['N2']

        z_N2 = data_['z_N2']
        

        z_data = z_N2.data.numpy()
        
        H = np.zeros((N2, m_L, m_L))
        for i in range(N2):
            H[i] -= np.outer(z_data[i], z_data[i])
            H[np.diag_indices(m_L)] += z_data[i]
    elif model.name_loss == 'binary classification':
        y = data_['y']
        y = y.data.numpy()
        tilde_y = y * (1-y)
        H = np.diag(y)
    else:
        sys.exit()
    
    
    return H

def get_HV(torch_V, data_, params):
    model = data_['model']

    # y = data_['y']
    # y = y.cpu().data.numpy()
    torch_y = data_['y']

    # V = torch_V.cpu().data.numpy()

    if model.name_loss == 'multi-class classification':
    
        N2 = params['N2']
        m_L = params['m_L']
        
        torch_V = torch_V.view(N2, m_L)
        # V = np.reshape(V, (N2, m_L))
        
        
        
        
        
        # HV = np.multiply(y, V)
        torch_HV = torch.mul(torch_y, torch_V)
        
        
        # sum_HV = np.sum(HV, 1) # length N2
        torch_sum_HV = torch.sum(torch_HV, dim=1) # length N2
        
        
        # HV = HV - sum_HV[:, None] * y
        torch_HV = torch_HV - torch_sum_HV[:, None] * torch_y

        # print('torch.max(torch_HV - torch.from_numpy(HV).cuda())')
        # print(torch.max(torch_HV - torch.from_numpy(HV).cuda()))
        # print('torch.min(torch_HV - torch.from_numpy(HV).cuda())')
        # print(torch.min(torch_HV - torch.from_numpy(HV).cuda()))
            
        
        # HV = np.reshape(HV, m_L * N2)
        torch_HV = torch_HV.view(m_L * N2)

    elif model.name_loss == 'binary classification':
        y = np.squeeze(y, axis=1)
        # print('test no squeeze')

        # print('V.shape')
        # print(V.shape)

        # print('np.multiply(y, V).shape')
        # print(np.multiply(y, V).shape)

        HV = np.multiply(y, V)
    else:
        sys.exit()
    
    return torch_HV

def compute_JV(V, data_, params):
    algorithm = params['algorithm']
    
    numlayers = params['numlayers']
    N2 = params['N2']
    
    if params['matrix_name'] == 'Fisher':

    
        v = torch.zeros(N2)
        if params['if_gpu']:
            v = v.cuda()

        # a_N2 = data_['a_N2']
        a_grad_N2 = data_['a_grad_N2']
        h_N2 = data_['h_N2']
    
        for l in range(numlayers):
    
            torch_V_W_l = V[l]['W']
            torch_V_b_l = V[l]['b']
            
            v += torch.sum(torch.mm(a_grad_N2[l], torch_V_W_l) * h_N2[l], dim = 1)
        
            
            v += torch.sum(torch_V_b_l * a_grad_N2[l], dim=1)
        v = v.data
    
    elif algorithm == 'SMW-GN':
        
        GN_cache = data_['GN_cache']
        
        m_L = params['m_L']
        
        a_grad = GN_cache['a_grad'] # a_grad[l]: N2, m_L, m_l
        h = GN_cache['h']

        v = torch.zeros(N2, m_L, device=params['device'])
        for l in range(numlayers):
            a_grad_l = a_grad[l]
            v += torch.sum(
                torch.matmul(a_grad_l, V[l]['W']) * h[l][:, None, :], dim=2)
            
            # v += torch.matmul(a_grad_l, torch.from_numpy(V['b'][l]))
            v += torch.matmul(a_grad_l, V[l]['b'])

        v = v.permute(1, 0)
        # v = np.swapaxes(v,0,1)

        
        v = torch.reshape(v, (m_L * N2,))
        # v = np.reshape(v, (m_L * N2,))

        # print('time for parallel')
        # print(time.process_time() - test_start_time_cpu)

        # print('test parallel')
        
    else:
        print('Error! 1502')
        sys.exit()
    
    return v

def get_cache_momentum(data_, params):
    algorithm = params['algorithm']
    N2 = params['N2']
    
    if algorithm == 'SMW-GN':

        a_grad_momentum = data_['a_grad']
        h_momentum = data_['h']
        
        GN_cache = {}
        GN_cache['a_grad'] = a_grad_momentum
        GN_cache['h'] = h_momentum
        
        
        data_['GN_cache'] = GN_cache

    elif algorithm == 'SMW-Fisher-batch-grad-momentum-exponential-decay' or\
    algorithm == 'SMW-Fisher-batch-grad-momentum':

        # N_iters = 30
        N_iters = params['N_iters']

        batch_grads_i = data_['model_regularized_grad_N2']
        batch_grads_test = data_['batch_grads_test']
        if len(data_['batch_grads']) == 0:

            

            batch_grads_test = {}
            batch_grads_test['W'] = []
            batch_grads_test['b'] = []
            for l in range(data_['model'].numlayers):
                batch_grads_test['W'].append(batch_grads_i['W'][l][np.newaxis,:])
                batch_grads_test['b'].append(batch_grads_i['b'][l][np.newaxis,:])

            
        elif len(data_['batch_grads']) < N_iters:
            for l in range(data_['model'].numlayers):
                # print('batch_grads_test')
                # print(batch_grads_test)
                
                batch_grads_test['W'][l] = np.concatenate(
                    (batch_grads_test['W'][l], batch_grads_i['W'][l][np.newaxis,:]), axis=0)
                batch_grads_test['b'][l] = np.concatenate(
                    (batch_grads_test['b'][l], batch_grads_i['b'][l][np.newaxis,:]), axis=0)
            
        else:
            # print('params[i]')
            # print(params['i'])

            # print('params[i] % N_iters')
            # print(params['i'] % N_iters)

            # replace_index = params['i'] % N_iters
            replace_index = 0
            swap_indices = np.asarray(
                list(range(replace_index+1, N_iters)) + list(range(0, replace_index+1)))

            replace_index = params['i'] % N_iters
            for l in range(data_['model'].numlayers):
                # batch_grads_test['W'][l][replace_index] = batch_grads_i['W'][l]
                # batch_grads_test['b'][l][replace_index] = batch_grads_i['b'][l]
                batch_grads_test['W'][l][0] = batch_grads_i['W'][l]
                batch_grads_test['b'][l][0] = batch_grads_i['b'][l]
                batch_grads_test['W'][l] = batch_grads_test['W'][l][swap_indices]
                batch_grads_test['b'][l] = batch_grads_test['b'][l][swap_indices]

            
        data_['batch_grads_test'] = batch_grads_test

        if len(data_['batch_grads']) == N_iters:
            data_['batch_grads'].popleft()
            data_['batch_grads_a_grad'].popleft()
            data_['batch_grads_h'].popleft()
        elif len(data_['batch_grads']) > N_iters:
            print('Error: len > N_iters')
            sys.exit()

        data_['batch_grads'].append(data_['model_regularized_grad_N2'])
        data_['batch_grads_a_grad'].append(data_['a_grad_N2'])
        data_['batch_grads_h'].append(data_['h_N2'])
        
    else:
        a_grad_N2 = data_['a_grad_N2']
        h_N2 = data_['h_N2']
    
        N1 = params['N1']
        i = params['i']
        numlayers = params['numlayers']
    # Update running estimates
        if algorithm == 'SMW-Fisher-momentum':
            
            a_grad_momentum = data_['a_grad_momentum']
            h_momentum = data_['h_momentum']
            
            rho = min(1 - 1/(i+1), 0.95)
        
            for l in range(numlayers):
                a_grad_momentum[l] = rho * a_grad_momentum[l] + (1-rho) * a_grad_N2[l]
                h_momentum[l] = rho * h_momentum[l] + (1-rho) * h_N2[l]
        elif algorithm == 'SMW-Fisher-momentum-D_t-momentum':
            
            a_grad_momentum = data_['a_grad_momentum']
            h_momentum = data_['h_momentum']
            
            rho = min(1 - 1/(i+1), 0.95)
        
            for l in range(numlayers):
                a_grad_momentum[l] = rho * a_grad_momentum[l] + (1-rho) * a_grad_N2[l]
                h_momentum[l] = rho * h_momentum[l] + (1-rho) * h_N2[l]
                
            a_grad_for_D_t = []
            h_for_D_t = []
            for l in range(numlayers):
                a_grad_for_D_t.append(a_grad_N2[l])
                h_for_D_t.append(h_N2[l])
                
            data_['a_grad_for_D_t'] = a_grad_for_D_t
            data_['h_for_D_t'] = h_for_D_t
        
        elif algorithm == 'Fisher-block' or\
        algorithm == 'SMW-Fisher-D_t-momentum' or\
        algorithm == 'GI-Fisher' or\
        algorithm == 'SMW-Fisher-BD':
            print('Error: no need to get momentum.')
            sys.exit()
            # a_grad_momentum = []
            # h_momentum = []
            # for l in range(numlayers):
                # a_grad_momentum.append(a_grad_N2[l])
                # h_momentum.append(h_N2[l])
            
    
        
        
        else:
            print('Error! 1503')
            sys.exit()
        
        data_['a_grad_momentum'] = a_grad_momentum
        data_['h_momentum'] = h_momentum


    return data_

def get_subtract(model_grad, delta, params):
    diff_p = get_zero(params)
    for l in range(params['numlayers']):
        for key in diff_p[l]:
            diff_p[l][key] = np.subtract(model_grad[l][key], delta[l][key])
    return diff_p

def get_subtract_torch(model_grad, delta):
    # diff_p = get_zero(params)
    diff_p = []
    for l in range(len(model_grad)):
        diff_p_l = {}
        for key in model_grad[l]:
            diff_p_l[key] = torch.sub(model_grad[l][key], delta[l][key])
        diff_p.append(diff_p_l)
    return diff_p

def get_plus(model_grad, delta):
    sum_p = []
    for l in range(len(model_grad)):
        sum_p_l = {}
        for key in model_grad[l]:
            sum_p_l[key] = np.add(model_grad[l][key], delta[l][key])
        sum_p.append(sum_p_l)
    return sum_p

def get_plus_torch(model_grad, delta):
    sum_p = []
    for l in range(len(model_grad)):
        sum_p_l = {}
        for key in model_grad[l]:
            sum_p_l[key] = model_grad[l][key] + delta[l][key]
        sum_p.append(sum_p_l)
    return sum_p

def get_if_nan(p):
    for l in range(len(p)):
        for key in p[l]:
            # print('p[l][key] != p[l][key]')
            # print(p[l][key] != p[l][key])
            if torch.sum(p[l][key] != p[l][key]):
                return True
    return False

def get_torch_tensor(p, params):
    p_torch = []
    for l in range(len(p)):
        p_torch_l = {}
        for key in p[l]:
            p_torch_l[key] = torch.from_numpy(p[l][key]).to(params['device'])
        p_torch.append(p_torch_l)
    return p_torch

def get_plus_scalar(alpha, model_grad):
    sum_p = []

    # numlayers = params['numlayers']
    for l in range(len(model_grad)):
        sum_p_l = {}
        for key in model_grad[l]:
            sum_p_l[key] = model_grad[l][key] + alpha
        sum_p.append(sum_p_l)
    return sum_p

def get_multiply_scalar(alpha, delta):
    alpha_p = []
    for l in range(len(delta)):
        alpha_p_l = {}
        for key in delta[l]:
            alpha_p_l[key] = alpha * delta[l][key]
        alpha_p.append(alpha_p_l)
    return alpha_p

def get_multiply_scalar_no_grad(alpha, delta):
    alpha_p = []
    for l in range(len(delta)):
        alpha_p_l = {}
        for key in delta[l]:
            alpha_p_l[key] = alpha * delta[l][key].data
        alpha_p.append(alpha_p_l)
    return alpha_p

def get_multiply_scalar_blockwise(alpha, delta, params):
    alpha_p = []
    for l in range(params['numlayers']):
        alpha_p_l = {}
        for key in delta[l]:
            alpha_p_l[key] = alpha[l] * delta[l][key]
        alpha_p.append(alpha_p_l)
    return alpha_p

def get_multiply_torch(alpha, delta):
    alpha_p = []
    for l in range(len(delta)):
        alpha_p_l = {}
        for key in delta[l]:
            alpha_p_l[key] = torch.mul(alpha[l][key], delta[l][key])
        alpha_p.append(alpha_p_l)
    return alpha_p

def get_multiply(alpha, delta):
    alpha_p = []
    for l in range(len(delta)):
        alpha_p_l = {}
        for key in delta[l]:
            alpha_p_l[key] = np.multiply(alpha[l][key], delta[l][key])
        alpha_p.append(alpha_p_l)
    return alpha_p

def get_weighted_sum_batch(hat_v, batch_grads_test, params):
    alpha_p = get_zero(params)
    for l in range(params['numlayers']):

        # print('hat_v.shape')
        # print(hat_v.shape)
        # print('batch_grads_test[W][l].shape')
        # print(batch_grads_test['W'][l].shape)
        # print('(hat_v * batch_grads_test[W][l]).shape')
        # print((hat_v * batch_grads_test['W'][l]).shape)

        # print('np.sum(hat_v * batch_grads_test[W][l], axis=0).shape')
        # print(np.sum(hat_v * batch_grads_test['W'][l], axis=0).shape)

        alpha_p['W'][l] = np.sum(hat_v[:, None, None] * batch_grads_test['W'][l], axis=0)
        alpha_p['b'][l] = np.sum(hat_v[:, None] * batch_grads_test['b'][l], axis=0)
    return alpha_p

def get_opposite(delta):
    numlayers = len(delta)
    
    p = []
    for l in range(numlayers):
        # if params['layers_params'][l]['name'] == 'fully-connected':
        p_l = {}
        for key in delta[l]:
            p_l[key] = -delta[l][key]
        # else:
            # print('Error: layer unsupported')
            # sys.exit()
        p.append(p_l)
        
    return p

def SMW_GN_update(data_, params):
    # a[l].grad: size N1 * m[l+1], it has a coefficient 1 / N1, which should be first compensate
    # h[l]: size N1 * m[l]
    # model.W[l]: size m[l+1] * m[l]
    
    model_grad = data_['model_regularized_grad_used_torch']
    model = data_['model']
    
    
    
    N1 = params['N1']
    N2 = params['N2']
    lambda_ = params['lambda_']
    
    m_L = data_['model'].layersizes[-1]

    params['m_L'] = m_L
    m_L = params['m_L']
    
    data_ = get_cache_momentum(data_, params)
  
    
    z_N2 = data_['z_N2']
    z_data = z_N2
    y = F.softmax(z_data, dim = 1)
    data_['y'] = y
   
    v = compute_JV(model_grad, data_, params)
   
    D_t = get_D_t(data_, params)

    v = torch.unsqueeze(v, -1)
    hat_v, _ = torch.solve(v.data, D_t.data)
    hat_v = torch.squeeze(hat_v, dim=1)

    # theoretically, cholesky should be faster than torch.solve()
    # however, this only happens in practice when the size of matrix is large
    # if matrix is small, torch.solve() is faster
    # since we always want to deal with small matricex, we choose to use torch.solve()

    
    if model.name_loss == 'binary classification':
        1
    elif model.name_loss == 'multi-class classification':
        hat_v = get_HV(hat_v, data_, params)
    else:
        print('Error: unknown loss.')
        sys.exit()
  
    
    
    delta = compute_sum_J_transpose_V_backp(hat_v, data_, params)
    delta = get_multiply_scalar(1 / N2, delta)
    delta = get_subtract_torch(model_grad, delta)
    
    delta = get_multiply_scalar(1 / lambda_, delta)
    
        
    p = get_opposite(delta)
   
    data_['p_torch'] = p
        
    return data_

def compute_sum_J_transpose_V(v, data_, params):
    a_grad_momentum = data_['a_grad_momentum'] # N2 * m[l+1]
    h_momentum = data_['h_momentum'] # N2 * m[l]
    
    numlayers = params['numlayers']
    
    delta = []

    for l in range(numlayers):
        delta_l = {}
        delta_l['b'] = torch.mv(a_grad_momentum[l].t(), v)
        delta_l['W'] = torch.mm(
            (v[:, None] * a_grad_momentum[l]).t(), h_momentum[l])
        delta.append(delta_l)
    
    return delta

def update_lambda(p, data_, params):
    true_algorithm = params['algorithm']
    if params['algorithm'] in ['SMW-Fisher-signVAsqrt-p',
                               'SMW-Fisher-VA-p',
                               'SMW-Fisher-momentum-p-sign',
                               'SMW-Fisher-momentum-p',
                               'SMW-Fisher-momentum',
                               'SMW-Fisher-sign']:
        params['algorithm'] = 'SMW-Fisher'
    elif params['algorithm'] in ['kfac-momentum-grad',
                                 'kfac-EF',
                                 'kfac-TR',
                                 'kfac-momentum-grad-TR',
                                 'kfac-CG',
                                 'kfac-momentum-grad-CG',]:
        params['algorithm'] = 'kfac'

    model = data_['model']
    X_mb_N1 = data_['X_mb_N1']
    t_mb_N1 = data_['t_mb_N1']
    loss_N1 = data_['regularized_loss']
    model_grad = data_['model_grad_used_torch']
    
    numlayers = params['numlayers']
    lambda_ = params['lambda_']
    boost = params['boost']
    drop = params['drop']
    
    algorithm = params['algorithm']
    ll_chunk = get_new_loss(model, p, X_mb_N1, t_mb_N1, params)
    oldll_chunk = loss_N1


    if oldll_chunk - ll_chunk < 0:
        rho = float("-inf")
    else:
        if algorithm in ['SMW-Fisher-different-minibatch',
                         'SMW-Fisher',
                         'SMW-GN',
                         'GI-Fisher',
                         'matrix-normal-same-trace',
                         'matrix-normal',
                         'Kron-BFGS-LM',
                         'Kron-BFGS-LM-sqrt']:
            denom = - 0.5 * get_dot_product_torch(model_grad, p)
        elif algorithm in ['SMW-Fisher-batch-grad-momentum-exponential-decay',
                           'ekfac-EF',
                           'kfac',
                           'kfac-test',
                           'kfac-no-max',
                           'kfac-NoMaxNoSqrt',
                           'SMW-Fisher-momentum',
                           'SMW-Fisher-D_t-momentum',
                           'SMW-Fisher-momentum-D_t-momentum',
                           'SMW-Fisher-BD',
                           'RMSprop-individual-grad-no-sqrt-LM',
                           'SMW-Fisher-batch-grad',
                           'SMW-Fisher-batch-grad-momentum']:
            denom = computeFV(p, data_, params)
                
            denom = get_dot_product_torch(p, denom)
            
            
            denom = -0.5 * denom
            denom = denom - get_dot_product_torch(model_grad, p)
                
        else:
            print('algorithm')
            print(algorithm)
            print('Error! 1504')
            sys.exit()
        
        rho = (oldll_chunk - ll_chunk) / denom
    
    
    # update lambda   
    if rho < 0.25:
        lambda_ = lambda_ * boost
    elif rho > 0.75:
        lambda_ = lambda_ * drop

    if true_algorithm in ['SMW-Fisher-signVAsqrt-p',
                          'SMW-Fisher-VA-p',
                          'SMW-Fisher-momentum-p-sign',
                          'SMW-Fisher-momentum-p',
                          'SMW-Fisher-momentum',
                          'SMW-Fisher-sign',
                          'kfac-momentum-grad',
                          'kfac-TR',
                          'kfac-momentum-grad-TR',
                          'kfac-EF',
                          'kfac-CG',
                          'kfac-momentum-grad-CG',]:
        params['algorithm'] = true_algorithm
        
    return lambda_

def GI_Fisher_update(data_, params):
    model_grad = data_['model_grad_used']

    N1 = params['N1']
    N2 = params['N2']

    data_ = get_cache_momentum(data_, params)

    G = get_D_t(data_, params)
    G_cho = torch.cholesky(G.data)
    

    # compute J * g
    v = compute_JV(model_grad, data_, params)
    hat_v = torch.cholesky_solve(v.data, G_cho)
    hat_v = torch.cholesky_solve(hat_v.data, G_cho)

    # compute J^T * (G^{-1} * G^{-1} * J * g)
    delta = compute_sum_J_transpose_V_backp(hat_v, data_, params)

    # dividing by n
    delta = get_multiply_scalar(1 / N2, delta)

    # get minus
    p = get_opppsite(delta, params)
    data_['p'] = p
    return data_, params
   
def SMW_Fisher_batch_grad_update(data_, params):
    model_grad = data_['model_grad_used']
    
    N_iters = params['N_iters']
    N2 = params['N2']
    lambda_ = params['lambda_']
    numlayers = params['numlayers']

    if params['algorithm'] == 'SMW-Fisher-batch-grad-momentum-exponential-decay' or\
    params['algorithm'] == 'SMW-Fisher-batch-grad-momentum':
        data_ = get_cache_momentum(data_, params)
    elif params['algorithm'] == 'SMW-Fisher-batch-grad':
        1
    else:
        print('Error: need more on cache')
        sys.exit()

    if params['algorithm'] == 'SMW-Fisher-batch-grad-momentum-exponential-decay':
        rho_kfac = params['rho_kfac']
        N_current = len(data_['batch_grads'])
        c_weights = np.asarray(list(range(N_current)))
        c_weights = N_current - 1 - c_weights
        c_weights = np.power(rho_kfac, c_weights)
        c_weights = c_weights * (1 - rho_kfac) / (1 - (rho_kfac**N_current))


    # compute J * g
    if params['algorithm'] == 'SMW-Fisher-batch-grad-momentum-exponential-decay' or\
    params['algorithm'] == 'SMW-Fisher-batch-grad-momentum':
        batch_grads = data_['batch_grads']

        # test_start_time = time.process_time()

        v = np.zeros(len(batch_grads))
        for i in range(len(batch_grads)):
            v[i] = get_dot_product(model_grad, batch_grads[i])

        if params['algorithm'] == 'SMW-Fisher-batch-grad-momentum-exponential-decay':
            v = np.sqrt(c_weights) * v
    elif params['algorithm'] == 'SMW-Fisher-batch-grad':
        model_grad_N2 = data_['model_regularized_grad_N2']
        v = get_dot_product(model_grad, model_grad_N2)
    else:
        print('Error: need more J')
        sys.exit()
        

    # compute D_t
    if params['algorithm'] == 'SMW-Fisher-batch-grad-momentum-exponential-decay' or\
    params['algorithm'] == 'SMW-Fisher-batch-grad-momentum':
        N_iters = params['N_iters']

        # delete old
        if len(data_['D_t_minus_lambda']) == N_iters:
            data_['D_t_minus_lambda'] = data_['D_t_minus_lambda'][1:, 1:]

        # add new
        if len(batch_grads) == 1:
            data_['D_t_minus_lambda'] =\
            np.ones((1,1)) * get_dot_product(batch_grads[0], batch_grads[0])
        else:

            D_t_i = np.zeros((len(batch_grads), 1))
            batch_grads_a_grad_j = data_['batch_grads_a_grad'][-1]
            batch_grads_h_j = data_['batch_grads_h'][-1]
            for i in range(len(batch_grads)):
                batch_grads_a_grad_i = data_['batch_grads_a_grad'][i]
                batch_grads_h_i = data_['batch_grads_h'][i]
                for l in range(numlayers):

                    D_t_i[i, 0] += 1 / (N2**2) * (
                        torch.mul(torch.mm(batch_grads_a_grad_j[l], batch_grads_a_grad_i[l].t()),
                                    torch.mm(batch_grads_h_j[l], batch_grads_h_i[l].t()) + 1)).sum()
              
            data_['D_t_minus_lambda'] = np.concatenate((data_['D_t_minus_lambda'], D_t_i[:-1]), axis=1)
            data_['D_t_minus_lambda'] = np.concatenate(
                (data_['D_t_minus_lambda'], np.transpose(D_t_i)), axis=0)
            
        if params['algorithm'] == 'SMW-Fisher-batch-grad-momentum-exponential-decay':
            D_t = data_['D_t_minus_lambda'] * np.outer(np.sqrt(c_weights), np.sqrt(c_weights))
        elif params['algorithm'] == 'SMW-Fisher-batch-grad-momentum':
            D_t = 1 / len(data_['D_t_minus_lambda']) * data_['D_t_minus_lambda']
        D_t = D_t + lambda_ * np.eye(len(data_['D_t_minus_lambda']))
            
    elif params['algorithm'] == 'SMW-Fisher-batch-grad':
        D_t = lambda_
        D_t += get_dot_product(model_grad_N2, model_grad_N2)
    else:
        print('Error: need more D_t')
        sys.exit()
        


    # compute D_t^{-1} * (J * g)
    if params['algorithm'] == 'SMW-Fisher-batch-grad-momentum-exponential-decay' or\
    params['algorithm'] == 'SMW-Fisher-batch-grad-momentum':# D_t_cho_fac = scipy.linalg.cho_factor(D_t)
        D_t_cho = torch.cholesky(D_t.data)
        hat_v = torch.cholesky_solve(v.data, D_t_cho)
    elif params['algorithm'] == 'SMW-Fisher-batch-grad':
        hat_v = v / D_t
    else:
        print('Error: need more solve')
        sys.exit()
        

    # compute J^T * (D_t^{-1} * (J * g))
    if params['algorithm'] == 'SMW-Fisher-batch-grad-momentum-exponential-decay' or\
    params['algorithm'] == 'SMW-Fisher-batch-grad-momentum':
        if params['algorithm'] == 'SMW-Fisher-batch-grad-momentum-exponential-decay':
            hat_v = hat_v * np.sqrt(c_weights)

        
        batch_grads_test = data_['batch_grads_test']


        p = get_weighted_sum_batch(hat_v, batch_grads_test, params)

    elif params['algorithm'] == 'SMW-Fisher-batch-grad':
        p = get_multiply_scalar(hat_v, model_grad_N2)
    else:
        print('Error: need more transpose')
        sys.exit()
        
    
    # rest of SMW
    if params['algorithm'] == 'SMW-Fisher-batch-grad-momentum':
        p = get_multiply_scalar(1 / N_iters, p)

    p = get_subtract(model_grad, p, params)
    
    p = get_multiply_scalar(1 / lambda_, p)

    
    p = get_opposite(p)
    data_['p'] = p
    return data_, params

def get_new_loss(model, p, x, t, params):
    model_new = copy.deepcopy(model)

    device = params['device']
    for l in range(model_new.numlayers):
        for key in model_new.layers_weight[l]:
            model_new.layers_weight[l][key].data += p[l][key].data

    reduction = 'mean'
    loss = get_regularized_loss_from_x_no_grad(
        model_new, x, t, reduction, params['tau'])
    return loss

def get_dot_product(delta_1, delta_2):
    dot_product = 0
    for l in range(len(delta_1)):
        for key in delta_1[l]:
            dot_product += np.sum(np.multiply(delta_1[l][key], delta_2[l][key]))
    return dot_product

def get_dot_product_blockwise(delta_1, delta_2):
    dot_product = []
    for l in range(len(delta_1)):
        dot_product_l = 0
        for key in delta_1[l]:
            dot_product_l += np.sum(np.multiply(delta_1[l][key], delta_2[l][key]))
        dot_product.append(dot_product_l)
    return dot_product

def get_dot_product_torch(delta_1, delta_2):
    dot_product = sum(
        [
            sum(
                [
                    torch.sum(torch.mul(delta_1_l[key], delta_2_l[key])).item() for key in delta_1_l
                ]
            ) for delta_1_l, delta_2_l in zip(delta_1, delta_2)
        ]
    )
            
    return dot_product

def get_dot_product_blockwise_torch(delta_1, delta_2):
    dot_product = []
    for l in range(len(delta_1)):
        dot_product_l = 0
        for key in delta_1[l]:
            dot_product_l += torch.sum(torch.mul(delta_1[l][key], delta_2[l][key]))
        dot_product.append(dot_product_l)
    return dot_product

def get_dot_product_batch(model_grad, batch_grads_test, params):
    # numlayers = params['numlayers']
    
    dot_product = np.zeros(len(batch_grads_test['W'][0]))
    for l in range(params['numlayers']):
        dot_product += np.sum(
            np.sum(np.multiply(model_grad['W'][l][None, :], batch_grads_test['W'][l]), axis=-1), axis=-1)
        dot_product += np.sum(np.multiply(model_grad['b'][l][None, :], batch_grads_test['b'][l]), axis=-1)
    
    return dot_product

def get_square(delta_1):
    numlayers = len(delta_1)
    sqaure_p = []
    for l in range(numlayers):
        sqaure_p_l = {}
        for key in delta_1[l]:
            sqaure_p_l[key] = np.square(delta_1[l][key])
        sqaure_p.append(sqaure_p_l)  
    return sqaure_p

def get_square_torch(delta_1):
    numlayers = len(delta_1)
    sqaure_p = []
    for l in range(numlayers):
        sqaure_p_l = {}
        for key in delta_1[l]:
            sqaure_p_l[key] = torch.mul(delta_1[l][key], delta_1[l][key])
        sqaure_p.append(sqaure_p_l)  
    return sqaure_p

def get_sqrt(delta_1):
    sqaure_p = []
    for l in range(len(delta_1)):
        sqaure_p_l = {}
        for key in delta_1[l]:
            sqaure_p_l[key] = np.sqrt(delta_1[l][key])
        sqaure_p.append(sqaure_p_l) 
    return sqaure_p

def get_sqrt_torch(delta_1):
    sqaure_p = []
    for l in range(len(delta_1)):
        sqaure_p_l = {}
        for key in delta_1[l]:
            sqaure_p_l[key] = torch.sqrt(delta_1[l][key])
        sqaure_p.append(sqaure_p_l) 
    return sqaure_p

def get_max_with_0(delta_1):
    sqaure_p = []
    for l in range(len(delta_1)):
        sqaure_p_l = {}
        for key in delta_1[l]:
            sqaure_p_l[key] = F.relu(delta_1[l][key])
        sqaure_p.append(sqaure_p_l) 
    return sqaure_p

def get_divide(delta_1, delta_2):
    numlayers = len(delta_1)
    sqaure_p = []
    for l in range(numlayers):
        sqaure_p_l = {}
        for key in delta_1[l]:
            sqaure_p_l[key] = np.divide(delta_1[l][key], delta_2[l][key])
        sqaure_p.append(sqaure_p_l)
    return sqaure_p

def get_divide_torch(delta_1, delta_2):
    numlayers = len(delta_1)
    sqaure_p = []
    for l in range(numlayers):
        sqaure_p_l = {}
        for key in delta_1[l]:
            sqaure_p_l[key] = torch.div(delta_1[l][key], delta_2[l][key])
        sqaure_p.append(sqaure_p_l)
    return sqaure_p

def computeFV(delta, data_, params):
    model = data_['model']
    
    N1 = params['N1']
    N2 = params['N2']

    algorithm = params['algorithm']
    v = compute_JV(delta, data_, params)
    if algorithm == 'SMW-GN':
        # v = v.data.numpy()
        v = get_HV(v, data_, params)
        # v = torch.from_numpy(v)
    delta = compute_sum_J_transpose_V_backp(v, data_, params)
    delta = get_multiply_scalar(1 / N2, delta)
    return delta
   
def get_homo_grad(model_grad_N1, params):
    device = params['device']

    homo_model_grad_N1 = []
    for l in range(params['numlayers']):
        if params['layers_params'][l]['name'] == 'fully-connected':
            
            homo_model_grad_N1_l = torch.cat(
                (model_grad_N1[l]['W'], model_grad_N1[l]['b'].unsqueeze(1)), dim=1)
        elif params['layers_params'][l]['name'] in ['conv',
                                                    'conv-no-activation']:
            # take Fashion-MNIST as an example
            # model_grad_N1[l]['W']: 32 * 1 * 5 * 5
            # model_grad_N1[l]['b']: 32
            # 32: conv_out_channels
            # 1: conv_in_channels
            # 5 * 5: conv_kernel_size
            
            homo_model_grad_N1_l = torch.cat(
                (
                    model_grad_N1[l]['W'].flatten(start_dim=1),
                    model_grad_N1[l]['b'].unsqueeze(dim=1)
                ),
                dim=1
            )
            
        elif params['layers_params'][l]['name'] in ['conv-no-bias-no-activation']:
            
            homo_model_grad_N1_l = model_grad_N1[l]['W'].flatten(start_dim=1)
            
        elif params['layers_params'][l]['name'] == 'BN':
            
            homo_model_grad_N1_l = torch.cat(
                (model_grad_N1[l]['W'], model_grad_N1[l]['b'])
            )
            
        else:
            print('Error: unsupported layer when homo grad for ' + params['layers_params'][l]['name'])
            sys.exit()
        homo_model_grad_N1.append(homo_model_grad_N1_l)

    return homo_model_grad_N1  
    
def Kron_SGD_update(data_, params):
    numlayers = params['numlayers']
    
    model_grad = data_['model_regularized_grad_used_torch']
    
    delta = []
    for l in range(numlayers):
        delta_l = {}
        
        mean_a_grad_l = torch.mean(data_['a_grad_N2'][l], dim=0)
        mean_h_l = torch.mean(data_['h_N2'][l], dim=0)
        
        delta_l['W'] = torch.ger(mean_a_grad_l, mean_h_l)
        delta_l['b'] = model_grad[l]['b']
        
        delta.append(delta_l)
        
    p = get_opposite(delta)
    data_['p_torch'] = p
        
    
    return data_, params

def HessianAction_scaled_BFGS_update(Kron_BFGS_matrices_l, l, data_, params):
    
    assert params['Kron_BFGS_action_h'] == 'HessianAction-scaled-BFGS'
    
    mean_h_l = torch.mean(data_['h_N2'][l], dim=0).data

    if params['Kron_BFGS_if_homo']:
        mean_h_l = torch.cat((mean_h_l, torch.ones(1, device=params['device'])), dim=0)

    H_l_h = Kron_BFGS_matrices_l['H']['h']
    s_l_h = torch.mv(H_l_h, mean_h_l)

    beta_ = params['Kron_BFGS_A_decay']
    s_l_h = s_l_h / beta_
    y_l_h = torch.mv(Kron_BFGS_matrices_l['A_LM'], s_l_h)






    beta_ = params['Kron_BFGS_A_decay']
    H_l_h = H_l_h / beta_

    Kron_BFGS_matrices_l['H']['h'], update_status = get_BFGS_formula_v2(H_l_h, s_l_h, y_l_h, mean_h_l, False)
    
    print('torch.norm(Kron_BFGS_matrices_l[H][h])')
    print(torch.norm(Kron_BFGS_matrices_l['H']['h']))

    if update_status != 0:


        sys.exit()
    return Kron_BFGS_matrices_l

def get_BFGS_PowellB0Damping(s_l_a, y_l_a, params):
    
    print('need to move')
    sys.exit()
    
    # B_0 = 1 / gamma * I
    
    delta = params['Kron_BFGS_H_epsilon']
    
    s_T_y = torch.dot(s_l_a, y_l_a)
    y_T_y = torch.dot(y_l_a, y_l_a)
    
    gamma = (y_T_y / s_T_y).item()
    
    if gamma < delta:
        gamma = delta
    
    
    
    alpha = params['Kron_BFGS_H_epsilon']

    s_T_s = torch.dot(s_l_a, s_l_a)

    s_B_s = s_T_s/gamma

    if s_T_y / s_B_s > alpha:
        1
    else:
        theta =  (1-alpha) * s_B_s / (s_B_s - s_T_y)
        y_l_a = theta * y_l_a + (1-theta) * s_l_a/gamma
        
    
    return s_l_a, y_l_a

def get_BFGS_DoubleDamping(s_l_a, y_l_a, l, data_, params):
    print('need to move')
    sys.exit()
    
    alpha = params['Kron_BFGS_H_epsilon']
        
    Kron_BFGS_matrices_l = data_['Kron_BFGS_matrices'][l]

    if params['Kron_BFGS_action_a'] == 'LBFGS':
        Hy = LBFGS_Hv(
            y_l_a,
            data_['Kron_LBFGS_s_y_pairs']['a_grad'][l],
            params,
            False
        )
    elif params['Kron_BFGS_action_a'] == 'BFGS':
        H_l_a_grad = Kron_BFGS_matrices_l['H']['a_grad']
        Hy = torch.mv(H_l_a_grad ,y_l_a)
    else:
        print('error: not implemented for ' + params['Kron_BFGS_action_a'])
        sys.exit()
    s_T_y = torch.dot(s_l_a, y_l_a)
    yHy = torch.dot(y_l_a, Hy)
    s_T_s = torch.dot(s_l_a, s_l_a)
    sigma = max(yHy.item(), s_T_s.item())
    if s_T_y / sigma > alpha:
        1
    else:
        if yHy >= s_T_s:
            theta =  ((1-alpha) * yHy / (yHy - s_T_y)).item()
            s_l_a = theta * s_l_a + (1-theta) * Hy
        else:
            theta =  (1-alpha) * s_T_s / (s_T_s - s_T_y)
            y_l_a = theta * y_l_a + (1-theta) * s_l_a
    
    return s_l_a, y_l_a
    
def get_block_BFGS_formula(H, s, y): 
    D = s
    
    D_t_y_inv = torch.mm(D.t(), y).inverse()
    I = torch.eye(H.size()[0]).cuda()
    
    H = torch.mm(torch.mm(D, D_t_y_inv), D.t()) +\
    torch.mm(
        torch.mm(
            I - torch.mm(torch.mm(D, D_t_y_inv), y.t()), H),
        I - torch.mm(torch.mm(y, D_t_y_inv), D.t()))
    
    
    return H

def get_BFGS_formula(H, s, y, g_k):
    s = s.data
    y = y.data

    rho_inv = torch.dot(s, y)

    if rho_inv <= 0:
    
        return H, 1
    elif rho_inv <= 10**(-4) * torch.dot(s, s) * np.sqrt(torch.dot(g_k, g_k).item()):
        return H, 2

    rho = 1 / rho_inv

    Hy = torch.mv(H, y)
    H_new = H.data + (rho**2 * torch.dot(y, torch.mv(H, y)) + rho) * torch.ger(s, s) -\
    rho * (torch.ger(s, Hy) + torch.ger(Hy, s))
    

    
    if torch.norm(H_new) > 2 * torch.norm(H):
        return H, 3
    elif torch.max(torch.isinf(H_new)):
        return H, 4
    else:
        H = H_new

    if torch.max(torch.isinf(H)):
        print('inf in H')
        print('s')
        print(s)
        print('y')
        print(y)
        sys.exit()

    return H, 0

def get_CG(func_A, b, x, max_iter, data_):
    r = func_A(x) - b
    p = - r
    r_k_norm = torch.sum(r * r)
    i = 0
    while i < max_iter:
        Ap = func_A(p)
        alpha = r_k_norm / torch.sum(p * Ap)

        x += alpha * p

        r += alpha * Ap
        r_kplus1_norm = torch.sum(r * r)
        beta = r_kplus1_norm / r_k_norm
        r_k_norm = r_kplus1_norm

        if r_kplus1_norm < 1e-10:
        # if torch.sum(x * func_A(x)) - torch.sum(x * b) < 1e-10:
            break
        p = beta * p - r
        i += 1

    return x

def get_safe_division(x, y):
    if x == 0 and y == 0:
        print('Error: x = 0 and y = 0 in safe division')
        sys.exit()
    elif x == 0 and y !=0:
        return 0
    elif x != 0 and y == 0:
        return 1e16
    else:
        if np.log(x) - np.log(y) < np.log(1e16):
            return np.exp(np.log(x) - np.log(y))
        else:
            return 1e16

def get_A_A_T_kfac(a, h, l, params):
    
    layers_params = params['layers_params']
    device = params['device']
    
    kernel_size = layers_params[l]['conv_kernel_size']
    padding = layers_params[l]['conv_padding']


    if layers_params[l]['name'] == '1d-conv':
        size_test_2_A_j = kernel_size *layers_params[l]['conv_in_channels']
    elif layers_params[l]['name'] == 'conv':
        
        size_test_2_A_j = kernel_size**2 *layers_params[l]['conv_in_channels']
    
    if params['Kron_BFGS_if_homo']:
        size_test_2_A_j += 1


    test_2_A_j = torch.zeros(
        size_test_2_A_j, size_test_2_A_j, device=device
    )

    if layers_params[l]['name'] == '1d-conv':
        # 1d-conv: a[l]: M * I * |T|
        h_l_padded = F.pad(h[l], (padding, padding), "constant", 0)
        
        h_homo_ones = torch.ones(h_l_padded.size(0), 1 ,device=device)
        
        

        for t in range(a[l].size(2)):
            # a[l].size(2) = |T|
            
            h_l_t = h_l_padded[:, :, t:t+kernel_size].data

            # in the flatten, delta changes the fastest
            h_l_t_flatten = h_l_t.flatten(start_dim=1)
            
            if params['Kron_BFGS_if_homo']:
                h_l_t_flatten = torch.cat((h_l_t_flatten, h_homo_ones), dim=1)

            test_2_A_j += torch.mm(h_l_t_flatten.t(), h_l_t_flatten)
    elif layers_params[l]['name'] == 'conv':
        # 2d-conv: a[l]: M * I * |T|, where |T| has two dimensions
        h_l_padded = F.pad(
            h[l], (padding, padding, padding, padding), "constant", 0
        )
        
        h_homo_ones = torch.ones(h_l_padded.size(0), 1 ,device=device)

        for t1 in range(a[l].size(2)):
            for t2 in range(a[l].size(3)):
                h_l_t = h_l_padded[:, :, t1:t1+kernel_size, t2:t2+kernel_size].data
                h_l_t_flatten = h_l_t.flatten(start_dim=1)
                
                if params['Kron_BFGS_if_homo']:
                    h_l_t_flatten = torch.cat((h_l_t_flatten, h_homo_ones), dim=1)
    
                test_2_A_j += torch.mm(h_l_t_flatten.t(), h_l_t_flatten)
                
                
    return test_2_A_j

def get_A_A_T_kron_bfgs_v5(h, l, params):  
    layers_params = params['layers_params']
    device = params['device']
                        
    in_channels = layers_params[l]['conv_in_channels']
    kernel_size = layers_params[l]['conv_kernel_size']
    padding = layers_params[l]['conv_padding']

    if layers_params[l]['name'] == '1d-conv':

        test_5_A_j = torch.zeros(
            in_channels,
            kernel_size,
            in_channels,
            kernel_size,
            device=device
        )

        h_l_padded = F.pad(h[l].data, (padding, padding), "constant", 0)
        
        T = h_l_padded.size(2)-kernel_size+1
        
        weight_conv1d = torch.ones(1, 1, T, device=device)

        for delta_2 in range(-kernel_size+1, kernel_size):
            
            
            if delta_2 > 0:
                h_l_diff_1 = h_l_padded[:, :, delta_2:h_l_padded.size(2)]
                h_l_diff_2 = h_l_padded[:, :, 0:h_l_padded.size(2)-delta_2]
            else:
                h_l_diff_1 = h_l_padded[:, :, 0:h_l_padded.size(2)+delta_2]
                h_l_diff_2 = h_l_padded[:, :, -delta_2:h_l_padded.size(2)]
            
            h_l_square = torch.einsum('ijl,ikl->jkl', h_l_diff_1.data, h_l_diff_2.data)
            
            h_l_square_conv = F.conv1d(
                h_l_square.view(h_l_square.size(0) * h_l_square.size(1), 1, -1),
                weight_conv1d
            ).view(h_l_square.size(0), h_l_square.size(1), -1)
            
            if delta_2 > 0:
                start_index = [0, delta_2]
            else:
                start_index = [-delta_2, 0]

            for delta in range(kernel_size-np.abs(delta_2)):
                test_5_A_j[
                    :, start_index[0] + delta, :, start_index[1] + delta
                ] = h_l_square_conv[:, :, delta].t()

            
                
        test_5_A_j = test_5_A_j.view(
        test_5_A_j.size(0) * test_5_A_j.size(1),
        test_5_A_j.size(2) * test_5_A_j.size(3),
    )
 
        
        if params['Kron_BFGS_if_homo']:
            homo_test_5_A_j = torch.zeros(
                test_5_A_j.size(0)+1, test_5_A_j.size(1)+1, device=device
            )
            
            homo_test_5_A_j[:-1, :-1] = test_5_A_j

            sum_h_l_padded = torch.sum(h_l_padded, dim=0)
            sum_h_l_padded = sum_h_l_padded.unsqueeze(1)
            weight_conv1d = torch.ones(1, 1, T, device=device) 
            homo_test_5_A_j[-1, :-1] =\
            F.conv1d(sum_h_l_padded, weight_conv1d).view(-1)
            
            homo_test_5_A_j[:-1, -1] = homo_test_5_A_j[-1, :-1]
            
            homo_test_5_A_j[-1, -1] = T * h_l_padded.size(0)
            
            test_5_A_j = homo_test_5_A_j

    elif layers_params[l]['name'] == 'conv':
        test_5_A_j = torch.zeros(
            in_channels,
            kernel_size,
            kernel_size,
            in_channels,
            kernel_size,
            kernel_size,
            device=device
        )

        h_l_padded = F.pad(
            h[l].data, (padding, padding, padding, padding), "constant", 0
        )
        
        T0 = h_l_padded.size(2)-kernel_size+1
        T1 = h_l_padded.size(3)-kernel_size+1
        
        
        weight_conv1d = torch.ones(1, 1, h_l_padded.size(2)-kernel_size+1, h_l_padded.size(3)-kernel_size+1, device=device)
        
        
        for delta_2_0 in range(-kernel_size+1, kernel_size):
            for delta_2_1 in range(-kernel_size+1, kernel_size):
                
                
                if delta_2_0 > 0:
                    h_l_diff_1 = h_l_padded[:, :, delta_2_0:h_l_padded.size(2)]
                    h_l_diff_2 = h_l_padded[:, :, 0:h_l_padded.size(2)-delta_2_0]
                else:
                    h_l_diff_1 = h_l_padded[:, :, 0:h_l_padded.size(2)+delta_2_0]
                    h_l_diff_2 = h_l_padded[:, :, -delta_2_0:h_l_padded.size(2)]
                    
                    
                if delta_2_1 > 0:
                    h_l_diff_1 = h_l_diff_1[:, :, :, delta_2_1:h_l_padded.size(3)]
                    h_l_diff_2 = h_l_diff_2[:, :, :, 0:h_l_padded.size(3)-delta_2_1]
                else:
                    h_l_diff_1 = h_l_diff_1[:, :, :, 0:h_l_padded.size(3)+delta_2_1]
                    h_l_diff_2 = h_l_diff_2[:, :, :, -delta_2_1:h_l_padded.size(3)]

                
                
                h_l_square = torch.einsum(
                    'ijlt,iklt->jklt', h_l_diff_1.data, h_l_diff_2.data
                )
                
                
                t1=0
                t2=0
                kfac_h_l_t =h_l_padded[:, :, t1:t1+kernel_size, t2:t2+kernel_size].data


                
                h_l_square_conv = F.conv1d(
                    h_l_square.view(h_l_square.size(0) * h_l_square.size(1), 1, *(h_l_square.size()[2:])),
                    weight_conv1d
                )
                
                
                h_l_square_conv = h_l_square_conv.view(h_l_square.size(0), h_l_square.size(1), *(h_l_square_conv.size()[2:]))
 
            
                if delta_2_0 > 0:
                    start_index_0 = [0, delta_2_0]
                else:
                    start_index_0 = [-delta_2_0, 0]

                if delta_2_1 > 0:
                    start_index_1 = [0, delta_2_1]
                else:
                    start_index_1 = [-delta_2_1, 0]

                for delta_0 in range(kernel_size-np.abs(delta_2_0)):
                    for delta_1 in range(kernel_size-np.abs(delta_2_1)):
                        test_5_A_j[
                            :, start_index_0[0] + delta_0, start_index_1[0] + delta_1,
                            :, start_index_0[1] + delta_0, start_index_1[1] + delta_1
                        ] = h_l_square_conv[:, :, delta_0, delta_1].t()

        test_5_A_j = test_5_A_j.view(
        test_5_A_j.size(0) * test_5_A_j.size(1) * test_5_A_j.size(2),
        test_5_A_j.size(3) * test_5_A_j.size(4) * test_5_A_j.size(5),
    )  

        if params['Kron_BFGS_if_homo']:
            homo_test_5_A_j = torch.zeros(
                test_5_A_j.size(0)+1, test_5_A_j.size(1)+1, device=device
            )
            
            homo_test_5_A_j[:-1, :-1] = test_5_A_j

            sum_h_l_padded = torch.sum(h_l_padded, dim=0)
 
            
            sum_h_l_padded = sum_h_l_padded.unsqueeze(1)

            
            weight_conv1d = torch.ones(1, 1, T0, T1, device=device) 
  
            
            homo_test_5_A_j[-1, :-1] =\
            F.conv1d(sum_h_l_padded, weight_conv1d).view(-1)
 
            
            homo_test_5_A_j[:-1, -1] = homo_test_5_A_j[-1, :-1]
            
            homo_test_5_A_j[-1, -1] = T0 * T1 * h_l_padded.size(0)
            
            test_5_A_j = homo_test_5_A_j
     
        
    return test_5_A_j

def get_A_A_T_kron_bfgs(h, l, params):
    
    layers_params = params['layers_params']
    device = params['device']
                        
    in_channels = layers_params[l]['conv_in_channels']
    kernel_size = layers_params[l]['conv_kernel_size']
    padding = layers_params[l]['conv_padding']

    if layers_params[l]['name'] == '1d-conv':

        test_5_A_j = torch.zeros(
            in_channels,
            kernel_size,
            in_channels,
            kernel_size,
            device=device
        )

        h_l_padded = F.pad(h[l], (padding, padding), "constant", 0)

        for delta_2 in range(-kernel_size+1, kernel_size):
            h_l_diff = h_l_padded[
                :,
                :,
                np.remainder(np.asarray(range(h[l].size(2)))-delta_2,h[l].size(2))
                                 ]


            cutting_indices_2 = np.arange(
                    np.maximum(0,delta_2),
                    np.minimum(h_l_padded.size(2),h_l_padded.size(2)+delta_2)
                )


            cutting_indices =\
            np.ix_(
                np.asarray(range(h_l_padded.size(0))),
                np.asarray(range(h_l_padded.size(1))),
                cutting_indices_2
            )

            h_l_diff_1 = h_l_padded[cutting_indices]
            h_l_diff_2 = h_l_diff[cutting_indices]
            einsum_ = torch.einsum('ijl,ikl->ijkl', h_l_diff_1.data, h_l_diff_2.data)


            h_l_square = torch.sum(einsum_, dim=0)

            sum_h_l_square = torch.sum(h_l_square, dim=-1)
            for delta in range(kernel_size-np.abs(delta_2)):

                # the length of minus part is:
                # kernel_size-np.abs(delta_2) - 1

                indices_dim_2 = np.arange(0, h_l_square.size(2))
                indcies_dim_2_included =\
                np.arange(0, h_l_square.size(2) - (kernel_size-np.abs(delta_2) - 1)) + delta

                indcies_dim_2_excludes =\
                np.setdiff1d(indices_dim_2, indcies_dim_2_included)

                if delta_2 > 0:
                    start_index = [0, delta_2]
                else:
                    start_index = [-delta_2, 0]

                test_5_A_j[
                    :, start_index[0] + delta, :, start_index[1] + delta
                ] = (
                    sum_h_l_square-\
                    torch.sum(
                        h_l_square[
                            :, :, indcies_dim_2_excludes
                        ],
                        dim=-1
                    )
                ).t()

        test_5_A_j = test_5_A_j.view(
        test_5_A_j.size(0) * test_5_A_j.size(1),
        test_5_A_j.size(2) * test_5_A_j.size(3),
    )

    elif layers_params[l]['name'] == 'conv':
        test_5_A_j = torch.zeros(
            in_channels,
            kernel_size,
            kernel_size,
            in_channels,
            kernel_size,
            kernel_size,
            device=device
        )

        h_l_padded = F.pad(
            h[l], (padding, padding, padding, padding), "constant", 0
        )
        for delta_2_0 in range(-kernel_size+1, kernel_size):
            for delta_2_1 in range(-kernel_size+1, kernel_size):

                diff_indices =\
                np.ix_(
                    np.asarray(range(h_l_padded.size(0))),
                    np.asarray(range(h_l_padded.size(1))),
                    np.remainder(
                        np.asarray(range(h_l_padded.size(2)))-delta_2_0,
                        h_l_padded.size(2)
                    ),
                    np.remainder(
                        np.asarray(range(h_l_padded.size(3)))-delta_2_1,
                        h_l_padded.size(3)
                    )
                )

                h_l_diff = h_l_padded[diff_indices]

                cutting_indices_2 = np.arange(
                    np.maximum(0,delta_2_0),
                    np.minimum(h_l_padded.size(2),h_l_padded.size(2)+delta_2_0)
                )

                cutting_indices_3 = np.arange(
                    np.maximum(0,delta_2_1),
                    np.minimum(h_l_padded.size(3),h_l_padded.size(3)+delta_2_1)
                )


                cutting_indices =\
                np.ix_(
                    np.asarray(range(h_l_padded.size(0))),
                    np.asarray(range(h_l_padded.size(1))),
                    cutting_indices_2,
                    cutting_indices_3
                )

                h_l_diff_1 = h_l_padded[cutting_indices]

                h_l_diff_2 = h_l_diff[cutting_indices]

                einsum_ = torch.einsum(
                    'ijlt,iklt->ijklt', h_l_diff_1.data, h_l_diff_2.data
                )

                h_l_square = torch.sum(einsum_, dim=0)
                

                
                sum_h_l_square_dim_2 = torch.sum(h_l_square, dim=[2]) # dim 3 remains open
                sum_h_l_square_dim_3 = torch.sum(h_l_square, dim=[3])

                sum_h_l_square = torch.sum(h_l_square, dim=[-1, -2])


                for delta_0 in range(kernel_size-np.abs(delta_2_0)):
                    for delta_1 in range(kernel_size-np.abs(delta_2_1)):

                        if delta_2_0 > 0:
                            start_index_0 = [0, delta_2_0]
                        else:
                            start_index_0 = [-delta_2_0, 0]

                        if delta_2_1 > 0:
                            start_index_1 = [0, delta_2_1]
                        else:
                            start_index_1 = [-delta_2_1, 0]


                        indices_dim_2 = np.arange(0, h_l_square.size(2))
                        indcies_dim_2_included =\
                        np.arange(0, h_l_square.size(2) - (kernel_size-np.abs(delta_2_0) - 1)) + delta_0

                        indcies_dim_2_excludes =\
                        np.setdiff1d(indices_dim_2, indcies_dim_2_included)

                        indices_dim_3 = np.arange(0, h_l_square.size(3))
                        indcies_dim_3_included =\
                        np.arange(0, h_l_square.size(3) - (kernel_size-np.abs(delta_2_1) - 1)) + delta_1

                        indcies_dim_3_excludes =\
                        np.setdiff1d(indices_dim_3, indcies_dim_3_included)


                        # this should be used for sum_h_l_square_dim_2
                        slicing_indices_minus_1 =\
                np.ix_(
                    np.asarray(range(h_l_square.size(0))),
                    np.asarray(range(h_l_square.size(1))),
                    indcies_dim_3_excludes
                )

                        # this should be used for sum_h_l_square_dim_3
                        slicing_indices_minus_2 =\
                np.ix_(
                    np.asarray(range(h_l_square.size(0))),
                    np.asarray(range(h_l_square.size(1))),
                    indcies_dim_2_excludes
                )

                        slicing_indices_plus =\
                np.ix_(
                    np.asarray(range(h_l_square.size(0))),
                    np.asarray(range(h_l_square.size(1))),
                    indcies_dim_2_excludes,
                    indcies_dim_3_excludes
                )
                        

                        test_5_A_j[
                            :, 
                            start_index_0[0]+delta_0,
                            start_index_1[0]+delta_1,
                            :,
                            start_index_0[1]+delta_0,
                            start_index_1[1]+delta_1
                        ] = sum_h_l_square            
        
                        test_5_A_j[
                            :, 
                            start_index_0[0]+delta_0,
                            start_index_1[0]+delta_1,
                            :,
                            start_index_0[1]+delta_0,
                            start_index_1[1]+delta_1
                        ] -=\
                        torch.sum(sum_h_l_square_dim_2[slicing_indices_minus_1], dim=[-1])
                   
        
                        test_5_A_j[
                            :, 
                            start_index_0[0]+delta_0,
                            start_index_1[0]+delta_1,
                            :,
                            start_index_0[1]+delta_0,
                            start_index_1[1]+delta_1
                        ] -=\
                        torch.sum(sum_h_l_square_dim_3[slicing_indices_minus_2], dim=[-1])
                        
                
            
                        test_5_A_j[
                            :, 
                            start_index_0[0]+delta_0,
                            start_index_1[0]+delta_1,
                            :,
                            start_index_0[1]+delta_0,
                            start_index_1[1]+delta_1
                        ] +=\
                        torch.sum(h_l_square[slicing_indices_plus], dim=[-2, -1])
                
                        
    
                        test_5_A_j[
                            :, 
                            start_index_0[0]+delta_0,
                            start_index_1[0]+delta_1,
                            :,
                            start_index_0[1]+delta_0,
                            start_index_1[1]+delta_1
                        ] =\
                        copy.deepcopy(test_5_A_j[
                            :, 
                            start_index_0[0]+delta_0,
                            start_index_1[0]+delta_1,
                            :,
                            start_index_0[1]+delta_0,
                            start_index_1[1]+delta_1
                        ].t())
        
                        '''
                        if l == 1:
                            
                        
                            print('(\
                        sum_h_l_square\
                       -torch.sum(sum_h_l_square_dim_2[slicing_indices_minus_1], dim=[-1])\
                       -torch.sum(sum_h_l_square_dim_3[slicing_indices_minus_2], dim=[-1])\
                       +torch.sum(h_l_square[slicing_indices_plus], dim=[-2, -1])\
                    ).t() -\
                            test_5_A_j[\
                                :, \
                                start_index_0[0]+delta_0,\
                                start_index_1[0]+delta_1,\
                                :,\
                                start_index_0[1]+delta_0,\
                                start_index_1[1]+delta_1\
                            ]')
                            
                            print((
                        sum_h_l_square\
                       -torch.sum(sum_h_l_square_dim_2[slicing_indices_minus_1], dim=[-1])\
                       -torch.sum(sum_h_l_square_dim_3[slicing_indices_minus_2], dim=[-1])\
                       +torch.sum(h_l_square[slicing_indices_plus], dim=[-2, -1])
                    ).t() -\
                            test_5_A_j[
                                :, 
                                start_index_0[0]+delta_0,
                                start_index_1[0]+delta_1,
                                :,
                                start_index_0[1]+delta_0,
                                start_index_1[1]+delta_1
                            ])
                            
                            print('test_5_A_j[\
                                :, \
                                start_index_0[0]+delta_0,\
                                start_index_1[0]+delta_1,\
                                :,\
                                start_index_0[1]+delta_0,\
                                start_index_1[1]+delta_1\
                            ].size()')
                            print(test_5_A_j[
                                :, 
                                start_index_0[0]+delta_0,
                                start_index_1[0]+delta_1,
                                :,
                                start_index_0[1]+delta_0,
                                start_index_1[1]+delta_1
                            ].size())
                            
                            sys.exit()
                            


                        test_5_A_j[
                            :, 
                            start_index_0[0]+delta_0,
                            start_index_1[0]+delta_1,
                            :,
                            start_index_0[1]+delta_0,
                            start_index_1[1]+delta_1
                        ] = (
                    sum_h_l_square\
                   -torch.sum(sum_h_l_square_dim_2[slicing_indices_minus_1], dim=[-1])\
                   -torch.sum(sum_h_l_square_dim_3[slicing_indices_minus_2], dim=[-1])\
                   +torch.sum(h_l_square[slicing_indices_plus], dim=[-2, -1])
                ).t()
                '''
                
    

        test_5_A_j = test_5_A_j.view(
        test_5_A_j.size(0) * test_5_A_j.size(1) * test_5_A_j.size(2),
        test_5_A_j.size(3) * test_5_A_j.size(4) * test_5_A_j.size(5),
    )  
        
    return test_5_A_j