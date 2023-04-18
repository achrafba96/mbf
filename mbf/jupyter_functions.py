from .data_utils import *
from .utils_functions import *
from .models import *
from .train import *
from .plots_utils import *

def train_model(
    home_path,
    dataset_name,
    model_name,
    algorithm,
    lr,
    damping_value,
    weight_decay,
):

    args = {}
    
    args['list_lr'] = [lr]
    args['weight_decay'] = weight_decay
    
    if dataset_name == 'CIFAR-10':
        if model_name == 'ResNet32':
            args['dataset'] = 'CIFAR-10-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias-no-regularization'
            args['initialization_pkg'] = 'normal'
        else:
            print('model_name')
            print(model_name)
            sys.exit()
        
    elif dataset_name == 'CIFAR-100':
        if model_name == 'VGG16':
            args['dataset'] = 'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN-no-regularization'
            args['initialization_pkg'] = 'normal'
        else:
            print('model_name')
            print(model_name)
            sys.exit()
            
    elif dataset_name == 'SVHN':
        if model_name == 'VGG11':
            args['dataset'] = 'SVHN-vgg11'
            args['initialization_pkg'] = 'normal'
        else:
            print('model_name')
            print(model_name)
            sys.exit()
        
    elif dataset_name == 'MNIST':
        assert model_name == 'MLP'
        args['dataset'] = 'MNIST-autoencoder-relu-N1-1000-sum-loss-no-regularization'
        args['initialization_pkg'] = 'normal'
    elif dataset_name == 'FACES':
        assert model_name == 'MLP'
        args['dataset'] = 'FacesMartens-autoencoder-relu-no-regularization'
        args['initialization_pkg'] = 'normal'
    elif dataset_name == 'CURVES':
        assert model_name == 'MLP'
        args['dataset'] = 'CURVES-autoencoder-relu-sum-loss-no-regularization'
        args['initialization_pkg'] = 'normal'
    else:
        print('dataset_name')
        print(dataset_name)
        sys.exit()

    
    if dataset_name in ['MNIST', 'CURVES']:
        # args['if_max_epoch'] = 1 # 0 means max_time
        args['if_max_epoch'] = 0 # 0 means max_time
        args['max_epoch/time'] = 500
        args['num_epoch_to_decay'] = 60
        args['lr_decay_rate'] = 1
        
    elif dataset_name in ['FACES']:
        # args['if_max_epoch'] = 1 # 0 means max_time
        args['if_max_epoch'] = 0 # 0 means max_time
        args['max_epoch/time'] = 2000
        args['num_epoch_to_decay'] = 60
        args['lr_decay_rate'] = 1
        
    elif dataset_name in ['CIFAR-10', 'CIFAR-100', 'SVHN']:
        if algorithm in ['SGD-m', 'Adam']:
            args['if_max_epoch'] = 1 
            args['max_epoch/time'] = 200
            args['num_epoch_to_decay'] = 60
            args['lr_decay_rate'] = 0.1
            
        elif algorithm in ['MBF', 'Shampoo', 'KFAC']:
            args['if_max_epoch'] = 1 
            args['max_epoch/time'] = 150
            args['num_epoch_to_decay'] = 50
            args['lr_decay_rate'] = 0.1
        else:
            print('algorithm')
            print(algorithm)
            sys.exit()
        
    else:
        print('dataset_name')
        print(dataset_name)
    
        sys.exit()
        
    args['momentum_gradient_rho'] = 0.9
    if algorithm == 'SGD-m':
        args['momentum_gradient_dampening'] = 0
        if dataset_name in ['MNIST', 'FACES', 'CURVES']:
            args['algorithm'] = 'SGD-momentum'
        elif dataset_name in ['CIFAR-10', 'CIFAR-100', 'SVHN']:
            args['algorithm'] = 'SGD-LRdecay-momentum'
        else:
            print('dataset_name')
            print(dataset_name)
            sys.exit()
        
        
    elif algorithm == 'Adam':
        args['RMSprop_epsilon'] = damping_value
        args['RMSprop_beta_2'] = 0.999
        args['momentum_gradient_dampening'] = 0.9 # i.e. beta_1
        if dataset_name in ['CIFAR-10', 'CIFAR-100', 'SVHN']:
            args['algorithm'] = 'Adam-noWarmStart-momentum-grad-LRdecay'
        elif dataset_name in ['MNIST', 'FACES', 'CURVES']:
            args['algorithm'] = 'Adam-noWarmStart-momentum-grad'
        else:
            print('dataset_name')
            print(dataset_name)
            sys.exit()
            
    elif algorithm == 'Shampoo':
        args['shampoo_if_coupled_newton'] = True
        args['shampoo_epsilon'] = damping_value
        args['if_Hessian_action'] = False
        args['shampoo_decay'] = 0.9
        args['shampoo_weight'] = 0.1
        args['momentum_gradient_dampening'] = 0
        
        
        if dataset_name in ['MNIST', 'FACES', 'CURVES']:
            args['algorithm'] = 'shampoo-allVariables-filterFlattening-warmStart-lessInverse-momentum-grad'
            args['shampoo_update_freq'] = 1
            args['shampoo_inverse_freq'] = 20
            
        elif dataset_name in ['CIFAR-100', 'CIFAR-10', 'SVHN']:
            args['algorithm'] = 'shampoo-allVariables-filterFlattening-warmStart-momentum-grad-LRdecay'
            args['shampoo_update_freq'] = 10
            args['shampoo_inverse_freq'] = 100
        else:
            print('dataset_name')
            print(dataset_name)
        
            sys.exit()
            
    elif algorithm == 'KFAC':
        args['kfac_if_update_BN'] = True
        args['kfac_if_BN_grad_direction'] = True
        args['kfac_rho'] = 0.9
        args['kfac_damping_lambda'] = damping_value
        args['momentum_gradient_dampening'] = 0
        
        
        if dataset_name in ['MNIST', 'FACES', 'CURVES']:
            args['algorithm'] = 'kfac-correctFisher-warmStart-no-max-no-LM-momentum-grad'
            args['kfac_if_svd'] = False
            args['kfac_cov_update_freq'] = 1
            args['kfac_inverse_update_freq'] = 20
            
        elif dataset_name in ['CIFAR-100', 'CIFAR-10', 'SVHN']:
            args['algorithm'] = 'kfac-correctFisher-warmStart-no-max-no-LM-momentum-grad-LRdecay'
            args['kfac_if_svd'] = False
            args['kfac_cov_update_freq'] = 10
            args['kfac_inverse_update_freq'] = 100
        else:
            print('dataset_name')
            print(dataset_name)
        
            sys.exit()
        
            
    elif algorithm in ['MBF']:
        args['algorithm'] = 'MBNGD-all-to-one-Avg-LRdecay'
        
        args['momentum_gradient_dampening'] = 0
        args['mbngd_damping_epsilon'] = 1e-8
        args['mbngd_damping_lambda'] = damping_value

        if dataset_name in ['MNIST', 'FACES', 'CURVES']:
            args['kfac_cov_update_freq'] = 2
            args['kfac_inverse_update_freq'] = 25
            
        elif dataset_name in ['CIFAR-10', 'CIFAR-100', 'SVHN']:
            args['kfac_cov_update_freq'] = 10
            args['kfac_inverse_update_freq'] = 100
            
        else:
            print('dataset_name')
            print(dataset_name)
            sys.exit()

    else:
        print('algorithm')
        print(algorithm)
        sys.exit()
    
    
    args['record_epoch'] = 1
    args['seed_number'] = 9999
    args['num_threads'] = 8
    
    args['kfac_rho'] = 0.9
    args['if_grafting'] = False
    args['home_path'] = home_path
    args['if_gpu'] = True
    args['if_test_mode'] = False
    args['if_auto_tune_lr'] = False
    args['if_grafting'] = False

    _ = tune_lr(args)
    return