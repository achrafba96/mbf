from .utils_functions import *
from .train import *
from .utils_plot import *
import os, sys

class SuppressPrints:
        #different from Alexander`s answer
        def __init__(self, suppress=True):
            self.suppress = suppress

        def __enter__(self):
            if self.suppress:
                self._original_stdout = sys.stdout
                sys.stdout = open(os.devnull, 'w')

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.suppress:
                sys.stdout.close()
                sys.stdout = self._original_stdout
        
def get_mean_and_scaled_error1(list_data, key, name_loss, if_max_epoch):
    list_curves_raw = []
    for data_ in list_data:
        list_curves_raw.append(data_[key])
        
    if if_max_epoch:
        len_ = max([len(curve) for curve in list_curves_raw])
    else:
        len_ = min([len(curve) for curve in list_curves_raw])
       
    print('len_')
    print(len_)
    list_curves = []
    for curve in list_curves_raw:
        if if_max_epoch:
            if len(curve) == len_:
                list_curves.append(curve)  
        else:
            list_curves.append(curve[:len_])
    for curve in list_curves:
        assert len(curve) == len(list_curves[0]) 
    list_curves = np.asarray(list_curves)
    print('key')
    print(key)
    
    if key in ['train_unregularized_minibatch_losses',
               'epochs',
               'timesCPU']:
        pass
    elif key == 'test_acces':
        if name_loss in ['logistic-regression-sum-loss',
                         'linear-regression-half-MSE',]:
            pass
        elif name_loss in ['multi-class classification',]:
            list_curves = 1 - list_curves
        else:
            print('name_loss')
            print(name_loss)
            sys.exit() 
    else:
        print(key)
        sys.exit()
    return np.mean(list_curves, axis=0), np.std(list_curves, axis=0) / math.sqrt(len(list_curves))



def get_plot_seed_result1(z_value, dataset, algorithms, 
                         if_max_epoch, list_y, path_to_home, if_title, alpha = 0.7, lw =3):
    plt.rc('font', family='serif')
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams.update({'font.size': 22})
    list_y_legend = []
    for key_y in list_y:        
        if key_y == 'train_unregularized_minibatch_losses':
            list_y_legend.append('Training loss')
        elif key_y == 'test_acces':
            list_y_legend.append('Validation error')
        else:
            print('key_y')
            print(key_y)
            sys.exit()

    list_x = ['epochs', 'timesCPU']
    list_x_legend = ['Epochs', 'Process time (in seconds)']

    # plt.figure(figsize=(30,40))
#     fig, axs = plt.subplots(len(list_y), len(list_x), figsize=(15,15))
    fig, axs = plt.subplots(len(list_y), len(list_x), figsize=(15,7.5*len(list_y)))

    fake_args = {'dataset': dataset}
    from_dataset_to_N1_N2(fake_args)

    N1 = fake_args['N1']
    N2 = fake_args['N2']
    for algorithm in algorithms:

    #     algorithm = 'Kron-BFGS-homo-no-norm-gate-miniBatchADamped-HessianActionV2-\
    #     momentum-s-y-DDV2-regularized-grad-momentum-grad'

        algorithm_name = algorithm['name']
        algorithm_name_legend = algorithm['legend']
        lr = algorithm['lr']
#         path_to_home = '/rigel/home/yr2322/gauss_newton/'

        path_to_dir = path_to_home + 'result/' +\
        dataset + '/' +\
        algorithm_name + '/if_gpu_True/alpha_' + str(lr) +\
        '/N1_' + str(N1) + '/N2_' + str(N2) + '/'

        list_data = []

        for file in os.listdir(path_to_dir):
            with open(path_to_dir + file, 'rb') as fp:
                data_ = pickle.load(fp)
            if data_['params']['if_test_mode'] == False:
                flag = True
                if 'params' in algorithm:
                    for key in algorithm['params']:
                        if key not in data_['params']:
                            flag = False
                            break
                        if algorithm['params'][key] != data_['params'][key]:
                            flag = False
                            break
                if flag:
                    print('data_[params]')
                    print(data_['params'])
                    list_data.append(data_)

        print('len(list_data)')
        print(len(list_data))

        for j in range(len(list_x)):
            key_x = list_x[j]
            name_loss = get_name_loss(dataset)
            mean_x, _ = get_mean_and_scaled_error1(list_data, key_x, name_loss, if_max_epoch)
            for i in range(len(list_y)):
                key_y = list_y[i]
                mean_y, scaled_error = get_mean_and_scaled_error1(list_data, key_y, name_loss, if_max_epoch)
                
                if key_y == 'test_acces':
                    print('1 - np.min(mean_y)')
                    print(1 - np.min(mean_y))
                
                if len(list_y) == 1:
                    ax = axs[j]
                else:
                    ax = axs[i, j]

                # see https://www.mathsisfun.com/data/confidence-interval.html
                ax.plot(mean_x, mean_y, label=algorithm_name_legend, lw = lw, alpha = alpha)
                
#                 ax.fill_between(
#                     mean_x, 
#                     mean_y-error/math.sqrt(len(list_data)), 
#                     mean_y+error/math.sqrt(len(list_data)), 
#                     alpha=0.5
#                 )
                ax.fill_between(
                    mean_x, 
                    mean_y - z_value * scaled_error, 
                    mean_y + z_value * scaled_error, 
                    alpha=0.5
                )
                

                ax.grid(True, color="#93a1a1", alpha=0.5)
                ax.set_yscale('log')
                
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)

                ax.set_xlabel(list_x_legend[j])
                ax.set_ylabel(list_y_legend[i])

                
    plt.legend()    
    if if_title:
        fig.suptitle(dataset)
        
    plt.tight_layout()
    
    path_to_dir = path_to_home + 'logs/plot_seed_result/' + dataset + '/'
    
    if not os.path.exists(path_to_dir):
        os.makedirs(path_to_dir)
    plt.savefig(
        path_to_dir + str(datetime.datetime.now().strftime('%Y-%m-%d-%X')) + '.pdf'
    )
    plt.show()
    
def plot_results_all(algorithms_jupyter, 
                     home_path = '/rigel/home/ab4689/ngd/',
                     dataset_name = 'FACES', 
                     full_legend = False,
                     if_print = True):
    with SuppressPrints(if_print):
        args = {}
        args['home_path'] = home_path
        args['if_gpu'] = True

        if dataset_name == 'CIFAR-10':
            name_dataset = 'CIFAR-10-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias-no-regularization'
        elif dataset_name == 'CIFAR-100':
            name_dataset = 'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN-no-regularization'
        elif dataset_name == 'MNIST':
            name_dataset = 'MNIST-autoencoder-relu-N1-1000-sum-loss-no-regularization'
        elif dataset_name == 'FACES':
            name_dataset = 'FacesMartens-autoencoder-relu-no-regularization'
        elif dataset_name == 'CURVES':
            name_dataset = 'CURVES-autoencoder-relu-sum-loss-no-regularization'
        elif dataset_name == 'SVHN':
            name_dataset = 'SVHN-vgg11'
        else:
            print('dataset_name')
            print(dataset_name)
            sys.exit()

        name_dataset_legend = name_dataset
        algorithms = []

        for algorithm_jupyter in algorithms_jupyter:
            algorithm = {}
            try:
                algorithm['lr'] = algorithm_jupyter['lr']
            except:
                1
                
            if algorithm_jupyter['name'] == 'KFAC':
                algorithm['params'] = {}
                algorithm['params']['kfac_if_svd'] = False
                if dataset_name in ['CIFAR-10', 'CIFAR-100', 'SVHN']:
                    algorithm['name'] = 'kfac-correctFisher-warmStart-no-max-no-LM-momentum-grad-LRdecay'
                    algorithm['params']['weight_decay'] = algorithm_jupyter['weight_decay']
                    mbngd_legend = ',wd=' + str(algorithm_jupyter['weight_decay'])
                    
                elif dataset_name in ['MNIST', 'FACES', 'CURVES']:
                    algorithm['name'] = 'kfac-correctFisher-warmStart-no-max-no-LM-momentum-grad'
                    algorithm['params']['kfac_damping_lambda'] = algorithm_jupyter['damping_value']
                    mbngd_legend = ',dp=' + str(algorithm_jupyter['damping_value'])
                    
                algorithm['legend'] = 'KFAC'
                if full_legend:
                    algorithm['legend'] = 'KFAC'+ mbngd_legend
                algorithms.append(copy.deepcopy(algorithm))


            elif algorithm_jupyter['name'] == 'MBF':
                algorithm['params'] = {}
                algorithm['params']['mbngd_damping_lambda'] = algorithm_jupyter['mbf_damping_lambda']
                mbngd_legend = ',dp=' + str(algorithm_jupyter['mbf_damping_lambda'])
                
                if dataset_name in ['CIFAR-10', 'CIFAR-100', 'SVHN']:
                    algorithm['name'] = 'MBNGD-all-to-one-Avg-LRdecay'
                    algorithm['params']['weight_decay'] = algorithm_jupyter['weight_decay']
                    mbngd_legend += ',wd=' + str(algorithm_jupyter['weight_decay'])
                elif dataset_name in ['MNIST', 'FACES', 'CURVES']:
                    algorithm['name'] = 'MBNGD-all-to-one-Avg-LRdecay'
                try:
                    algorithm['params']['kfac_cov_update_freq'] = algorithm_jupyter['T1']
                    mbngd_legend += ',T1=' + str(algorithm_jupyter['T1'])
                except:
                    1
                try:
                    algorithm['params']['kfac_inverse_update_freq'] = algorithm_jupyter['T2']
                    mbngd_legend += ',T2=' + str(algorithm_jupyter['T2'])
                except:
                    1
                if full_legend:
                    algorithm['legend'] = 'MBF'+ mbngd_legend
                else:
                    algorithm['legend'] = 'MBF'
                algorithms.append(copy.deepcopy(algorithm))

            elif algorithm_jupyter['name'] == 'Shampoo':
                algorithm['params'] = {}
                if dataset_name in ['CIFAR-10', 'CIFAR-100', 'SVHN']:
                    algorithm['name'] = 'shampoo-allVariables-filterFlattening-warmStart-momentum-grad-LRdecay'
                    algorithm['params']['weight_decay'] = algorithm_jupyter['weight_decay']
                    mbngd_legend = ',wd=' + str(algorithm_jupyter['weight_decay'])
                elif dataset_name in ['MNIST', 'FACES', 'CURVES']:
                    algorithm['name'] = 'shampoo-allVariables-filterFlattening-warmStart-lessInverse-momentum-grad'
                    algorithm['params']['shampoo_epsilon'] = algorithm_jupyter['damping_value']
                    mbngd_legend = ',dp=' + str(algorithm_jupyter['damping_value'])
                algorithm['legend'] = 'Shampoo'
                if full_legend:
                    algorithm['legend'] = 'Shampoo'+ mbngd_legend
                algorithms.append(copy.deepcopy(algorithm))

            elif algorithm_jupyter['name'] == 'Adam':
                algorithm['params'] = {}
                if dataset_name in ['CIFAR-10', 'CIFAR-100', 'SVHN']:
                    algorithm['name'] = 'Adam-noWarmStart-momentum-grad-LRdecay'
                    algorithm['params']['weight_decay'] = algorithm_jupyter['weight_decay']
                    mbngd_legend = ',wd=' + str(algorithm_jupyter['weight_decay'])
                elif dataset_name in ['MNIST', 'FACES', 'CURVES']:
                    algorithm['name'] = 'Adam-noWarmStart-momentum-grad'
                    algorithm['params']['RMSprop_epsilon'] = algorithm_jupyter['damping_value']
                    mbngd_legend = ',dp=' + str(algorithm_jupyter['damping_value'])
                    
                algorithm['legend'] = 'Adam'
                if full_legend:
                    algorithm['legend'] = 'Adam'+ mbngd_legend
                algorithms.append(copy.deepcopy(algorithm))

            elif algorithm_jupyter['name'] == 'SGD-m':
                algorithm['params'] = {}
                if dataset_name in ['CIFAR-10', 'CIFAR-100', 'SVHN']:
                    algorithm['name'] = 'SGD-LRdecay-momentum'
                    algorithm['params']['weight_decay'] = algorithm_jupyter['weight_decay']
                    mbngd_legend = ',wd=' + str(algorithm_jupyter['weight_decay'])
                    
                elif dataset_name in ['MNIST', 'FACES', 'CURVES']:
                    algorithm['name'] = 'SGD-momentum'
                    mbngd_legend = ''
                algorithm['legend'] = 'SGD-m'
                
                if full_legend:
                    algorithm['legend'] = 'SGD-m'+ mbngd_legend
                algorithms.append(copy.deepcopy(algorithm))
            else:
                print('algorithm_jupyter')
                print(algorithm_jupyter)
                sys.exit()

        if dataset_name in ['CIFAR-10', 'CIFAR-100', 'SVHN']:
            args['tuning_criterion'] = 'test_acc'
            args['list_y'] = ['training unregularized minibatch loss',
                      'testing error']
            args['if_max_epoch'] = 1

        elif dataset_name in ['MNIST', 'FACES', 'CURVES']:
        #     args['tuning_criterion'] = 'test_acc'
            # args['tuning_criterion'] = 'train_loss'
            # args['tuning_criterion'] = 'train_acc'
            # args['tuning_criterion'] = 'train_minibatch_acc'
            args['tuning_criterion'] = 'train_minibatch_loss'
        #     args['list_y'] = ['training unregularized minibatch loss',
        #                   'testing error']
            args['list_y'] = ['training unregularized minibatch loss', 'testing error']
            args['list_y'] = ['training unregularized minibatch loss']
            # args['if_max_epoch'] = 1
            args['if_max_epoch'] = 0
        else:
            print('dataset_name')
            print(dataset_name)
            sys.exit()


        args['list_x'] = ['epoch', 'cpu time']
        args['x_scale'] = 'linear'
        # args['x_scale'] = 'log'

        args['if_lr_in_legend'] = True
        args['if_show_legend'] = True
        args['if_test_mode'] = False

        args['color'] = None
        args['if_title'] = False

        get_plot(name_dataset, name_dataset_legend, algorithms, args)

        return name_dataset, algorithms, args
    
def plot_seed_results_final(name_dataset, algorithms, args, 
                            plot_metrics = ['train_unregularized_minibatch_losses', 'test_acces'],
                            z_value = 1.96, if_max_epoch = True,
                            if_print = True, lw = 2, alpha = 1):
    with SuppressPrints(if_print):
        get_plot_seed_result1(z_value, name_dataset, algorithms, if_max_epoch, plot_metrics, args['home_path'], if_title = False, lw = lw, alpha = alpha)