import sys
import torch
import os
import pickle
import numpy as np

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import gzip

import urllib.request

def load_downScaledMNIST(batch_size, test_batch_size, train_dir):
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.Resize([16, 16]),
        transforms.ToTensor(),
        transforms.Normalize((0.,), (1.0,)),
    ])

    # Normalize the test set same as training set without augmentation
    transform_test = transforms.Compose([
        transforms.Resize([16, 16]),
        transforms.ToTensor(),
        transforms.Normalize((0.,), (1.0,)),
    ])
    
    
    
    root_dir = train_dir

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root=root_dir, train=True, transform=transform_train, download=True),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root=root_dir, train=False, transform=transform_test,download=True),
        batch_size=test_batch_size,
        shuffle=False,
        pin_memory=True)
    
    return train_loader, test_loader

def load_cifar100(name_dataset, home_path, batch_size, test_batch_size):
    print('==> Preparing data..')
    if name_dataset == 'CIFAR-100-NoAugmentation':
        transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
    ])
    elif name_dataset in ['CIFAR-100',
                          'CIFAR-100-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias',
                          'CIFAR-100-onTheFly-ResNet34-BNNoAffine',
                          'CIFAR-100-onTheFly-ResNet34-BN',
                          'CIFAR-100-onTheFly-ResNet34-BN-BNshortcut',
                          'CIFAR-100-onTheFly-ResNet34-BN-BNshortcutDownsampleOnly',
                          'CIFAR-100-onTheFly-ResNet34-BN-BNshortcutDownsampleOnly-NoBias',
                          'CIFAR-100-onTheFly-N1-128-ResNet34-BN-BNshortcutDownsampleOnly-NoBias',
                          'CIFAR-100-onTheFly-N1-128-ResNet34-BN-PaddingShortcutDownsampleOnly-NoBias',
                          'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout',
                          'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine-no-regularization',
                          'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN',
                          'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN-no-regularization',
                          'CIFAR-100-onTheFly-vgg16-NoLinear-BN-no-regularization',
                          'CIFAR-100-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout',
                          'CIFAR-100-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout-no-regularization',
                          'CIFAR-100-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine',
                          'CIFAR-100-onTheFly-AllCNNC']:
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
    ])
    else:
        print('error: need to check for ' + name_dataset)
        sys.exit()
    # Normalize the test set same as training set without augmentation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root=home_path + 'data/' + name_dataset + '_data', train=True, transform=transform_train, download=True),
        batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root=home_path + 'data/' + name_dataset + '_data', train=False, transform=transform_test,download=True),
        batch_size=test_batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, test_loader

def load_svhn(name_dataset, home_path, batch_size, test_batch_size):
    print('==> Preparing data..')
    
    transform_train = transforms.Compose([
        transforms.Pad(padding=2),
        transforms.RandomCrop(size=(32, 32)),
        transforms.ColorJitter(brightness=63. / 255., saturation=[0.5, 1.5], contrast=[0.2, 1.8]),
        transforms.ToTensor(),
        transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
    ])
     
    # Normalize the test set same as training set without augmentation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614)),
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.SVHN(root=home_path + 'data/' + name_dataset + '_data', split = 'train', transform=transform_train, download=True),
        batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.SVHN(root=home_path + 'data/' + name_dataset + '_data', split = 'test', transform=transform_test,download=True),
        batch_size=test_batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    return train_loader, test_loader

def load_cifar(name_dataset, home_path, batch_size, test_batch_size):
    print('==> Preparing data..')
    if name_dataset in ['CIFAR-10-NoAugmentation-vgg11',
                        'CIFAR-10-NoAugmentation-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout']:
        # https://arxiv.org/pdf/1811.03600.pdf
        transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    elif name_dataset in ['CIFAR',
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
                          'CIFAR-10-AllCNNC',
                          'CIFAR-10-N1-128-AllCNNC',
                          'CIFAR-10-N1-512-AllCNNC']:
        # RandomCrop and RandomHorizontalFlip do NOT affect normalization,
        # because their order can be exchanged
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]) 
    else:
        print('error: need to check for ' + name_dataset)
        sys.exit()
    # (0.2023, 0.1994, 0.2010): AVERAGE std per image (degress of freedom = 1),
    # NOT std of the tensor of size (50000, 32, 32)
    
    # other normalization:
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    # see https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/trainer.py

    # Normalize the test set same as training set without augmentation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=home_path + 'data/' + name_dataset + '_data', train=True, transform=transform_train, download=True),
        batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=home_path + 'data/' + name_dataset + '_data', train=False, transform=transform_test,download=True),
        batch_size=test_batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    return train_loader, test_loader

def extract_labels(filename, one_hot=False):
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                'Invalid magic number %d in MNIST label file: %s' %
                (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        if one_hot:
            return dense_to_one_hot(labels)
        return labels

def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]

def extract_images(filename):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                'Invalid magic number %d in MNIST image file: %s' %
                (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        

        data = data.reshape(num_images, rows, cols, 1)
        
        return data

def maybe_download(SOURCE_URL, filename, work_directory):
    """Download the data from Yann's website, unless it's already here."""
    if not os.path.exists(work_directory):
        os.makedirs(work_directory)
    filepath = os.path.join(work_directory, filename)

    if not os.path.exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
    return filepath

class DataSet(object):
    def __init__(self, images, labels, if_autoencoder, input_reshape, fake_data=False):
        if fake_data:
            self._num_examples = 10000
        else:
            assert images.shape[0] == labels.shape[0], (
                "images.shape: %s labels.shape: %s" % (images.shape,
                                                       labels.shape))
            self._num_examples = images.shape[0]
            if input_reshape == 'fully-connected':
                images = np.swapaxes(images, 2, 3)
                images = np.swapaxes(images, 1, 2)
                images = images.reshape(images.shape[0],
                                    images.shape[1] * images.shape[2] * images.shape[3])
            elif input_reshape == '2d-CNN':
                images = np.moveaxis(images, [1, 2, 3], [2, 3, 1])
        
            elif input_reshape == '1d-CNN':      
                images = images.squeeze(-1)
                images = np.swapaxes(images, 1, 2)

            images = images.astype(np.float32)
            if if_autoencoder:
                labels = images
            

        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1.0 for _ in range(784)]
            fake_label = 0
            return [fake_image for _ in range(batch_size)], [
                fake_label for _ in range(batch_size)]
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]

def read_data_sets(name_dataset, name_model, home_path, fake_data=False, one_hot=False):
    
    if name_dataset in ['MNIST-N1-1000',
                        'MNIST-no-regularization',
                        'MNIST-one-layer']:
        name_dataset = 'MNIST'
    elif name_dataset in ['DownScaledMNIST-N1-1000-no-regularization']:
        name_dataset = 'DownScaledMNIST-no-regularization'
    elif name_dataset in ['MNIST-autoencoder-no-regularization',
                          'MNIST-autoencoder-N1-1000',
                          'MNIST-autoencoder-N1-1000-no-regularization',
                          'MNIST-autoencoder-N1-1000-sum-loss-no-regularization',
                          'MNIST-autoencoder-relu-N1-1000-sum-loss-no-regularization',
                          'MNIST-autoencoder-relu-N1-1000-sum-loss',
                          'MNIST-autoencoder-relu-N1-100-sum-loss',
                          'MNIST-autoencoder-relu-N1-500-sum-loss',
                          'MNIST-autoencoder-relu-N1-1-sum-loss',
                          'MNIST-autoencoder-reluAll-N1-1-sum-loss',
                          'MNIST-autoencoder-N1-1000-sum-loss']:
        name_dataset = 'MNIST-autoencoder'
    elif name_dataset in ['FACES-autoencoder-no-regularization',
                          'FACES-autoencoder-sum-loss-no-regularization',
                          'FACES-autoencoder-relu-sum-loss-no-regularization',
                          'FACES-autoencoder-relu-sum-loss',
                          'FACES-autoencoder-sum-loss']:
        name_dataset = 'FACES-autoencoder'
    elif name_dataset in ['FacesMartens-autoencoder-relu-no-regularization',
                          'FacesMartens-autoencoder-relu-N1-500',
                          'FacesMartens-autoencoder-relu-N1-100',]:
        name_dataset = 'FacesMartens-autoencoder-relu'
    elif name_dataset in ['CIFAR-deep',
                          'CIFAR-10-vgg16',
                          'CIFAR-10-vgg11',
                          'CIFAR-10-vgg16-NoAdaptiveAvgPoolNoDropout',
                          'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout',
                          'CIFAR-10-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout',
                          'CIFAR-10-onTheFly-N1-256-vgg16-NoAdaptiveAvgPool',
                          'CIFAR-10-onTheFly-N1-512-vgg16-NoAdaptiveAvgPoolNoDropout',
                          'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN',
                          'CIFAR-10-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine',
                          'CIFAR-10-onTheFly-ResNet32-BNNoAffine',
                          'CIFAR-10-onTheFly-ResNet32-BN',
                          'CIFAR-10-onTheFly-ResNet32-BN-BNshortcut',
                          'CIFAR-10-onTheFly-ResNet32-BN-BNshortcutDownsampleOnly',
                          'CIFAR-10-onTheFly-ResNet32-BN-BNshortcutDownsampleOnly-NoBias',
                          'CIFAR-10-onTheFly-N1-128-ResNet32-BN-BNshortcutDownsampleOnly-NoBias',
                          'CIFAR-10-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias',
                          'CIFAR-10-AllCNNC',
                          'CIFAR-10-N1-128-AllCNNC',
                          'CIFAR-10-N1-512-AllCNNC']:
        name_dataset = 'CIFAR'
    elif name_dataset in ['CIFAR-10-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout',
                          'CIFAR-10-NoAugmentation-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout',
                          'CIFAR-10-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout-BN',
                          'CIFAR-10-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine']:  
        name_dataset = 'CIFAR-10-NoAugmentation-vgg11'
    elif name_dataset in ['CIFAR-100-NoAugmentation-vgg16-NoAdaptiveAvgPoolNoDropout',
                          'CIFAR-100-NoAugmentation-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout']:
        name_dataset = 'CIFAR-100-NoAugmentation'
    elif name_dataset in ['CIFAR-100-onTheFly-ResNet34-BNNoAffine',
                          'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout',
                          'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN',
                          'CIFAR-100-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout',
                          'CIFAR-100-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine',
                          'CIFAR-100-onTheFly-AllCNNC']:
        name_dataset = 'CIFAR-100'
    elif name_dataset in ['CURVES-autoencoder-no-regularization',
                          'CURVES-autoencoder-sum-loss-no-regularization',
                          'CURVES-autoencoder-relu-sum-loss-no-regularization',
                          'CURVES-autoencoder-relu-sum-loss',
                          'CURVES-autoencoder-relu-N1-100-sum-loss',
                          'CURVES-autoencoder-relu-N1-500-sum-loss',
                          'CURVES-autoencoder-sum-loss',
                          'CURVES-autoencoder-shallow',
                          'CURVES-autoencoder-Botev',
                          'CURVES-autoencoder-Botev-sum-loss-no-regularization']:
        name_dataset = 'CURVES-autoencoder'
    elif name_dataset == 'Subsampled-ImageNet-vgg16':
        name_dataset = 'Subsampled-ImageNet-simple-CNN'
    elif name_dataset == 'sythetic-linear-regression-N1-1':
        name_dataset = 'sythetic-linear-regression'
    elif name_dataset in ['Fashion-MNIST-N1-60',
                          'Fashion-MNIST-N1-60-no-regularization',
                          'Fashion-MNIST-N1-256-no-regularization',
                          'Fashion-MNIST-GAP-N1-60-no-regularization']:
        name_dataset = 'Fashion-MNIST'
    elif name_dataset in ['SVHN-ResNet34']:
        name_dataset = 'SVHN'    
    
        
    
    class DataSets(object):
        pass
    data_sets = DataSets()
    if fake_data:
        data_sets.train = DataSet([], [], fake_data=True)
        data_sets.validation = DataSet([], [], fake_data=True)
        data_sets.test = DataSet([], [], fake_data=True)
        return data_sets
 
    
    train_dir = home_path + 'data/' + name_dataset + '_data'
    
    VALIDATION_SIZE = 0
    
    if name_dataset in ['MNIST',
                        'Fashion-MNIST']:
        if_autoencoder = False
        
        if name_dataset in ['MNIST',
                            'MNIST-one-layer']:
            SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
        elif name_dataset == 'Fashion-MNIST':
            SOURCE_URL =\
            'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
            # https://github.com/zalandoresearch/fashion-mnist
        
        
        TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
        TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
        TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
        TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

    
        local_file = maybe_download(SOURCE_URL, TRAIN_IMAGES, train_dir)
        train_images = extract_images(local_file)
        local_file = maybe_download(SOURCE_URL, TRAIN_LABELS, train_dir)
        train_labels = extract_labels(local_file, one_hot=one_hot)
        local_file = maybe_download(SOURCE_URL, TEST_IMAGES, train_dir)
        test_images = extract_images(local_file)
        local_file = maybe_download(SOURCE_URL, TEST_LABELS, train_dir)
        test_labels = extract_labels(local_file, one_hot=one_hot)

        train_images = np.multiply(train_images, 1.0 / 255.0)
        test_images = np.multiply(test_images, 1.0 / 255.0)
        
        train_labels = np.int64(train_labels)
        test_labels = np.int64(test_labels)
        
        test_train_images = train_images.reshape((train_images.shape[0], -1))
        
        test_A = np.matmul(np.transpose(test_train_images), test_train_images)
        
        print('np.amin(np.diag(test_A))')
        print(np.amin(np.diag(test_A)))
   
    

    elif name_dataset == 'STL-10-simple-CNN':
        
        SOURCE_URL = 'http://ai.stanford.edu/~acoates/stl10/'
        TRAIN_IMAGES = 'stl10_binary.tar.gz'
        
        local_file = maybe_download(SOURCE_URL, TRAIN_IMAGES, train_dir)
        
        extract_images(local_file)
        
        sys.exit()

    elif name_dataset == 'UCI-HAR':
        # https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/
        
        if_autoencoder = False
        
        if os.path.isfile(train_dir + '/' + 'UCI_HAR_data_np.pkl'):
            with open(train_dir + '/' + 'UCI_HAR_data_np.pkl', 'rb') as filename_pkl:
                data_np = pickle.load(filename_pkl)
                
                train_images = data_np['train_images']
                train_labels = data_np['train_labels']
                test_images = data_np['test_images']
                test_labels = data_np['test_labels']
        else:
        
            SOURCE_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00240/'
            IMAGES_ = 'UCI%20HAR%20Dataset.zip'

            local_file = maybe_download(SOURCE_URL, IMAGES_, train_dir)

            import zipfile
            with zipfile.ZipFile(local_file,"r") as zip_ref:
                zip_ref.extractall(train_dir)

            train_images = np.loadtxt(train_dir + '/' + 'UCI HAR Dataset/' + 'train/' + 'X_train.txt')
            train_labels = np.loadtxt(train_dir + '/' + 'UCI HAR Dataset/' + 'train/' + 'y_train.txt')
            test_images = np.loadtxt(train_dir + '/' + 'UCI HAR Dataset/' + 'test/' + 'X_test.txt')
            test_labels = np.loadtxt(train_dir + '/' + 'UCI HAR Dataset/' + 'test/' + 'y_test.txt')

            UCI_HAR_data_np = {}
            UCI_HAR_data_np['train_images'] = train_images
            UCI_HAR_data_np['train_labels'] = train_labels
            UCI_HAR_data_np['test_images'] = test_images
            UCI_HAR_data_np['test_labels'] = test_labels
            with open(train_dir + '/' + 'UCI_HAR_data_np.pkl', 'wb') as filename_pkl:
                pickle.dump(UCI_HAR_data_np, filename_pkl)
                
            os.remove(local_file)
            import shutil
            shutil.rmtree(train_dir + '/' + 'UCI HAR Dataset')

        train_images = train_images[:, :, np.newaxis, np.newaxis]
        test_images = test_images[:, :, np.newaxis, np.newaxis]
        
        train_labels -= 1
        test_labels -= 1
        
        train_labels = np.int64(train_labels)
        test_labels = np.int64(test_labels)
        
    elif name_dataset == 'MNIST-autoencoder':
        
        print('Begin loading data...')
        
        if_autoencoder = True
        
        SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
        
        TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
        TEST_IMAGES = 't10k-images-idx3-ubyte.gz'

        local_file = maybe_download(SOURCE_URL, TRAIN_IMAGES, train_dir)
        train_images = extract_images(local_file)
        
        

        local_file = maybe_download(SOURCE_URL, TEST_IMAGES, train_dir)
        test_images = extract_images(local_file)

        
        # see "Reducing the Dimensionality of Data with Neural Networks"
        train_images = np.multiply(train_images, 1.0 / 255.0)
        test_images = np.multiply(test_images, 1.0 / 255.0)

        train_labels = train_images
        test_labels = test_images
        
    elif name_dataset == 'FACES-autoencoder':
        
        if_autoencoder = True
        
        with open(train_dir + '/' + 'faces.pkl', 'rb') as filename_pkl:
            data_np = pickle.load(filename_pkl)
        
        # data_np[0], 103500, train
        # data_np[1], 20700, validation
        # data_np[2], 41400, test

        train_images = data_np[0][0]
        test_images = data_np[2][0]
        
        train_images = train_images[:, :, np.newaxis, np.newaxis]
        test_images = test_images[:, :, np.newaxis, np.newaxis]
        
        train_labels = train_images
        test_labels = test_images
        
        
    elif name_dataset == 'CURVES-autoencoder':
        if_autoencoder = True
        
        SOURCE_URL = 'http://www.cs.toronto.edu/~jmartens/'
        TRAIN_IMAGES = 'digs3pts_1.mat'
        
        local_file = maybe_download(SOURCE_URL, TRAIN_IMAGES, train_dir)


        import scipy.io as sio
        
        images_ = sio.loadmat(local_file)
   
        train_images = np.asarray(images_['bdata'])
        test_images = np.asarray(images_['bdatatest'])
        
        train_images = train_images[:, :, np.newaxis, np.newaxis]
        test_images = test_images[:, :, np.newaxis, np.newaxis]
        
        train_labels = train_images
        test_labels = test_images
        
    elif name_dataset == 'FacesMartens-autoencoder-relu':
        
        
        
        if_autoencoder = True
        
        SOURCE_URL = 'http://www.cs.toronto.edu/~jmartens/'
        TRAIN_IMAGES = 'newfaces_rot_single.mat'
        
        local_file = maybe_download(SOURCE_URL, TRAIN_IMAGES, train_dir)
        
     
        import scipy.io as sio
        
        images_ = sio.loadmat(local_file)
  
        
            
        images_ = np.asarray(images_['newfaces_single'])

        images_ = np.transpose(images_)
        
        train_images = images_[:103500]
        test_images = images_[-41400:]
        
        train_images = train_images[:, :, np.newaxis, np.newaxis]
        test_images = test_images[:, :, np.newaxis, np.newaxis]
        
        train_labels = train_images
        test_labels = test_images
        
    elif name_dataset == 'Subsampled-ImageNet-simple-CNN':
        if_autoencoder = False
        # need to move YiRen_imagenet_sample/ to train_dir manually
        data_loader = load_subsampled_imagenet(train_dir)
        print('Extracting from data loader...')
        X_Y = next(iter(data_loader))
        
        print('Done Extracting.')
        
        X = X_Y[0]
        Y = X_Y[1]
        
        
        
        X = X.data.numpy()
        Y = Y.data.numpy()
 
        
        X = np.moveaxis(X, [1, 2, 3], [3, 1, 2])

        train_images = X[:100]
        train_labels = Y[:100]
        test_images = X[100:]
        test_labels = Y[100:]
        
    elif name_dataset in ['DownScaledMNIST-no-regularization']:
        if_autoencoder = False
        if os.path.isfile(train_dir + '/' + 'data_np.pkl'):
            with open(train_dir + '/' + 'data_np.pkl', 'rb') as filename_pkl:
                data_np = pickle.load(filename_pkl)
                
                train_images = data_np['train_images']
                train_labels = data_np['train_labels']
                test_images = data_np['test_images']
                test_labels = data_np['test_labels']
                
            
        else:
        
            train_, test_ = load_downScaledMNIST(batch_size=60000, test_batch_size=10000, train_dir=train_dir)

            

            train_ = next(iter(train_))
            test_ = next(iter(test_))
            
            


            train_images = train_[0].data.numpy()
            train_labels = train_[1].data.numpy()
            test_images = test_[0].data.numpy()
            test_labels = test_[1].data.numpy()


            train_images = np.swapaxes(train_images, 1, 2)
            train_images = np.swapaxes(train_images, 2, 3)
            test_images = np.swapaxes(test_images, 1, 2)
            test_images = np.swapaxes(test_images, 2, 3)
   
            

            data_np = {}
            data_np['train_images'] = train_images
            data_np['train_labels'] = train_labels
            data_np['test_images'] = test_images
            data_np['test_labels'] = test_labels
            
            
            
            import shutil
            
            shutil.rmtree(train_dir + '/')
            
            
            os.mkdir(train_dir + '/')

            with open(train_dir + '/' + 'data_np.pkl', 'wb') as filename_pkl:
                pickle.dump(data_np, filename_pkl)
        
    
    elif name_dataset in ['CIFAR',
                          'CIFAR-10-NoAugmentation-vgg11',
                          'CIFAR-100',
                          'CIFAR-100-NoAugmentation']:
        
        
        
        if_autoencoder = False
        
        if os.path.isfile(train_dir + '/' + 'data_np.pkl'):
            with open(train_dir + '/' + 'data_np.pkl', 'rb') as filename_pkl:
                data_np = pickle.load(filename_pkl)
                
                train_images = data_np['train_images']
                train_labels = data_np['train_labels']
                test_images = data_np['test_images']
                test_labels = data_np['test_labels']
          
        else:
            
            
        
            if name_dataset == 'CIFAR':
                train_, test_ = load_cifar(name_dataset, home_path, batch_size=50000, test_batch_size=10000)
                
                
            elif name_dataset == 'CIFAR-10-NoAugmentation-vgg11':
                
                train_, test_ = load_cifar(name_dataset, home_path, batch_size=50000, test_batch_size=10000)
            elif name_dataset in ['CIFAR-100',
                                  'CIFAR-100-NoAugmentation']:
        
                train_, test_ = load_cifar100(name_dataset, home_path, batch_size=50000, test_batch_size=10000)

            train_ = next(iter(train_))
            test_ = next(iter(test_))


            train_images = train_[0].data.numpy()
            train_labels = train_[1].data.numpy()
            test_images = test_[0].data.numpy()
            test_labels = test_[1].data.numpy()
            

            train_images = np.swapaxes(train_images, 1, 2)
            train_images = np.swapaxes(train_images, 2, 3)
            test_images = np.swapaxes(test_images, 1, 2)
            test_images = np.swapaxes(test_images, 2, 3)
    
            data_np = {}
            data_np['train_images'] = train_images
            data_np['train_labels'] = train_labels
            data_np['test_images'] = test_images
            data_np['test_labels'] = test_labels
            
            import shutil
            
            shutil.rmtree(train_dir + '/')
            os.mkdir(train_dir + '/')

            with open(train_dir + '/' + 'data_np.pkl', 'wb') as filename_pkl:
                pickle.dump(data_np, filename_pkl)
    
    elif name_dataset in ['SVHN']:
        if_autoencoder = False
        
        if os.path.isfile(train_dir + '/' + 'data_np.pkl'):
            with open(train_dir + '/' + 'data_np.pkl', 'rb') as filename_pkl:
                data_np = pickle.load(filename_pkl)
                train_images = data_np['train_images']
                train_labels = data_np['train_labels']
                test_images = data_np['test_images']
                test_labels = data_np['test_labels']
        else:
            train_, test_ = load_svhn(name_dataset, home_path, batch_size=50000, test_batch_size=10000)

            train_ = next(iter(train_))
            test_ = next(iter(test_))

            train_images = train_[0].data.numpy()
            train_labels = train_[1].data.numpy()
            test_images = test_[0].data.numpy()
            test_labels = test_[1].data.numpy()
            
            train_images = np.swapaxes(train_images, 1, 2)
            train_images = np.swapaxes(train_images, 2, 3)
            test_images = np.swapaxes(test_images, 1, 2)
            test_images = np.swapaxes(test_images, 2, 3)
  
            data_np = {}
            data_np['train_images'] = train_images
            data_np['train_labels'] = train_labels
            data_np['test_images'] = test_images
            data_np['test_labels'] = test_labels
            
            import shutil
            
            shutil.rmtree(train_dir + '/')
            os.mkdir(train_dir + '/')
            with open(train_dir + '/' + 'data_np.pkl', 'wb') as filename_pkl:
                pickle.dump(data_np, filename_pkl)
        
    elif name_dataset == 'webspam':
        if_autoencoder = False
  
        if os.path.isfile(train_dir + '/webspam_data_np.pkl'):
            with open(train_dir + '/webspam_data_np.pkl', 'rb') as filename_pkl:
                webspam_data_np = pickle.load(filename_pkl)
                x = webspam_data_np['x']
                lines_y = webspam_data_np['lines_y']
        else:
        
            SOURCE_URL = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/'
            file_name = 'webspam_wc_normalized_unigram.svm.bz2'

            local_file = maybe_download(SOURCE_URL, file_name, train_dir)


            import bz2,shutil

            with bz2.BZ2File(train_dir + '/' + "webspam_wc_normalized_unigram.svm.bz2") as fr, open(train_dir + '/' + "output.bin","wb") as fw:
                shutil.copyfileobj(fr,fw)
                
            os.remove(train_dir + '/webspam_wc_normalized_unigram.svm.bz2')

            with open(train_dir + '/output.bin', "r") as file:

                lines = file.readlines()
             
            os.remove(train_dir + '/output.bin')

            lines_1 = [line.split() for line in lines]

            lines_y = [line[0] for line in lines_1]
            lines_x = [line[1:] for line in lines_1]

            from tqdm import tqdm



            x = np.zeros((len(lines_x), 254))
            i = 0

            #set_feature = set()

            for line in tqdm(lines_x):
                #print(line)

                np_line = np.asarray([line_i.split(':') for line_i in line])

                #np_line[:, 1] = np_line[:, 1].astype(np.float)



                #print(np_line[:, 0])

                x[i, np_line[:, 0].astype(np.int) - 1] = np_line[:, 1].astype(np.float)

                #set_feature = set_feature.union(set(list(np_line[:, 0].astype(np.int))))

                i +=1

            lines_y = np.asarray(lines_y)
            lines_y = lines_y.astype(np.int)
            lines_y = (lines_y + 1) / 2

            webspam_data_np = {}
            webspam_data_np['x'] = x
            webspam_data_np['lines_y'] = lines_y

            with open(train_dir + '/webspam_data_np.pkl', 'wb') as filename_pkl:
                pickle.dump(webspam_data_np, filename_pkl)
        
        train_images = x[:300000]
        test_images = x[300000:]
        train_labels = lines_y[:300000]
        test_labels = lines_y[300000:]
        
        
    
        
        print('train_images.shape')
        print(train_images.shape)
        
       
        mean_train = np.mean(train_images, axis=0)
        std_train = np.std(train_images, axis=0)
     
        std_train = np.maximum(std_train, 1 / np.sqrt(len(train_images[0]))) # https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization
        
        print('np.linalg.norm(train_images, ord=1, axis=-1)[:10]')
        print(np.linalg.norm(train_images, ord=1, axis=-1)[:10])
    
        train_images = train_images - mean_train[None, :]
        train_images = train_images / std_train[None, :]

        test_images = test_images - mean_train[None, :]

        test_images = test_images / std_train[None, :]
        
        train_images = train_images[:, :, np.newaxis, np.newaxis]
        test_images = test_images[:, :, np.newaxis, np.newaxis]
        
    elif name_dataset == 'sythetic-linear-regression':
        if_autoencoder = False
        # http://proceedings.mlr.press/v70/zhou17a/zhou17a.pdf
        p = 100
        q = 50
        N = 1000
        
        np.random.seed(127)
        
        # images_: N * p
        mean = np.zeros(p)
        rho = 0.5
        cov = (1-rho**2) * np.eye(p) + rho**2 * np.ones((p, p))
        images_ = np.random.multivariate_normal(mean, cov, size=N)
        
        # coefficient
        beta_ = np.random.rand(p, q)
        
        # labes_: N * q
        
        labels_ = np.matmul(images_, beta_)
        labels_ += np.random.normal(size=labels_.shape)
        
        labels_ = np.float32(labels_)
        images_ = np.float32(images_)
        
        images_ = images_[:, :, np.newaxis, np.newaxis]
        
        train_images = images_[:int(0.9*N)]
        train_labels = labels_[:int(0.9*N)]
        test_images = images_[int(0.9*N):]
        test_labels = labels_[int(0.9*N):]
        
    else:
        print('error: Dataset not supported for ' + name_dataset)
        sys.exit()
        

    validation_images = train_images[:1]
    validation_labels = train_labels[:1]
    # fake
    
    train_images = train_images[VALIDATION_SIZE:]
    train_labels = train_labels[VALIDATION_SIZE:]

    if name_model in ['simple-CNN',
                      'CNN',
                      'vgg16',
                      'vgg11',
                      'ResNet32',
                      'ResNet34',
                      'AllCNNC']:
        input_reshape = '2d-CNN'
    elif name_model in ['1d-CNN']:
        input_reshape = '1d-CNN'
    elif name_model == 'fully-connected':
        input_reshape = 'fully-connected'
    else:
        print('Error: unknown model name for ' + name_model)
        sys.exit()

    data_sets.train = DataSet(train_images, train_labels, if_autoencoder, input_reshape)
    data_sets.validation = DataSet(validation_images, validation_labels, if_autoencoder, input_reshape)
    data_sets.test = DataSet(test_images, test_labels, if_autoencoder, input_reshape)
        
    
    return data_sets

class DataSet_v2():
    
    def __init__(self, train_loader):
        self.train_loader = train_loader
        self.batch_iterator = iter(train_loader)
    
    def next_batch(self, batch_size):
        try:
            return next(self.batch_iterator)
        except:
            self.batch_iterator = iter(self.train_loader)
            return next(self.batch_iterator)

def read_data_sets_v2(name_dataset, params):
    
    class DataSets_v2(object):
        pass
    
    dataset  = DataSets_v2()
    
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
                        'CIFAR-10-AllCNNC',
                        'CIFAR-10-N1-128-AllCNNC',
                        'CIFAR-10-N1-512-AllCNNC',
                        'CIFAR-100-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias',
                        'CIFAR-100-onTheFly-ResNet34-BNNoAffine',
                        'CIFAR-100-onTheFly-ResNet34-BN',
                        'CIFAR-100-onTheFly-ResNet34-BN-BNshortcut',
                        'CIFAR-100-onTheFly-ResNet34-BN-BNshortcutDownsampleOnly',
                        'CIFAR-100-onTheFly-ResNet34-BN-BNshortcutDownsampleOnly-NoBias',
                        'CIFAR-100-onTheFly-N1-128-ResNet34-BN-BNshortcutDownsampleOnly-NoBias',
                        'CIFAR-100-onTheFly-N1-128-ResNet34-BN-PaddingShortcutDownsampleOnly-NoBias',
                        'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout',
                        'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine-no-regularization',
                        'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN',
                        'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN-no-regularization',
                        'CIFAR-100-onTheFly-vgg16-NoLinear-BN-no-regularization',
                        'CIFAR-100-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout',
                        'CIFAR-100-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout-no-regularization',
                        'CIFAR-100-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine',
                        'CIFAR-100-onTheFly-AllCNNC']:
    
        dataset.num_train_data = 50000
    elif name_dataset in ['SVHN-ResNet34', 'SVHN-vgg11']:
        dataset.num_train_data = 73257
    else:
        print('error: need to check for ' + name_dataset)
        sys.exit()
        
    N1 = params['N1']
        
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
                        'CIFAR-10-AllCNNC',
                        'CIFAR-10-N1-128-AllCNNC',
                        'CIFAR-10-N1-512-AllCNNC']:
    
        train_loader, test_loader = load_cifar(name_dataset, params['home_path'], N1, N1)
    elif name_dataset in ['CIFAR-100-onTheFly-N1-128-ResNet32-BN-PaddingShortcutDownsampleOnly-NoBias',
                          'CIFAR-100-onTheFly-ResNet34-BNNoAffine',
                          'CIFAR-100-onTheFly-ResNet34-BN',
                          'CIFAR-100-onTheFly-ResNet34-BN-BNshortcut',
                          'CIFAR-100-onTheFly-ResNet34-BN-BNshortcutDownsampleOnly',
                          'CIFAR-100-onTheFly-ResNet34-BN-BNshortcutDownsampleOnly-NoBias',
                          'CIFAR-100-onTheFly-N1-128-ResNet34-BN-BNshortcutDownsampleOnly-NoBias',
                          'CIFAR-100-onTheFly-N1-128-ResNet34-BN-PaddingShortcutDownsampleOnly-NoBias',
                          'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout',
                          'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine-no-regularization',
                          'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN',
                          'CIFAR-100-onTheFly-vgg16-NoAdaptiveAvgPoolNoDropout-BN-no-regularization',
                          'CIFAR-100-onTheFly-vgg16-NoLinear-BN-no-regularization',
                          'CIFAR-100-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout',
                          'CIFAR-100-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout-no-regularization',
                          'CIFAR-100-onTheFly-N1-256-vgg16-NoAdaptiveAvgPoolNoDropout-BNNoAffine',
                          'CIFAR-100-onTheFly-AllCNNC']:
        train_loader, test_loader = load_cifar100(name_dataset, params['home_path'], N1, N1)
    elif name_dataset in ['SVHN-ResNet34', 'SVHN-vgg11']:
        train_loader, test_loader = load_svhn(name_dataset, params['home_path'], N1, N1)
    else:
        print('error: need to check for ' + name_dataset)
        sys.exit()
    
    dataset.train = DataSet_v2(train_loader)
    dataset.test_generator = test_loader
    
    return dataset
 
def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot
  
def load_subsampled_imagenet(train_dir):
    
    train_path = train_dir + '/' + 'YiRen_imagenet_sample/train/'
    transform = transforms.Compose(
        [transforms.Resize((256,256)), transforms.ToTensor()]
    )


    imagenet_data = datasets.ImageFolder(train_path, transform=transform)
    data_loader = torch.utils.data.DataLoader(
        imagenet_data,
        batch_size=200,
        shuffle=True,
        num_workers=0
    )
    return data_loader
    