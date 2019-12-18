import numpy as np
import os, pickle

from keras.datasets import mnist, fashion_mnist

# ==============================================================================
# Global Variables 
# ==============================================================================
BASE_DIR = "data"

# ==============================================================================
# MNIST Digits
# ==============================================================================
def load_mnist_data():
    (Xs_tr, Ys_tr), (Xs_te, Ys_te) = mnist.load_data()

    Xs_tr = Xs_tr.reshape(Xs_tr.shape[0], Xs_tr.shape[1] * Xs_tr.shape[2])
    Xs_te = Xs_te.reshape(Xs_te.shape[0], Xs_te.shape[1] * Xs_te.shape[2])
    return (Xs_tr, Ys_tr), (Xs_te, Ys_te)


# ==============================================================================
# Fashion MNIST Digits
# ==============================================================================
def load_fashion_mnist_data():
    (Xs_tr, Ys_tr), (Xs_te, Ys_te) = fashion_mnist.load_data()

    Xs_tr = Xs_tr.reshape(Xs_tr.shape[0], Xs_tr.shape[1] * Xs_tr.shape[2])
    Xs_te = Xs_te.reshape(Xs_te.shape[0], Xs_te.shape[1] * Xs_te.shape[2])
    return (Xs_tr, Ys_tr), (Xs_te, Ys_te)


# ==============================================================================
# CIFAR-10 Images
# ==============================================================================
def load_cifar10_data():
    data_path = os.path.join(BASE_DIR, "cifar10")
    data_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    data_file = "data_batch_"

    img_size = 32
    num_channels = 3
    img_size_flat = img_size * img_size * num_channels
    num_classes = 10

    _num_files_train = 5
    _num_imgs_per_file = 10000
    _num_imgs_train = _num_files_train * _num_imgs_per_file

    file_name = os.path.join(data_path, data_file + "1")
    f = open(file_name, 'rb')
    data = pickle.load(f, encoding='bytes')
    X = data['data'] #   (50000, 3072)
    y = data['labels'] #   (50000,)

    for i in range(4):
        file_name = os.path.join(data_path, data_file + str(i+2))
        f = open(file_name, 'rb')
        data = pickle.load(f, encoding='bytes')

        X = np.concatenate((X, data['data']))
        y = np.concatenate((y, data['labels']))
        f.close()

    X = X.astype(np.float32)
    y = y.astype(np.float32)

    file_name = os.path.join(data_path, "test_batch")
    f = open(file_name, 'rb')
    data = pickle.load(f, encoding='bytes')
    X_test = np.array(data['data']) #   (10000, 3072)
    y_test = np.array(data['labels']) #   (10000,)
    X_test = X_test.astype(np.float32)
    y_test = y_test.astype(np.float32)
    f.close()

    return (X, y), (X_test, y_test)
