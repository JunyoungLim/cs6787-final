import keras
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.losses import categorical_crossentropy
from keras import utils
from keras import backend as K

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

from dataset import *

import pickle, os

# This random seed is to have the experiment to run over
# the same train/validation sets consistently.
RANDOM_SEED = 6787

NUM_EPOCHS = 100
NUM_CLASSES = 10
IMG_DIM = 4


# normalization of the features and categorical conversion of the labels
def preprocess_feature_label_pairs(X, Y):
    # Image data format = channels_last
    X = X.reshape(X.shape[0], IMG_DIM, IMG_DIM, 1)
    X = X.astype('float32') / 255
    # convert class vectors to binary class matrices
    Y = utils.to_categorical(Y, NUM_CLASSES)
    return X, Y
    

def experiment(minibatch_size, optimizer, num_channels, out_dim_dense, data_name, model_name):
    
    # Load the data and split train/validation
    pickled_file = os.path.join("embeddings", "embedding_{}_{}".format(data_name, model_name))
    Xs_tr, Ys_tr, Xs_te, Ys_te, dr_model = pickle.load(open( pickled_file, "rb" ))
    Xs_tr, Xs_vl, Ys_tr, Ys_vl = train_test_split(Xs_tr, Ys_tr, test_size=0.1, random_state=RANDOM_SEED)
    
    # preprocess the data for the neural network
    Xs_tr, Ys_tr = preprocess_feature_label_pairs(Xs_tr, Ys_tr)
    Xs_vl, Ys_vl = preprocess_feature_label_pairs(Xs_vl, Ys_vl)
    Xs_te, Ys_te = preprocess_feature_label_pairs(Xs_te, Ys_te)
    input_shape = (IMG_DIM, IMG_DIM, 1)

    print("Training on:",
          Xs_tr.shape[0], 'train samples,',
          Xs_vl.shape[0], 'validation samples,',
          Xs_te.shape[0], 'test samples')

    model = Sequential()
    # A 2D convolution layer using (3×3) filter size, with 32 channels, and a ReLU activation.
    model.add(Conv2D(num_channels, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    # A 2D MaxPool layer with a (2×2) downsampling factor.
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Another 2D convolution layer using (3×3) filter size, with 32 channels, and a ReLU activation.
    #model.add(Conv2D(num_channels, (3, 3), activation='relu'))
    # Another 2D MaxPool layer with a (2×2) downsampling factor.
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    # A dense layer with a 128-dimensional output and a ReLU activation.
    model.add(Flatten())
    model.add(Dense(out_dim_dense, activation='relu'))
    # A softmax layer with a 10-dimensional output and a softmax activation,
    # which is the final layer of the network and maps to a distribution over the 10 classes of the MNIST dataset
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    model.compile(loss=categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    
    # training loss/error, validation error before training
    init_train_loss, init_train_accuracy = model.evaluate(Xs_tr, Ys_tr, verbose=0)
    init_validation_loss, init_validation_accuracy = model.evaluate(Xs_vl, Ys_vl, verbose=0)

    history = model.fit(Xs_tr, Ys_tr,
                        batch_size=minibatch_size,
                        epochs=NUM_EPOCHS,
                        verbose=1,
                        validation_data=(Xs_vl, Ys_vl))
    
    score = model.evaluate(Xs_te, Ys_te, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    tr_loss  = [init_train_loss] + history.history['loss']
    tr_err   = [1 - init_train_accuracy] + [1 - x for x in history.history['accuracy']]
    val_loss = [init_validation_loss] + history.history['val_loss']
    val_err  = [1 - init_validation_accuracy] + [1 - x for x in history.history['val_accuracy']]
    return tr_loss, val_loss, tr_err, val_err

    
# Generate the plot with training loss/error, and validation error
def generate_plot(models, plot_title):
    names = []
    errs = []
    for model, err in models.items():
        names += [model]
        errs += err
        plt.plot(np.arange(NUM_EPOCHS+1), err)
    # plt.xticks(np.arange(NUM_EPOCHS+1))
    y_max = max(errs)
    # y_max = 0.5
    plt.yticks(np.arange(0, y_max, y_max / 10))
    plt.title(plot_title)
    plt.ylabel('')
    plt.xlabel('epoch')
    plt.legend(names, loc='upper right')
    plt.savefig(os.path.join("images", plot_title + ".png"))
    plt.show()


if __name__ == '__main__':
     
    tr_loss_1, val_loss_1, tr_err_1, val_err_1 = experiment(minibatch_size=128,
                                                            optimizer=optimizers.SGD(lr=0.001, momentum=0.99),
                                                            num_channels=32,
                                                            out_dim_dense=128,
                                                            data_name="cifar10",
                                                            model_name="pca")
        
    tr_loss_2, val_loss_2, tr_err_2, val_err_2 = experiment(minibatch_size=128,
                                                            optimizer=optimizers.SGD(lr=0.001, momentum=0.99),
                                                            num_channels=32,
                                                            out_dim_dense=128,
                                                            data_name="cifar10",
                                                            model_name="nmf")

    tr_loss_3, val_loss_3, tr_err_3, val_err_3 = experiment(minibatch_size=128,
                                                            optimizer=optimizers.SGD(lr=0.001, momentum=0.99),
                                                            num_channels=32,
                                                            out_dim_dense=128,
                                                            data_name="cifar10",
                                                            model_name="umap")

    models = {
        "pca": tr_loss_1,
        "nmf": tr_loss_2,
        "umap": tr_loss_3,
    }
    generate_plot(models, "Training Loss on CIFAR-10")

    models = {
        "pca": tr_err_1,
        "nmf": tr_err_2,
        "umap": tr_err_3,
    }
    generate_plot(models, "Training Error on CIFAR-10")

    models = {
        "pca": val_loss_1,
        "nmf": val_loss_2,
        "umap": val_loss_3,
    }
    generate_plot(models, "Validation Loss on CIFAR-10")

    models = {
        "pca": val_err_1,
        "nmf": val_err_2,
        "umap": val_err_3,
    }
    generate_plot(models, "Validation Error on CIFAR-10")
    print("Done")   
