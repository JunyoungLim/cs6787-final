# load models
from trimap import TRIMAP
from umap import UMAP
from sklearn.decomposition import NMF, PCA
from sklearn.preprocessing import MinMaxScaler

# load datasets
from dataset import *

# load auxiliary packages
import matplotlib.pyplot as plt
import utils
import time
import os, pickle


# ==============================================================================
# Global Variables 
# ==============================================================================
BASE_DIR = "embeddings"


# ==============================================================================
# Datasets 
# Entry format: (name, ((X_tr, y_tr), (X_ts, y_ts)))
# ==============================================================================
datasets = [
    ("mnist", load_mnist_data()),
    # ("fashion_mnist", load_fashion_mnist_data()),
    # ("cifar10", load_cifar10_data()),
]


# ==============================================================================
# Main 
# for embedding and pickling 
# ==============================================================================
for data_name, ((X_tr, y_tr), (X_ts, y_ts)) in datasets:
    # load models
    n_components = 9
    models = [
        # ("pca", PCA(n_components=n_components)),
        ("trimap", TRIMAP(n_dims=n_components, apply_pca=False)),
        # ("nmf", NMF(n_components=n_components, init='random')),
        #("umap", UMAP(n_components=n_components)),
    ]

    # run embedding
    for model_name, model in models:
        print(model)

        t_start = time.time()
        X_tr = model.fit_transform(X_tr)
        X_te = model.fit_transform(X_te)
        duration = time.time() - t_start
        print("took %2.5f seconds" % duration)

        file_name = os.path.join(BASE_DIR, "embedding_{}_{}".format(data_name, model_name))
        pickle.dump((X_tr, y_tr, X_ts, y_ts, None), open(file_name, "wb"))


print('Done')