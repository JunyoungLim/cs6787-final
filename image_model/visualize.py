# load models
from trimap import TRIMAP
from umap import UMAP
from sklearn.decomposition import NMF, PCA

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
BASE_DIR = "images"


# ==============================================================================
# Datasets 
# Entry format: (name, ((X_tr, y_tr), (X_ts, y_ts)))
# ==============================================================================
datasets = [
    #("mnist", load_mnist_data()),
    #("fashion_mnist", load_fashion_mnist_data()),
    ("cifar10", load_cifar10_data()),
]


# ==============================================================================
# Main 
# for embedding and pickling 
# ==============================================================================
for data_name, ((X_tr, y_tr), (X_ts, y_ts)) in datasets:
    # load models
    n_components = 2
    models = [
        ("pca", PCA(n_components=n_components)),
        ("trimap", TRIMAP(n_dims=n_components, apply_pca=False)),
        ("nmf", NMF(n_components=n_components, init='random')),
        ("umap", UMAP(n_components=n_components)),
    ]

    # run embedding
    for model_name, model in models:
        print(model)

        t_start = time.time()
        embedding = model.fit_transform(X_tr)
        duration = time.time() - t_start
        print("took %2.5f seconds" % duration)
        print(embedding.shape)
        
        gs = utils.global_score(X_tr, embedding)
        print("global score %2.2f" % gs)

        plt.figure(figsize=(20,20))
        colors = ['b', 'g', 'r', 'c', 'm','y','plum','darkorange','slategray','salmon']
        target_names = np.unique(y_tr)
        lw = 2
        for color, i, target_name in zip(colors, target_names, target_names):
            plt.scatter(embedding[y_tr == i, 0], embedding[y_tr == i, 1], color=color, alpha=.8, lw=lw, label=target_name, s=1)
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.title('{} of {} dataset'.format(model_name, data_name))
        plt.savefig(os.path.join(BASE_DIR, '{}_{}.png'.format(data_name, model_name)))


print('Done')


