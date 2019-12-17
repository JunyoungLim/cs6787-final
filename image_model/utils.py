import numpy as np
from sklearn.decomposition import PCA

def global_loss(X, Y):
    X = X - np.mean(X, axis=0)
    Y = Y - np.mean(Y, axis=0)
    A = X.T @ (Y @ np.linalg.inv(Y.T @ Y))
    return np.mean(np.power(X.T - A @ Y.T, 2))

def global_score(X, Y):
        """
        Global score
        Input
        ------
        X: Instance matrix
        Y: Embedding
        """
        n_dims = Y.shape[1]
        Y_pca = PCA(n_components=n_dims).fit_transform(X)
        gs_pca = global_loss(X, Y_pca)
        gs_emb = global_loss(X, Y)
        return np.exp(-(gs_emb - gs_pca) / gs_pca)