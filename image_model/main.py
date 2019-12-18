
# ==============================================================================
# Main 
# for experimenting and testing
# ==============================================================================
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
(Xs_tr, Ys_tr), (Xs_te, Ys_te) = fashion_mnist.load_data()

Xs_tr = Xs_tr.reshape(Xs_tr.shape[0], 784)

plt.figure(figsize=(8,4))

plt.subplot(1, 2, 1)
plt.imshow(Xs_tr[0].reshape(28,28),
              cmap = plt.cm.gray, interpolation='nearest',
              clim=(0, 255))
plt.xlabel('784 components', fontsize = 14)
plt.title('Original Image', fontsize = 20)

from sklearn.decomposition import PCA
pca = PCA(n_components=256)
Xs_tr = pca.fit_transform(Xs_tr)
plt.subplot(1, 2, 2);
plt.imshow(Xs_tr[0].reshape(16, 16),
              cmap = plt.cm.gray, interpolation='nearest',
              clim=(0, 255));
plt.xlabel('256 components', fontsize = 14)
plt.title('Reduced Image', fontsize = 20)
plt.savefig('sample.png')