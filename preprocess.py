import numpy as np
import faiss
import pdb

def preprocess_features(npdata, pca=256):
    """Preprocess an array of features.
    Args:
        npdata (np.array N * ndim): features to preprocess
        pca (int): dim of output
    Returns:
        np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
    """
    _, ndim = npdata.shape
    npdata =  npdata.astype('float32')

    # Apply PCA-whitening with Faiss
    mat = faiss.PCAMatrix (ndim, pca, eigen_power=-0.5)
    mat.train(npdata)
    assert mat.is_trained
    npdata = mat.apply_py(npdata)

    # L2 normalization
    row_sums = np.linalg.norm(npdata, axis=1)
    npdata = npdata / row_sums[:, np.newaxis]

    return npdata

if  __name__ == "__main__":
    pca = 256
    features = np.fromfile('data/unlabeled/imgnet_rotation/features/res50_rotation.bin',
                           dtype=np.float32).reshape(1281167, -1)
    pca_feat = preprocess_features(features, pca)
    pca_feat.tofile('data/unlabeled/imgnet_rotation/features/res50_rotation_pca.bin')
