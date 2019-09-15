import numpy as np
import time
import faiss
from source import eval_cluster
import preprocess

def run_kmeans(x, nmb_clusters, verbose=False):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """
    n_data, d = x.shape

    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)

    # Change faiss seed at each k-means so that the randomly picked
    # initialization centroids do not correspond to the same feature ids
    # from an epoch to another.
    clus.seed = np.random.randint(1234)

    clus.niter = 20
    clus.max_points_per_centroid = 10000000
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config)

    # perform the training
    clus.train(x, index)
    _, I = index.search(x, 1)
    losses = faiss.vector_to_array(clus.obj)
    if verbose:
        print('k-means loss evolution: {0}'.format(losses))

    return [int(n[0]) for n in I], losses[-1]

class Kmeans:
    def __init__(self, k):
        self.k = k

    def cluster(self, feat, verbose=False):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster
        """
        end = time.time()

        # PCA-reducing, whitening and L2-normalization
        xb = preprocess.preprocess_features(feat)

        # cluster the data
        I, loss = run_kmeans(xb, self.k, verbose)
        self.labels = I
        if verbose:
            print('k-means time: {0:.0f} s'.format(time.time() - end))

        return loss

if __name__ == "__main__":
    num_classes = 1000
    deepcluster = Kmeans(num_classes)
    print("loading features ...")
    features = np.fromfile('data/unlabeled/imgnet_rotation/features/res50_rotation.bin',
                           dtype=np.float32).reshape(1281167, -1)
    with open('data/unlabeled/imgnet_rotation/meta.txt', 'r') as f:
        lines = f.readlines()
    labels = np.array([int(l.strip()) for l in lines])
    print("clustering ...")
    clustering_loss = deepcluster.cluster(features, verbose=True)
    pred = np.array(deepcluster.labels)
    hist = np.bincount(pred, minlength=num_classes)
    minimal_cls_size, maximal_cls_size = hist.min(), hist.max()
    prec, rec, fscore = eval_cluster.fscore(labels, pred)
    print("prec: {:.5g}, rec: {:.5g}, fscore: {:.5g}".format(prec, rec, fscore))
