import os
import argparse
import numpy as np
import json
import pickle
import time
import sklearn.cluster as cluster

class Timer:
    DEBUG = True

    def __init__(self, name='task'):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.DEBUG:
            print('{} consumes {} s'.format(self.name, time.time() - self.start))
        return exc_type is None

def load_knn(ofn):
    if ofn.endswith('.pkl'):
        return pickle.load(open(ofn, 'rb'))
    elif ofn.endswith('.json'):
        return json.load(open(ofn, 'r'))
    else:
        raise ValueError('Unknown suffix: {}. Only support .pkl and .json.'.format(ofn))

def dump2json(ofn, data, force=False):
    if os.path.exists(ofn) and not force:
        return

    def default(obj):
        if isinstance(obj, np.int32):
            return int(obj)
        elif isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, set) or isinstance(obj, np.ndarray):
            return list(obj)
        else:
            raise TypeError(
            "Unserializable object {} of type {}".format(obj, type(obj)))

    with open(ofn, 'w') as of:
        json.dump(data, of, default=default)

def normalize(vec):
    vec /= np.linalg.norm(vec, axis=1).reshape(-1,1)
    return vec

def read_meta(fn_meta):
    lb2idxs = {}
    lbs = []
    with open(fn_meta) as f:
        for idx, x in enumerate(f.readlines()[1:]):
            lb = int(x.strip())
            if lb not in lb2idxs:
                lb2idxs[lb] = []
            lb2idxs[lb] += [idx]
            lbs.append(lb)

    inst_num = len(lbs)
    cls_num = len(lb2idxs)
    print('#cls: {}, #inst: {}'.format(cls_num, inst_num))
    return lb2idxs, lbs, cls_num, inst_num

def labels2clusters(labels):
    lb2idxs = {}
    for idx, lb in enumerate(labels):
        if lb not in lb2idxs:
            lb2idxs[lb] = []
        lb2idxs[lb].append(idx)
    clusters = [idxs for _, idxs in lb2idxs.items()]
    return clusters


def KMeans(feat, n_clusters=2):
    kmeans = cluster.KMeans(n_clusters=n_clusters,
                            n_jobs=8,
                            random_state=0).fit(feat)
    return kmeans.labels_


def MiniBatchKMeans(feat, n_clusters=2, batch_size=6):
    kmeans = cluster.MiniBatchKMeans(n_clusters=n_clusters,
                                    batch_size=batch_size,
                                    random_state=0).fit(feat)
    return kmeans.labels_


def spectral(feat, n_clusters=2):
    spectral = cluster.SpectralClustering(n_clusters=n_clusters,
                                        assign_labels="discretize",
                                        affinity="nearest_neighbors",
                                        random_state=0).fit(feat)
    return spectral.labels_


def hierarchy(feat, n_clusters=2, knn=30):
    from sklearn.neighbors import kneighbors_graph
    knn_graph = kneighbors_graph(feat, knn, include_self=False)
    hierarchy = cluster.AgglomerativeClustering(n_clusters=n_clusters,
                                                connectivity=knn_graph,
                                                linkage='ward').fit(feat)
    return hierarchy.labels_


def fast_hierarchy(feat, distance=0.7, hmethod='single'):
    import fastcluster
    import scipy.cluster
    links = fastcluster.linkage_vector(feat,
                                       method=hmethod)
    labels_ = scipy.cluster.hierarchy.fcluster(links,
                                               distance,
                                               criterion='distance')
    return labels_


def dbscan(feat, eps=0.3, min_samples=10):
    db = cluster.DBSCAN(eps=eps, min_samples=min_samples, n_jobs=8).fit(feat)
    return db.labels_


def knn_dbscan(sparse_affinity, eps=0.75, min_samples=10):
    db = cluster.DBSCAN(eps=eps, min_samples=min_samples, n_jobs=8, metric='precomputed').fit(sparse_affinity)
    return db.labels_


def hdbscan(feat, min_samples=10):
    import hdbscan
    db = hdbscan.HDBSCAN(min_cluster_size=min_samples)
    labels_ = db.fit_predict(feat)
    return labels_


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SKLearn Clustering')
    parser.add_argument('--data', type=str)
    parser.add_argument('--feature', type=str)
    parser.add_argument('--feat-dim', default=256, type=int)
    parser.add_argument('--ncluster', default=1000, type=int)
    parser.add_argument('--batch-size', default=100, type=int)
    parser.add_argument('--eps', default=0.7, type=float)
    parser.add_argument('--min-samples', default=10, type=int)
    parser.add_argument('--knn', default=30, type=int)
    parser.add_argument('--hmethod', default='single', type=str)
    parser.add_argument('--method', type=str)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    assert args.method
    assert 1 <= args.split_num <= 9

    method2op = {
        'kmeans': KMeans,
        'mini_batch_kmeans': MiniBatchKMeans,
        'spectral': spectral,
        'hierarchy': hierarchy,
        'fast_hierarchy': fast_hierarchy,
        'dbscan': dbscan,
        'hdbscan': hdbscan,
        'knn_dbscan': knn_dbscan,
    }

    cluster_func = method2op[args.method]

    ofn_prefix = 'baseline_output/'

    if args.method == 'dbscan' or args.method == 'knn_dbscan':
        ofn = os.path.join(ofn_prefix, '{}_{}_eps_{}_min_{}/clusters.json'.format(args.data, args.method, args.eps, args.min_samples))
    elif args.method == 'hdbscan':
        ofn = os.path.join(ofn_prefix, '{}_{}_min_{}/clusters.json'.format(args.data, args.method, args.min_samples))
    elif args.method == 'fast_hierarchy':
        ofn = os.path.join(ofn_prefix, '{}_{}_eps_{}_hmethod_{}/clusters.json'.format(args.data, args.method, args.eps, args.hmethod))
    elif args.method == 'hierarchy':
        ofn = os.path.join(ofn_prefix, '{}_{}_ncluster_{}_knn_{}/clusters.json'.format(args.data, args.method, args.ncluster, args.knn))
    elif args.method == 'mini_batch_kmeans':
        ofn = os.path.join(ofn_prefix, '{}_{}_ncluster_{}_bs_{}/clusters.json'.format(args.data, args.method, args.ncluster, args.batch_size))
    else:
        ofn = os.path.join(ofn_prefix, '{}_{}_ncluster_{}/clusters.json'.format(args.data, args.method, args.ncluster))

    if os.path.exists(ofn) and not args.force:
        print('{} has already existed. Please set force=True to overwrite.'.format(ofn))
        exit()
    if not os.path.exists(os.path.dirname(ofn)):
        os.makedirs(os.path.dirname(ofn))

    feat_dim = args.feat_dim

    lb2idxs, idx2lb, cls_num, inst_num = read_meta("data/unlabeled/{}/meta.txt".format(args.data))
    feat = np.fromfile("data/unlabeled/{}/features/{}.bin".format(args.data, args.feature), dtype=np.float32, count=inst_num*feat_dim).reshape(inst_num, feat_dim)
    feat = normalize(feat)

    with Timer('{}'.format(args.method)):
        if args.method == 'dbscan':
            labels = cluster_func(feat, eps=args.eps, min_samples=args.min_samples)
        elif args.method == 'knn_dbscan':
            from scipy.sparse import csr_matrix
            # load knn and construct sparse mat
            knn_fn = 'data/unlabeled/{}/knn/{}_k{}.json'.format(args.data, args.feature, args.knn)
            knns = load_knn(knn_fn)
            row, col, data = [], [], []
            for row_i, knn in enumerate(knns):
                ns, dists = knn
                for n, dist in zip(ns, dists):
                    if 1 - dist < 0.7:
                        continue
                    row.append(row_i)
                    col.append(n)
                    data.append(dist)
            sparse_affinity = csr_matrix((data, (row, col)), shape=(inst_num, inst_num))
            # clustering
            labels = cluster_func(sparse_affinity, eps=args.eps, min_samples=args.min_samples)
        elif args.method == 'hdbscan':
            labels = cluster_func(feat, min_samples=args.min_samples)
        elif args.method == 'fast_hierarchy':
            labels = cluster_func(feat, distance=args.eps, hmethod=args.hmethod)
        elif args.method == 'hierarchy':
            labels = cluster_func(feat, n_clusters=args.ncluster, knn=args.knn)
        elif args.method == 'mini_batch_kmeans':
            labels = cluster_func(feat, n_clusters=args.ncluster, batch_size=args.batch_size)
        else:
            labels = cluster_func(feat, n_clusters=args.ncluster)

        clusters = labels2clusters(labels)
        # dump to json
        print('dump clusters to', ofn)
        dump2json(ofn, clusters, force=args.force)

