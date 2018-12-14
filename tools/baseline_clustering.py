import os
import argparse
import numpy as np
import json
import pickle
import time
import sklearn.cluster as cluster
import multiprocessing
import sys
sys.path.append("source")
import eval_cluster

import pdb

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
        for idx, x in enumerate(f.readlines()):
            lb = int(x.strip())
            if lb not in lb2idxs:
                lb2idxs[lb] = []
            lb2idxs[lb] += [idx]
            lbs.append(lb)

    inst_num = len(lbs)
    cls_num = len(lb2idxs)
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
                            n_jobs=multiprocessing.cpu_count(),
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
    db = cluster.DBSCAN(eps=eps, min_samples=min_samples, n_jobs=multiprocessing.cpu_count()).fit(feat)
    return db.labels_


def knn_dbscan(sparse_affinity, eps=0.75, min_samples=10):
    db = cluster.DBSCAN(eps=eps, min_samples=min_samples, n_jobs=multiprocessing.cpu_count(), metric='precomputed').fit(sparse_affinity)
    return db.labels_


def hdbscan(feat, min_samples=10):
    import hdbscan
    db = hdbscan.HDBSCAN(min_cluster_size=min_samples)
    labels_ = db.fit_predict(feat)
    return labels_


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SKLearn Clustering')
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--feature', type=str, required=True)
    parser.add_argument('--feat-dim', default=256, type=int)
    parser.add_argument('--ncluster', default=1000, type=int)
    parser.add_argument('--batch-size', default=100, type=int)
    parser.add_argument('--eps', default=0.75, type=float)
    parser.add_argument('--min-samples', default=10, type=int)
    parser.add_argument('--knn', default=30, type=int)
    parser.add_argument('--aro-th', default=2, type=float)
    parser.add_argument('--hmethod', default='single', type=str)
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    assert args.method in ['kmeans', 'mini_batch_kmeans', 'spectral', 'hierarchy', 'fast_hierarchy', 'dbscan', 'hdbscan', 'knn_dbscan', 'approx_rank_order']


    start = time.time()

    with open("data/unlabeled/{}/list.txt".format(args.data), 'r') as f:
        fns = f.readlines()
    inst_num = len(fns)
    ofn_prefix = 'baseline_output/{}_{}_'.format(args.data, args.method)
    print("Method: {}".format(args.method))

    if args.method == 'dbscan' or args.method == 'knn_dbscan':
        ofn = ofn_prefix + 'eps_{}_min_{}/meta.txt'.format(args.eps, args.min_samples)
    elif args.method == 'hdbscan':
        ofn = ofn_prefix + 'min_{}/meta.txt'.format(args.min_samples)
    elif args.method == 'fast_hierarchy':
        ofn = ofn_prefix + 'eps_{}_hmethod_{}/meta.txt'.format(args.eps, args.hmethod)
    elif args.method == 'hierarchy':
        ofn = ofn_prefix + 'ncluster_{}_knn_{}/meta.txt'.format(args.ncluster, args.knn)
    elif args.method == 'mini_batch_kmeans':
        ofn = ofn_prefix + 'ncluster_{}_bs_{}/meta.txt'.format(args.ncluster, args.batch_size)
    elif args.method == "approx_rank_order":
        ofn = ofn_prefix + "knn_{}_th_{}/meta.txt".format(args.knn, args.aro_th)
    else:
        ofn = ofn_prefix + 'ncluster_{}/meta.txt'.format(args.ncluster)

    if os.path.exists(ofn) and not args.force:
        with open(ofn, 'r') as f:
            labels = f.readlines()
        labels = np.array([int(l.strip()) for l in labels])
        print("********\nWarning: the result is loaded from file: {}. If you want to overwrite it, set \"--force\"\n********".format(ofn))

    else:
        if not os.path.exists(os.path.dirname(ofn)):
            os.makedirs(os.path.dirname(ofn))
    
        feat_dim = args.feat_dim
        feat = np.fromfile("data/unlabeled/{}/features/{}.bin".format(args.data, args.feature), dtype=np.float32, count=inst_num*feat_dim).reshape(inst_num, feat_dim)
        feat = normalize(feat)
    
        if args.method == 'dbscan':
            labels = dbscan(feat, eps=args.eps, min_samples=args.min_samples)
        elif args.method == 'knn_dbscan':
            from scipy.sparse import csr_matrix
            # load knn and construct sparse mat
            knn_fn = 'data/unlabeled/{}/knn/{}_k{}.npz'.format(args.data, args.feature, args.knn)
            knn_file = np.load(knn_fn)
            knn_idx, knn_dist = knn_file['idx'], knn_file['dist']
            row, col, data = [], [], []
            for row_i in range(knn_idx.shape[0]):
                ns = knn_idx[row_i]
                dists = knn_dist[row_i]
                valid = np.where(ns > -1)
                ns, dists = ns[valid], dists[valid]
                for n, dist in zip(ns, dists):
                    if 1 - dist < 0.7:
                        continue
                    row.append(row_i)
                    col.append(n)
                    data.append(dist)
            sparse_affinity = csr_matrix((data, (row, col)), shape=(inst_num, inst_num))
            # clustering
            labels = knn_dbscan(sparse_affinity, eps=args.eps, min_samples=args.min_samples)
        elif args.method == 'hdbscan':
            labels = hdbscan(feat, min_samples=args.min_samples)
        elif args.method == 'fast_hierarchy':
            labels = fast_hierarchy(feat, distance=args.eps, hmethod=args.hmethod)
        elif args.method == 'hierarchy':
            labels = hierarchy(feat, n_clusters=args.ncluster, knn=args.knn)
        elif args.method == 'mini_batch_kmeans':
            labels = MiniBatchKMeans(feat.astype(np.float64), n_clusters=args.ncluster, batch_size=args.batch_size)
        elif args.method == "approx_rank_order":
            from approx_rank_order_cluster import build_index, calculate_symmetric_dist, perform_clustering
            app_nearest_neighbors, dists = build_index(feat, n_neighbors=args.knn)
            distance_matrix = calculate_symmetric_dist(app_nearest_neighbors)
            labels = perform_clustering(feat, n_neighbors=args.knn, th=args.aro_th)
        elif args.method == "kmeans":
            labels = KMeans(feat.astype(np.float64), n_clusters=args.ncluster)
        elif args.method == "spectral":
            labels = spectral(feat, n_clusters=args.ncluster)

        with open(ofn, 'w') as f:
            f.writelines(["{}\n".format(l) for l in labels])
        print("#cluster: {}".format(len(np.unique(labels))))
        print("Save as: {}".format(ofn))


    if args.evaluate:
        if not os.path.isfile("data/unlabeled/{}/meta.txt".format(args.data)):
            raise Exception("Meta file not exist, please remove argument \"evaluate\" or create meta file: {}".format("data/unlabeled/{}/meta.txt".format(args.data)))
        with open("data/unlabeled/{}/meta.txt".format(args.data), 'r') as f:
            meta = f.readlines()
        meta = np.array([int(l.strip()) for l in meta])
        print('prec / recall / fscore: {:.4g}, {:.4g}, {:.4g}'.format(*eval_cluster.fscore(meta, labels)))

    print("time: {:.2f} s".format(time.time() - start))
