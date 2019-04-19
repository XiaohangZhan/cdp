import numpy as np
from source import cdp, graph
from sklearn.neighbors import NearestNeighbors

class CDP(object):
    def __init__(self, k, th, metric='minkowski', max_sz=1000, step=0.05, debug_info=False):
        '''
        k: k in KNN searching.
        th: threshold, (0, 1)
        metric: choose one from [
            'cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan',
            'braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice',
            'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'minkowski',
            'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener',
            'sokalsneath', 'sqeuclidean', 'yule']
        max_sz: maximal size of a cluster
        step: the step to increase the threshold
        debug_info: switch on debug mode when more detailed informations will be printed
        '''
        self.k = k
        self.th = th
        self.metric = metric
        self.max_sz = max_sz
        self.step = step
        self.debug_info = debug_info
        assert metric in [
            'cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan',
            'braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice',
            'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'minkowski',
            'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener',
            'sokalsneath', 'sqeuclidean', 'yule']

    def fit(self, X):
        assert len(X.shape) == 2, "X should be in two dims"
        num = X.shape[0]
        # pair selection
        nbrs = NearestNeighbors(n_neighbors=self.k, metric=self.metric, algorithm='ball_tree').fit(X)
        distances, indices = nbrs.kneighbors(X)
        norm_dist = distances / distances.max()

        pairs, scores = cdp.sample((indices, norm_dist), [], th=self.th)

        # propagation
        if self.debug_info:
            print("Propagation ...")
        components = graph.graph_propagation(pairs, scores, self.max_sz, self.step)

        # collect results
        cdp_res = []
        for c in components:
            cdp_res.append(sorted([n.name for n in c]))
        pred = -1 * np.ones(num, dtype=np.int)
        for i,c in enumerate(cdp_res):
            pred[np.array(c)] = i

        valid = np.where(pred != -1)
        _, unique_idx = np.unique(pred[valid], return_index=True)
        pred_unique = pred[valid][np.sort(unique_idx)]
        pred_mapping = dict(zip(list(pred_unique), range(pred_unique.shape[0])))
        pred_mapping[-1] = -1
        pred = np.array([pred_mapping[p] for p in pred])
        self.labels_ = pred

        # analyse results
        if self.debug_info:
            num_valid = len(valid[0])
            num_class = len(pred_unique)
            print("\n------------- Analysis --------------")
            print('num_images: {}\tnum_class: {}\tnum_per_class: {:.2g}'.format(num_valid, num_class, num_valid/float(num_class)))
            print("Discard ratio: {:.4g}".format(1 - num_valid / float(len(pred))))
