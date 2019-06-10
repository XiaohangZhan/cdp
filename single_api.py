import numpy as np

from source import cdp, graph, knn

class CDP(object):
    def __init__(self, k, th, metric='cosinesimil', max_sz=1000, step=0.05, max_iter=100, debug_info=False):
        '''
        k: k in KNN searching.
        th: threshold, (0, 1)
        metric: choose one from ['cosinesimil']
        max_sz: maximal size of a cluster
        step: the step to increase the threshold
        max_iter: maximal iteration in propagation
        debug_info: switch on debug mode when more detailed informations will be printed
        '''
        self.k = k
        self.th = th
        self.metric = metric
        self.max_sz = max_sz
        self.step = step
        self.max_iter = max_iter
        self.debug_info = debug_info
        assert metric in ['cosinesimil', 'l1', 'l2', 'linf', 'angulardist', 'bit_hamming']

    def fit(self, X):
        assert len(X.shape) == 2, "X should be in two dims"
        num = X.shape[0]
        # pair selection
        neighbours = knn.knn_nmslib(X, self.k, space=self.metric)
        length = np.array([len(n[0]) for n in neighbours])
        tofill = np.where(length < self.k)[0]
        for idx in tofill:
            neighbours[idx] = [knn.fill_array(neighbours[idx][0], -1, self.k), knn.fill_array(neighbours[idx][1], -1., self.k)]
        indices = np.array([n[0] for n in neighbours])
        distances = np.array([n[1] for n in neighbours])

        distances = (distances - distances.min()) / (distances.max() - distances.min()) # normalized to 0-1

        pairs, scores = cdp.sample((indices, distances), [], th=self.th)

        # propagation
        if self.debug_info:
            print("\nPropagation ...")
        components = graph.graph_propagation(pairs, scores, self.max_sz, self.step, self.max_iter)

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
