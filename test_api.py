import os
import numpy as np

from single_api import CDP
from source import eval_cluster

# params
metric = 'cosinesimil' # supported: 'cosinesimil', 'l1', 'l2', 'linf', 'angulardist', 'bit_hamming'
th_dict = {'cosinesimil': 0.54, 'l1': 0.2, 'l2': 0.2, 'angulardist': 0.35} # reference threshold for face data
th = th_dict[metric]
K = 15
max_sz = 600
step = 0.05
max_iter = 100

# clustering
X = np.fromfile("data/unlabeled/emore_u200k/features/nas.bin", dtype=np.float32).reshape(-1, 256)
cdp = CDP(K, th, metric=metric, max_sz=max_sz, step=step, max_iter=max_iter, debug_info=True)
cdp.fit(X)
pred = cdp.labels_

# evaluation
if os.path.isfile("data/unlabeled/emore_u200k/meta.txt"):
    print("\n------------- Evaluation --------------")
    with open("data/unlabeled/emore_u200k/meta.txt", 'r') as f:
        label = f.readlines()
        label = np.array([int(l.strip()) for l in label])
    assert len(label) == len(pred)
    
    valid = np.where(pred != -1)
    _, unique_idx = np.unique(pred[valid], return_index=True)
    pred_unique = pred[valid][np.sort(unique_idx)]
    num_class = len(pred_unique) 
    pred_with_singular = pred.copy()
    pred_with_singular[np.where(pred == -1)] = np.arange(num_class, num_class + (pred == -1).sum())
    print('(singular removed) prec / recall / fscore: {:.4g}, {:.4g}, {:.4g}'.format(*eval_cluster.fscore(label[valid], pred[valid])))
    print('(singular kept) prec / recall / fscore: {:.4g}, {:.4g}, {:.4g}'.format(*eval_cluster.fscore(label, pred_with_singular)))
