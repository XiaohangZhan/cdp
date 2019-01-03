#!/bin/bash
# --method in ['kmeans', 'mini_batch_kmeans', 'spectral', 'hierarchy', 'fast_hierarchy', 'dbscan', 'hdbscan', 'knn_dbscan', 'approx_rank_order']
python -u tools/baseline_clustering.py \
    --data emore_u200k \
    --feature nas \
    --feat-dim 256 \
    --method kmeans \
    --ncluster 2577 \
    --knn 80 \
    --aro-th 100 \
    --batch-size 200 \
    --evaluate
