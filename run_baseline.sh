#!/bin/bash
srun -p CPU -w BJ-IDC1-10-10-30-78 python -u tools/baseline_clustering.py \
    --data emore_u200k \
    --feature nas \
    --feat-dim 256 \
    --method mini_batch_kmeans \
    --ncluster 2000 \
    --knn 15 \
    --force
