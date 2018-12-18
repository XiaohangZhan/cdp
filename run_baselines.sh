#!/bin/bash
python -u tools/baseline_clustering.py \
    --data emore_u200k \
    --feature nas \
    --feat-dim 256 \
    --method kmeans \
    --ncluster 2577 \
    --knn 15 \
    --aro-th 100 \
    --batch-size 200 \
    --evaluate
