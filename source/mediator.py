import sys
import os

def train_mediator(args):
    cmd = "srun -p Test -n1 --gres=gpu:8 python -u source/mlp.py \
            --exp-name {} --input-dim {} --trainset-fn {} \
            --batch-size {} --valset-fn"
    raise NotImplemented

def test_mediator(args):
    raise NotImplemented
