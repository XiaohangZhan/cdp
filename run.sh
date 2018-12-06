#!/bin/bash
srun -p $2 --gres=gpu:8 python -u main.py --config $1
