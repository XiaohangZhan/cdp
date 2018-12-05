#!/bin/bash
srun -p CPU -w BJ-IDC1-10-10-30-79 python -u main.py --config $1
