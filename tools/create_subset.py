import numpy as np
from glob import glob
import pdb

num = 600000
source = "data/unlabeled/emore_u1.4m/"
target = "data/unlabeled/emore_u600k/"
fns = glob("{}/features/*.bin".format(source))

feat_dim = 256

with open("{}/list.txt".format(source), 'r') as f:
    lines = f.readlines()
with open("{}/list.txt".format(target), 'w') as f:
    f.writelines(lines[:num])
with open("{}/meta.txt".format(source), 'r') as f:
    lines = f.readlines()
with open("{}/meta.txt".format(target), 'w') as f:
    f.writelines(lines[:num])

for fn in fns:
    feat = np.fromfile(fn, dtype=np.float32).reshape(-1, feat_dim)
    feat[:num, :].tofile(fn.replace(source, target))
