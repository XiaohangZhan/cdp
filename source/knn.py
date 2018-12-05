import pickle
import numpy as np
import nmslib
import json
import multiprocessing
import os

def load_feats(fn, feature_dim=256):
    return np.fromfile(fn, dtype=np.float32).reshape(-1, feature_dim)

def dump2json(ofn, data, force=False):
    if os.path.exists(ofn) and not force:
        return

    def default(obj):
        if isinstance(obj, np.int32):
            return int(obj)
        elif isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, set) or isinstance(obj, np.ndarray):
            return list(obj)
        else:
            raise TypeError(
            "Unserializable object {} of type {}".format(obj, type(obj)))

    with open(ofn, 'w') as of:
        json.dump(data, of, default=default)

def knn_nmslib(feats, m, k, output_dir):
    fn = "{}/{}_k{}.json".format(output_dir, m, k)
    if not os.path.isfile(fn):
        print("\nSearch KNN for {}".format(m))
        index = nmslib.init(method='hnsw', space='cosinesimil')
        index.addDataPointBatch(feats)
        index.createIndex({'post': 2}, print_progress=True)
        neighbours = index.knnQueryBatch(feats, k=k, num_threads=multiprocessing.cpu_count())
        dump2json(fn, neighbours)
        #with open(fn, 'wb') as f:
        #    pickle.dump(neighbours, f)
        print("\n")
    else:
        print("KNN file already exists: {}".format(fn))

def get_hist(topk):
    hist = {}
    for t in topk:
        for i in t:
            if i not in hist.keys():
                hist[i] = 1
            else:
                hist[i] += 1
    return hist

def fill_array(array, fill, length):
    assert length >= array.shape[0], "Cannot fill"
    if length == array.shape[0]:
        return array
    array2 = fill * np.ones((length), dtype=array.dtype)
    array2[:array.shape[0]] = array
    return array2

def create_knn(args):
    members = [args.base] + args.committee
    output_dir = 'data/{}/knn/'.format(args.data_name)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    with open("data/{}/list.txt".format(args.data_name), 'r') as f:
        fns = f.readlines()
    args.total_num = len(fns)

    # check feature files exist
    for m in members:
        if not os.path.isfile('data/{}/features/{}.bin'.format(args.data_name, m)):
            raise Exception('Feature file not exist: data/{}/features/{}.bin'.format(args.data_name, m))

    # create knn files with nmslib
    print("KNN Processing")
    for m in members:
        feats = load_feats('data/{}/features/{}.bin'.format(args.data_name, m), args.feat_dim)
        assert feats.shape[0] == args.total_num, "Feature length of [{}] not consistent with list file, {} vs {}".format(m, feats.shape[0], args.total_num)
        knn_nmslib(feats, m, args.k, output_dir)
