import numpy as np

def knn_acc(knn, feat, k):
    feat /= np.linalg.norm(feat, axis=1).reshape(-1, 1)
    featT = feat.T
    recall = 0
    for i in range(feat.shape[0]):
        if i % 1000 == 0 and i != 0:
            print("Processing: {} / {}, rec: {}".format(i, feat.shape[0], recall / float(i)))
        simi = feat[i:i+1, :].dot(featT).squeeze()
        topk = np.argsort(simi)[::-1][:k]
        recall += len(np.intersect1d(knn[i], topk, assume_unique=True))
    recall /= float(feat.shape[0] * k)
    return recall

if __name__ == "__main__":
    knn_fn = "data/unlabeled/emore_u200k/knn/nas_k15.npz"
    knn_file = np.load(knn_fn)
    knn, knn_dist = knn_file['idx'], knn_file['dist']
    feat = np.fromfile("data/unlabeled/emore_u200k/features/nas.bin", dtype=np.float32).reshape(-1, 256)
    k = 15
    print(knn_acc(knn, feat, k))
