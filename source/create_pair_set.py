import numpy as np
import os
import pickle
import time
import sys
import pdb

def get_vote_feat(committee, pairs):
    start = time.time()
    votefeat = []
    for i in range(committee.shape[1]):
        votefeat.append(np.array([1.0 if p[1] in committee[p[0], i, :] else 0.0 for p in pairs]).astype(np.float32))
    print('vote feat done. time: {}'.format(time.time() - start))
    return np.array(votefeat).T

def get_base_feat(feature, pairs):
    return np.hstack((feature[pairs[:,0]], feature[pairs[:,1]]))

def cosine_similarity(feat1, feat2):
    assert feat1.shape == feat2.shape
    feat1 /= np.linalg.norm(feat1, axis=1).reshape(-1,1)
    feat2 /= np.linalg.norm(feat2, axis=1).reshape(-1,1)
    return np.array([np.dot(feat1[i,:], feat2[i,:]) for i in range(feat1.shape[0])])[:,np.newaxis]
    
def get_dist_feat(features, pairs):
    start = time.time()
    cosine_simi = []
    for i,feat in enumerate(features):
        print("getting dist feat #{}".format(i))
        cosine_simi.append(cosine_similarity(feat[pairs[:,0],:], feat[pairs[:,1],:]))
    print('dist feat done. time: {}'.format(time.time() - start))
    return np.concatenate(cosine_simi, axis=1)

def get_knn_feat(all_dist, pairs, topk):
    start = time.time()
    cosine_simi = []
    for i in range(topk.shape[1]):
        for j in range(topk.shape[2]):
            topk_simi = np.array([1.0 - all_dist[i][tuple(sorted([p[0], topk[p[0], i, j]]))] for p in pairs])[:, np.newaxis]
            cosine_simi.append(topk_simi)
    for i in range(topk.shape[1]):
        for j in range(topk.shape[2]):
            topk_simi = np.array([1.0 - all_dist[i][tuple(sorted([p[1], topk[p[1], i, j]]))] for p in pairs])[:, np.newaxis]
            cosine_simi.append(topk_simi)
    print('knn feat done. time: {}'.format(time.time() - start))
    return np.concatenate(cosine_simi, axis=1)

def create_pair(knn):
    num = knn.shape[0]
    kkk = knn.shape[1]
    anchor = np.repeat(np.arange(num), kkk)[:,np.newaxis]
    pairs = np.hstack((anchor, knn.flatten()[:,np.newaxis]))
    pairs = pairs[np.where(pairs[:,0] != pairs[:,1])[0],:]
    pairs = np.unique(np.sort(pairs, axis=1), axis=0)
    return pairs

def get_label(id_label, pairs):
    return (id_label[pairs[:,0]] == id_label[pairs[:,1]]).astype(np.float32)[:,np.newaxis]

def main():
    model_name = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'densenet121', 'vgg16bn', 'inceptionv3', 'ir', 'nas']
    #model_name = ['nas', 'nas', 'nas', 'nas', 'nas', 'nas', 'nas', 'nas', 'nas']
    #model_name = ['nas', 'resnet50', 'resnet50', 'resnet50', 'resnet50', 'resnet50', 'resnet50', 'resnet50', 'resnet50']
    #exps = ['base-resnet18-lr0.5', 'base-resnet34-lr0.5', 'base-resnet50-lr0.3', 'base-densenet121-lr0.1', 'base-vgg16_bn-lr0.1', 'base-irv2-lr0.1', 'base-resnet101-lr0.1', 'base-nasnetasmall-lr0.1', 'base-inception_v3-lr0.3']
    #exps = ['base-nasnetasmall-lr0.1', 'base-nasnetasmall-lr0.1-rand1', 'base-nasnetasmall-lr0.1-rand2', 'base-nasnetasmall-lr0.1-rand3', 'base-nasnetasmall-lr0.1-rand4', 'base-nasnetasmall-lr0.1-rand5', 'base-nasnetasmall-lr0.1-rand6', 'base-nasnetasmall-lr0.1-rand7', 'base-nasnetasmall-lr0.1-rand7']
    #exps = ['selector2-resnet18', 'selector2-resnet34', 'selector2-resnet50', 'selector2-densenet121', 'selector2-vgg16_bn', 'selector2-irv2', 'selector2-resnet101', 'selector2-nasnetasmall', 'selector2-inception_v3']
    #exps = ['resnet18_arc_lr0.1', 'resnet34_arc_lr0.1', 'resnet50_arc_lr0.06', 'resnet101_arc_lr0.05', 'densenet121_arc_lr0.05', 'vgg16bn_arc_lr0.05', 'inceptionv3_arc_lr0.06', 'ir_arc_lr0.05', 'nas_arc_lr0.05']
    exps = ['resnet18_arc', 'resnet34_arc', 'resnet50_arc', 'resnet101_arc', 'densenet121_arc','vgg16bn_arc', 'inceptionv3_arc', 'ir_arc', 'nas_arc']
    #label_fn = '/mnt/lustre/zhanxiaohang/data/MsCeleb/split/base_0.2.txt'
    #label_fn = '/mnt/lustre/zhanxiaohang/data/MsCeleb/split/extra_0.8.txt'
    dataid = 'actor100w+80w0'
    #label_fn = '/mnt/lustre/zhanxiaohang/data/actor/split/labellist_12'
    label_fn = '/mnt/lustre/zhanxiaohang/data/actor100w+80w/split/meta_100w+80w.txt_0'
    feat_root = '/mnt/lustre/zhanxiaohang/proj/multitask_yl/multitask/checkpoints/actor100w+80w/base_bn'
    baseidx = 4
    kkk = 20
    feat_dim = 512
    committee = 'committee'
    output_root = 'output.actor100w+80w'
    votef = True
    distf = True
    knnf = False
    #feature_type = 'vote'
    #feature_type = 'dist'
    feature_type = 'vote+dist'
    #feature_type = 'vote+dist+knnoffline'

    output = '{}/cluster/{}/pairs/feat-top{}-{}-{}-{}.npy'.format(output_root, committee, kkk, dataid, model_name[baseidx], feature_type)
    pair_fn = '{}/cluster/{}/pairs/pair-top{}-{}-{}.npy'.format(output_root, committee, kkk, dataid, model_name[baseidx])

    if not os.path.isfile(label_fn):
        raise Exception('No such file: {}'.format(label_fn))
    # loading
    topk = np.load('{}/knn/{}/top{}_{}_nmslib/all_topk.npy'.format(output_root, committee, kkk, dataid))
    if distf:
        features = []
        for exp in exps:
            features.append(np.fromfile('{}/{}/{}_best.bin'.format(feat_root, exp, dataid), dtype=np.float32).reshape(-1, feat_dim))
    if knnf:
        with open('{}/knn/{}/top{}_{}_nmslib/all_dist.pkl'.format(output_root, committee, kkk, dataid), 'rb') as f:
            all_dist = pickle.load(f)

    if not os.path.isdir(os.path.dirname(output)):
        os.makedirs(os.path.dirname(output))
    base = topk[:,baseidx,:]
    committee = topk[:,np.delete(np.arange(9), baseidx),:]

    # get pairs
    if os.path.isfile(pair_fn):
        print('loading pairs')
        pairs = np.load(pair_fn)
    else:
        print('creating pairs')
        pairs = create_pair(base)
        np.save(pair_fn, pairs)

    # create similarity dict
    # get features
    if votef:
        print('get vote feat')
        votefeat = get_vote_feat(committee, pairs)
    if distf:
        print('get dist feat')
        distfeat = get_dist_feat(features, pairs)
    if knnf:
        print('get knn feat')
        knnfeat = get_knn_feat(all_dist, pairs, topk)
    with open(label_fn, 'r') as f:
        lines = f.readlines()
    print('get label')
    id_label = np.array([int(l.strip().split()[-1]) for l in lines])
    label = get_label(id_label, pairs)

    # save
    print('hstack')
    all_feat = []
    if distf:
        all_feat.append(distfeat)
    if votef:
        all_feat.append(votefeat)
    if knnf:
        all_feat.append(knnfeat)
    feat_label = np.hstack(tuple(all_feat + [label]))
    #pdb.set_trace()
    print('save')
    np.save(output, feat_label)
    print('done')

if __name__ == '__main__':
    main()
