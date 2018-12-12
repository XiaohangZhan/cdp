import numpy as np
import os
import json
import time
from .utils import log

import pdb

def get_relationship_feat(committee, pairs):
    start = time.time()
    votefeat = []
    for i,cmt in enumerate(committee):
        log("\t\tprocessing: {}/{}".format(i, len(committee)))
        knn = cmt[0]
        k = knn.shape[1]
        find0 = (knn[pairs[:,0], :] == np.tile(pairs[:,1:], (1, k))).any(axis=1)
        find1 = (knn[pairs[:,1], :] == np.tile(pairs[:,:1], (1, k))).any(axis=1)
        votefeat.append(np.array([1.0 if p[1] in knn[p[0]] or p[0] in knn[p[1]] else 0.0 for p in pairs]).astype(np.float32))
    log('\t\trelationship feature done. time: {}'.format(time.time() - start))
    return np.array(votefeat).T

def cosine_similarity(feat1, feat2):
    assert feat1.shape == feat2.shape
    feat1 /= np.linalg.norm(feat1, axis=1).reshape(-1,1)
    feat2 /= np.linalg.norm(feat2, axis=1).reshape(-1,1)
    return np.array([np.dot(feat1[i,:], feat2[i,:]) for i in range(feat1.shape[0])])[:,np.newaxis]
    
def get_affinity_feat(features, pairs):
    start = time.time()
    cosine_simi = []
    for i,feat in enumerate(features):
        log("\t\tprocessing: {}/{}".format(i, len(features)))
        cosine_simi.append(cosine_similarity(feat[pairs[:,0],:], feat[pairs[:,1],:]))
    log('\t\taffinity feature done. time: {}'.format(time.time() - start))
    return np.concatenate(cosine_simi, axis=1)

def get_structure_feat(members, pairs):
    start = time.time()
    distr_commnb = []
    for i,m in enumerate(members):
        log("\t\tprocessing: {}/{}".format(i, len(members)))
        comm_neighbor = np.array([len(np.intersect1d(m[p[0]][0], m[p[1]][0], assume_unique=True)) for p in pairs]).astype(np.float32)[:,np.newaxis]
        distr_commnb.append(comm_neighbor)
    log('\t\tstructure feature done. time: {}'.format(time.time() - start))
    return np.hstack(distr_commnb)

def create_pairs(base):
    pairs = []
    knn = base[0]
    k = knn.shape[1]
    anchor = np.tile(np.arange(len(knn)).reshape(len(knn), 1), (1, knn.shape[1]))
    selidx = np.where((knn != -1) & (knn != anchor))
    pairs = np.hstack((anchor[selidx].reshape(-1, 1), knn[selidx].reshape(-1, 1)))
#    for i in range(len(base)):
#        knn = np.array(base[i][0])
#        kkk = len(knn)
#        anchor = i * np.ones((kkk,1), dtype=np.int)
#        ps = np.sort(np.hstack((anchor, knn[:,np.newaxis])), axis=1)
#        pairs.append(ps)
#    pairs = np.vstack(pairs)
    # remove single point
#    keepidx = np.where(pairs[:,0] != pairs[:,1])[0]
#    pairs = pairs[keepidx,:]
    pairs = np.unique(pairs, axis=0)
    return pairs

def get_label(id_label, pairs):
    return (id_label[pairs[:,0]] == id_label[pairs[:,1]]).astype(np.float32)[:,np.newaxis]

def create(data_name, args, phase='test'):
    if phase == 'test':
        output = "{}/output/pairset/k{}".format(args.exp_root, args.k)
    else:
        output = "data/{}/pairset/k{}".format(data_name, args.k)
    members = [args.base] + args.committee

    # loading
    if 'affinity' in args.mediator['input'] and not os.path.isfile(output + "/affinity.npy"):
        log("\tLoading features")
        features = []
        for m in members:
            features.append(np.fromfile('data/{}/features/{}.bin'.format(data_name, m), dtype=np.float32).reshape(-1, args.feat_dim))

    if not os.path.isfile(output + "/pairs.npy") or not os.path.isfile(output + "/structure.npy"):
        log("\tLoading base KNN")
        with open('data/{}/knn/{}_k{}.json'.format(data_name, args.base, args.k), 'r') as f:
            knn_base = json.load(f)
    
        if 'relationship' in args.mediator['input'] or 'structure' in args.mediator['input']:
            log("\tLoading committee KNN")
            knn_committee = []
            committee_knn_fn = ['data/{}/knn/{}_k{}.json'.format(data_name, cmt, args.k) for cmt in args.committee]
            for cfn in committee_knn_fn:
                with open(cfn, 'r') as f:
                    knn_committee.append(json.load(f))

    if not os.path.isdir(output):
        os.makedirs(output)

    # get pairs
    if os.path.isfile(output + "/pairs.npy"):
        log('\tLoading pairs')
        pairs = np.load(output + "/pairs.npy")
    else:
        log('\tgetting pairs')
        pairs = create_pairs(knn_base)
        np.save(output + "/pairs.npy", pairs)
    log('\tgot {} pairs'.format(len(pairs)))

    # get features
    if 'relationship' in args.mediator['input']:
        if not os.path.isfile(output + "/relationship.npy"):
            log('\tgetting relationship features')
            relationship_feat = get_relationship_feat(knn_committee, pairs)
            np.save(output + "/relationship.npy", relationship_feat)
        else:
            log("\trelationship features exist")

    if 'affinity' in args.mediator['input']:
        if not os.path.isfile(output + "/affinity.npy"):
            log('\tgetting affinity features')
            affinity_feat = get_affinity_feat(features, pairs)
            np.save(output + "/affinity.npy", affinity_feat)
        else:
            log("\taffinity features exist")

    if 'structure' in args.mediator['input']:
        if not os.path.isfile(output + "/structure.npy"):
            log('\tgetting structure features')
            structure_feat = get_structure_feat([knn_base] + knn_committee, pairs)
            np.save(output + "/structure.npy", structure_feat)
        else:
            log("\tstructure features exist")

    # get labels when training
    if phase == 'train' or args.evaluation:
        if not os.path.isfile(output + "/pair_label.npy"):
            if not os.path.isfile("data/{}/meta.txt".format(data_name)):
                raise Exception("Meta file not exist: {}, please create meta.txt or set evaluation to False".format("data/{}/meta.txt".format(data_name)))
            with open("data/{}/meta.txt".format(data_name), 'r') as f:
                lines = f.readlines()
            log('\tgetting pairs label')
            id_label = np.array([int(l.strip()) for l in lines])
            label = get_label(id_label, pairs)
            np.save(output + "/pair_label.npy", label)
        else:
            log("\tpairs label exist")
