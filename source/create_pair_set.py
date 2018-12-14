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
        find0 = (knn[pairs[:,0], :] == np.tile(pairs[:,1:], (1, k))).any(axis=1, keepdims=True)
        find1 = (knn[pairs[:,1], :] == np.tile(pairs[:,:1], (1, k))).any(axis=1, keepdims=True)
        votefeat.append((find0 | find1).astype(np.float32))
    log('\t\trelationship feature done. time: {}'.format(time.time() - start))
    return np.hstack(votefeat)

def cosine_similarity(feat1, feat2):
    assert feat1.shape == feat2.shape
    feat1 /= np.linalg.norm(feat1, axis=1).reshape(-1, 1)
    feat2 /= np.linalg.norm(feat2, axis=1).reshape(-1, 1)
    return np.einsum('ij,ij->i', feat1, feat2).reshape(-1, 1).reshape(-1, 1) # row-wise dot
    
def get_affinity_feat(features, pairs):
    start = time.time()
    cosine_simi = []
    for i,feat in enumerate(features):
        log("\t\tprocessing: {}/{}".format(i, len(features)))
        cosine_simi.append(cosine_similarity(feat[pairs[:,0],:], feat[pairs[:,1],:]))
    log('\t\taffinity feature done. time: {}'.format(time.time() - start))
    return np.concatenate(cosine_simi, axis=1)

def intersection(array1, array2):
    '''
    To find row wise intersection size.
    Input: array1, array2: Nxk np array
    '''
    N, k = array1.shape
    tile1 = np.tile(array1.reshape(N, k, 1), (1, 1, k))
    tile2 = np.tile(array2.reshape(N, 1, k), (1, k, 1))
    inter_num = ((tile1 == tile2) & (tile1 != -1) & (tile2 != -1)).sum(axis=(1,2))
    return inter_num

def get_structure_feat(members, pairs):
    start = time.time()
    distr_commnb = []
    for i,m in enumerate(members):
        log("\t\tprocessing: {}/{}".format(i, len(members)))
        knn = m[0]
        #comm_neighbor = np.array([len(np.intersect1d(knn[p[0]], knn[p[1]], assume_unique=True)) for p in pairs]).astype(np.float32)[:,np.newaxis]
        comm_neighbor = intersection(knn[pairs[:,0], :], knn[pairs[:,1], :])[:, np.newaxis]
        distr_commnb.append(comm_neighbor)
    log('\t\tstructure feature done. time: {}'.format(time.time() - start))
    return np.hstack(distr_commnb)

def create_pairs(base):
    pairs = []
    knn = base[0]
    anchor = np.tile(np.arange(len(knn)).reshape(len(knn), 1), (1, knn.shape[1]))
    selidx = np.where((knn != -1) & (knn != anchor))
    pairs = np.hstack((anchor[selidx].reshape(-1, 1), knn[selidx].reshape(-1, 1)))
    pairs = np.sort(pairs, axis=1)
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
        knn_file = np.load('data/{}/knn/{}_k{}.npz'.format(data_name, args.base, args.k))
        knn_base = (knn_file['idx'], knn_file['dist'])
    
        if 'relationship' in args.mediator['input'] or 'structure' in args.mediator['input']:
            log("\tLoading committee KNN")
            knn_committee = []
            committee_knn_fn = ['data/{}/knn/{}_k{}.npz'.format(data_name, cmt, args.k) for cmt in args.committee]
            for cfn in committee_knn_fn:
                knn_file = np.load(cfn)
                knn_committee.append((knn_file['idx'], knn_file['dist']))

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
