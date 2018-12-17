import numpy as np
import os
import pdb
import sys
import time
import json
from . import graph
import pickle
from . import eval_cluster
from .create_pair_set import create
from .mediator import Mediator
from .utils import log

def sample(base, committee, accept=7, th=0.7):
    pairs = []
    scores = []
    if len(committee) > 0:
        knn = base[0]
        k = knn.shape[1]
        tile_knn = np.tile(knn.reshape(len(knn), -1, 1), (1, 1, k))
        simi = 1.0 - base[1]
        anchor = np.tile(np.arange(len(knn)).reshape(len(knn), 1), (1, knn.shape[1]))
        vote_num = np.zeros(knn.shape, dtype=np.int)
        for cmt in committee:
            tile_cmt = np.tile(cmt[0].reshape(len(cmt[0]), 1, -1), (1, k, 1))
            vote_num += (tile_knn == tile_cmt).sum(axis=2)
        selidx = np.where((simi > th) & (vote_num >= accept) & (knn != -1) & (knn != anchor))

    else:
        knn = base[0]
        simi = 1.0 - base[1]
        anchor = np.tile(np.arange(len(knn)).reshape(len(knn), 1), (1, knn.shape[1]))
        selidx = np.where((simi > th) & (knn != -1) & (knn != anchor))

    pairs = np.hstack((anchor[selidx].reshape(-1, 1), knn[selidx].reshape(-1, 1)))
    scores = simi[selidx]

    pairs = np.sort(pairs, axis=1)
    pairs, unique_idx = np.unique(pairs, return_index=True, axis=0)
    scores = scores[unique_idx]
    return pairs, scores

def cdp(args):
    exp_root = os.path.dirname(args.config)
    setattr(args, 'exp_root', exp_root)

    with open("data/{}/list.txt".format(args.data_name), 'r') as f:
        fns = f.readlines()
    args.total_num = len(fns)

    if args.strategy == "vote":
        output_cdp = '{}/output/k{}_{}_accept{}_th{}'.format(exp_root, args.k, args.strategy, args.vote['accept_num'], args.vote['threshold'])
    elif args.strategy == "mediator":
        output_cdp = '{}/output/k{}_{}_{}{}{}_th{}'.format(
            exp_root, 
            args.k, 
            args.strategy, 
            int('relationship' in args.mediator['input']),
            int('affinity' in args.mediator['input']),
            int('structure' in args.mediator['input']),
            args.mediator['threshold'])
    elif args.strategy == 'groundtruth':
        output_cdp = '{}/output/gt'.format(exp_root)
    else:
        raise Exception('No such strategy: {}'.format(args.strategy))

    output_sub = '{}/sz{}_step{}'.format(output_cdp, args.propagation['max_sz'], args.propagation['step'])
    log('Output folder: {}'.format(output_sub))
    outmeta = '{}/meta.txt'.format(output_sub)
    if not os.path.isdir(output_sub):
        os.makedirs(output_sub)

    # pair selection
    if args.strategy == 'vote':
        pairs, scores = vote(output_cdp, args)
    elif args.strategy == 'mediator':
        pairs, scores = mediator(args)
    else:
        pairs, scores = groundtruth(args)
    log("\tpair num: {}".format(len(pairs)))

    # propagation
    log("Propagation ...")
    components = graph.graph_propagation(pairs, scores, args.propagation['max_sz'], args.propagation['step'])

    # collect results
    cdp_res = []
    for c in components:
        cdp_res.append(sorted([n.name for n in c]))
    pred = -1 * np.ones(args.total_num, dtype=np.int)
    for i,c in enumerate(cdp_res):
        pred[np.array(c)] = i

    valid = np.where(pred != -1)
    _, unique_idx = np.unique(pred[valid], return_index=True)
    pred_unique = pred[valid][np.sort(unique_idx)]
    pred_mapping = dict(zip(list(pred_unique), range(pred_unique.shape[0])))
    pred_mapping[-1] = -1
    pred = np.array([pred_mapping[p] for p in pred])

    # analyse results
    num_valid = len(valid[0])
    num_class = len(pred_unique)
    log("\n------------- Analysis --------------")
    log('num_images: {}\tnum_class: {}\tnum_per_class: {:.2g}'.format(num_valid, num_class, num_valid/float(num_class)))
    log("Discard ratio: {:.4g}".format(1 - num_valid / float(len(pred))))

    # evaluate
    if args.evaluation:
        log("\n------------- Evaluation --------------")
        if not os.path.isfile("data/{}/meta.txt".format(args.data_name)):
            raise Exception("Meta file not exist: {}, please create meta.txt or set evaluation to False".format("data/{}/meta.txt".format(args.data_name)))
        with open("data/{}/meta.txt".format(args.data_name), 'r') as f:
            label = f.readlines()
            label = np.array([int(l.strip()) for l in label])

        assert len(label) == len(pred), "numbers of labels and predictions are different: {} vs {}".format(len(label), len(pred))

        # pair evaluation
        log("Pair accuracy: {:.4g}".format((label[pairs[:,0]] == label[pairs[:,1]]).sum() / float(len(pairs))))

        # clustering evaluation
        pred_with_singular = pred.copy()
        pred_with_singular[np.where(pred == -1)] = np.arange(num_class, num_class + (pred == -1).sum()) # to assign -1 with new labels
        log('(singular removed) prec / recall / fscore: {:.4g}, {:.4g}, {:.4g}'.format(*eval_cluster.fscore(label[valid], pred[valid])))
        log('(singular kept) prec / recall / fscore: {:.4g}, {:.4g}, {:.4g}'.format(*eval_cluster.fscore(label, pred_with_singular)))

    # write to list
    new_label = ['{}\n'.format(p) for p in pred]
    if not os.path.isdir(os.path.dirname(outmeta)):
        os.makedirs(os.path.dirname(outmeta))
    log('Writing to: {}'.format(outmeta))
    with open(outmeta, 'w') as f:
        f.writelines(new_label)

def vote(output, args):
    assert args.vote['accept_num'] <= len(args.committee)
    base_knn_fn = 'data/{}/knn/{}_k{}.npz'.format(args.data_name, args.base, args.k)
    committee_knn_fn = ['data/{}/knn/{}_k{}.npz'.format(args.data_name, cmt, args.k) for cmt in args.committee]

    if not os.path.isfile(output + '/vote_pairs.npy'):
        log('Extracting pairs by voting ...')
        knn_file = np.load(base_knn_fn)
        knn_base = (knn_file['idx'], knn_file['dist'])

        knn_committee = []
        for i,cfn in enumerate(committee_knn_fn):
            knn_file = np.load(cfn)
            knn_committee.append((knn_file['idx'], knn_file['dist']))

        pairs, scores = sample(knn_base, knn_committee, accept=args.vote['accept_num'], th=args.vote['threshold'])
        np.save(output + '/vote_pairs.npy', pairs)
        np.save(output + '/vote_scores.npy', scores)
    else:
        log('Loading pairs by voting ...')
        pairs = np.load(output + '/vote_pairs.npy')
        scores = np.load(output + '/vote_scores.npy')
    return pairs, scores

def mediator(args):

    args.mediator['model_name'] = "data/{}/models/k{}_{}{}{}.pth.tar".format(
        args.mediator['train_data_name'],
        args.k,
        int('relationship' in args.mediator['input']),
        int('affinity' in args.mediator['input']),
        int('structure' in args.mediator['input']),
    )
    med = Mediator(args)
    if not os.path.isfile(args.mediator['model_name']) or args.mediator['force_retrain']:
        log("Creating pair set for: labeled")
        create(args.mediator['train_data_name'], args, phase="train")
        log("Training")
        med.train()
    else:
        log("Mediator model exists: {}".format(args.mediator['model_name']))
    
    log("Creating pair set for: unlabeled")
    create(args.data_name, args)
    log("Testing")
    med.test() 
    raw_pairs = np.load("{}/output/pairset/k{}/pairs.npy".format(args.exp_root, args.k))
    pair_pred = np.load("{}/output/pairset/k{}/pairs_pred.npy".format(args.exp_root, args.k))
    sel = np.where(pair_pred > args.mediator['threshold'])[0]
    pairs = raw_pairs[sel, :]
    scores = pair_pred[sel]
    return pairs, scores

def groundtruth(args):
    raw_pairs = np.load("{}/output/{}/k{}/pairs.npy".format(args.exp_root, args.data_name, args.k))
    pair_gt = np.load("{}/output/{}/k{}/pair_label.npy".format(args.exp_root, args.data_name, args.k))
    pairs = raw_pairs[np.where(pair_gt == 1)[0], :]
    pairs = pairs[np.where(pairs[:,0] != pairs[:,1])]
    pairs = np.unique(np.sort(pairs, axis=1), axis=0)
    scores = np.ones((pairs.shape[0]), dtype=np.float32)
    return pairs, scores
