import numpy as np
import os
import pdb
import sys
import time
from . import graph
import pickle
from . import eval_cluster

def get_hist(cmt):
    cmt = [idx for c in cmt for idx in c]
    hist = {}
    for i in cmt:
        if i == -1:
            continue
        if i in hist.keys():
            hist[i] += 1
        else:
            hist[i] = 1
    return hist

def sample_task(i, base, committee, vote_num=7, th=0.7):
    pairs = []
    scores = []
    hist = get_hist([c[i][0] for c in committee])
    knn = base[i][0]
    simi = 1.0 - base[i][1]
    for j, k in enumerate(knn):
        if k != -1 and k in hist.keys() and hist[k] >= vote_num and k != i and simi[j] > th:
            pairs.append(sorted([i, k]))
            scores.append(simi[j])
    return pairs, scores
    
def helper(args):
    sample_task(*args)

def sample_parallel(base, committee, vote_num=7, th=0.7):
    import multiprocessing
    import tqdm
    pool = multiprocessing.Pool(16)
    if len(committee) > 0:
        res = list(tqdm.tqdm(pool.imap(helper, zip(range(len(base)), [base]*len(base), [committee]*len(base), [vote_num]*len(base), [th]*len(base))), total=len(base)))
    pairs, scores = zip(*res)
    pairs = np.array(pairs)
    scores = np.array(scores)
    pairs, unique_idx = np.unique(pairs, return_index=True, axis=0)
    scores = scores[unique_idx]
    return pairs, scores

def sample(base, committee, vote_num=7, th=0.7):
    pairs = []
    scores = []
    if len(committee) > 0:
        for i in range(len(base)):
            hist = get_hist([c[i][0] for c in committee])
            knn = base[i][0]
            simi = 1.0 - base[i][1]
            for j, k in enumerate(knn):
                if k != -1 and k in hist.keys() and hist[k] >= vote_num and k != i and simi[j] > th:
                    pairs.append(sorted([i, k]))
                    scores.append(simi[j])
    else:
        for i in range(len(base)):
            knn = base[i][0]
            simi = 1.0 - base[i][1]
            for j,k in enumerate(knn):
                if k != -1 and k != i and simi[j] > th:
                    pairs.append(sorted([i, k]))
                    scores.append(simi[j])
    pairs = np.array(pairs)
    scores = np.array(scores)
    pairs, unique_idx = np.unique(pairs, return_index=True, axis=0)
    scores = scores[unique_idx]
    return pairs, scores

def cdp(args):
    exp_root = os.path.dirname(args.config)

    with open("data/{}/list.txt".format(args.data_name), 'r') as f:
        fns = f.readlines()
    args.total_num = len(fns)

    if args.strategy == "vote":
        output_cdp = '{}/output/{}_accept{}_th{}'.format(exp_root, args.strategy, args.vote['accept_num'], args.vote['threshold'])
    elif args.strategy == "mediator":
        output_cdp = '{}/output/{}_th{}'.format(exp_root, args.strategy, args.mediator['threshold'])
    else:
        raise Exception('No such strategy: {}'.format(args.strategy))

    output_sub = '{}/sz{}_step{}'.format(output_cdp, args.propagation['max_sz'], args.propagation['step'])
    print('Output folder: {}'.format(output_sub))
    outcdp = output_sub + '/cdp.pkl'
    outpred = output_sub + '/pred.npy'
    outlist = '{}/list.txt'.format(output_sub)
    outmeta = '{}/meta.txt'.format(output_sub)
    if not os.path.isdir(output_sub):
        os.makedirs(output_sub)

    # pair selection
    if args.strategy == 'vote':
        pairs, scores = vote(output_cdp, args)
    else:
        pairs, scores == mediator(args)
    print("pair num: {}".format(len(pairs)))

    # propagation
    if not os.path.isfile(outcdp):
        print("Propagation ...")
        comps = graph.graph_propagation(pairs, scores, args.propagation['max_sz'], args.propagation['step'])
        cdp_res = []
        for c in comps:
            cdp_res.append(sorted([n.name for n in c]))
        pred = -1 * np.ones(args.total_num, dtype=np.int)
        for i,c in enumerate(cdp_res):
            pred[np.array(c)] = i
        # save
        with open(outcdp, 'wb') as f:
            pickle.dump(cdp_res, f)
        np.save(outpred, pred)
    else:
        print('Loading CDP results ...')
        with open(outcdp, 'rb') as f:
            cdp_res = pickle.load(f)
        pred = np.load(outpred)

    # analyse results
    print("\n------------- Analysis --------------")
    nodes = np.unique(np.array([nn for cc in cdp_res for nn in cc]))
    print('num_nodes: {}\tnum_class: {}\tnum_per_class: {}'.format(nodes.shape[0], len(cdp_res), nodes.shape[0] / float(len(cdp_res))))
    num_per_class = np.array([len(c) for c in cdp_res])
    hist = np.bincount(num_per_class)
    print("image number of the largest class: {}".format(num_per_class.max()))
    print("the number of classes whose contain 1,2,...,20 images: {}".format(hist[:20]))

    valid = np.where(pred != -1)
    print("Discard ratio: {}".format(1 - len(valid[0]) / float(len(pred))))
    pred_unique = np.unique(pred[valid])
    pred_mapping = dict(zip(list(pred_unique), range(pred_unique.shape[0])))

    # evaluate
    print("\n------------- Evaluation --------------")
    if args.evaluation:
        if not os.path.isfile("data/{}/meta.txt".format(args.data_name)):
            raise Exception("Meta file not exist: {}".format("data/{}/meta.txt".format(args.data_name)))
        with open("data/{}/meta.txt".format(args.data_name), 'r') as f:
            label = f.readlines()
            label = np.array([int(l.strip()) for l in label])

        # pair evaluation
        print("Pair accuracy: {}".format((label[pairs[:,0]] == label[pairs[:,1]]).sum() / float(len(pairs))))

        # clustering evaluation
        evaluate_cluster(label[valid], np.array([pred_mapping[pred[v]] for v in valid[0]]))

    # write to list
    new_fns = [fns[v] for v in valid[0]]
    new_label = ['{}\n'.format(pred_mapping[pred[v]]) for v in valid[0]]
    if not os.path.isdir(os.path.dirname(outlist)):
        os.makedirs(os.path.dirname(outlist))
    print('Writing to: {}'.format(outlist))
    print('Writing to: {}'.format(outmeta))
    with open(outlist, 'w') as f:
        f.writelines(new_fns)
    with open(outmeta, 'w') as f:
        f.write('{} {}\n'.format(len(new_label), len(cdp_res)))
        f.writelines(new_label)

def vote(output, args):
    assert args.vote['accept_num'] <= len(args.committee)
    base_knn_fn = 'data/{}/knn/{}.pkl'.format(args.data_name, args.base)
    committee_knn_fn = ['data/{}/knn/{}.pkl'.format(args.data_name, cmt) for cmt in args.committee]
    if not os.path.isfile(output + '/vote_pairs.npy'):
        print('Extracting pairs by voting ...')
        with open(base_knn_fn, 'rb') as f:
            knn_base = pickle.load(f)
        knn_committee = []
        for i,cfn in enumerate(committee_knn_fn):
            with open(cfn, 'rb') as f:
                knn_cmt = pickle.load(f)
                knn_committee.append(knn_cmt)
        pairs, scores = sample(knn_base, knn_committee, vote_num=args.vote['accept_num'], th=args.vote['threshold'])
        np.save(output + '/vote_pairs.npy', pairs)
        np.save(output + '/vote_scores.npy', scores)
    else:
        print('Loading pairs by voting ...')
        pairs = np.load(output + '/vote_pairs.npy')
        scores = np.load(output + '/vote_scores.npy')
    return pairs, scores

def mediator(args):
    raw_pairs = np.load(args.mediator['pair_file'])
    pair_pred = np.load(args.mediator['pair_pred_file'])
    sel = np.where((pair_pred > args.mediator['threshold']) & (raw_pairs[:,0] != raw_pairs[:,1]))[0]
    pairs = raw_pairs[sel, :]
    scores = pair_pred[sel]
    pairs, unique_idx = np.unique(np.sort(pairs, axis=1), return_index=True, axis=0)
    scores = scores[unique_idx]
    return pairs, scores

def groundtruth(args):
    raw_pairs = np.load(args.groundtruth['pair_file'])
    pair_gt = np.load(args.groundtruth['pair_gt_fn']).astype(np.int)
    pairs = raw_pairs[np.where(pair_gt == 1)[0], :]
    pairs = pairs[np.where(pairs[:,0] != pairs[:,1])]
    pairs = np.unique(np.sort(pairs, axis=1), axis=0)
    scores = np.ones((pairs.shape[0]), dtype=np.float32)
    return pairs, scores

def evaluate_cluster(label, pred):
    prec, recall, fmi = eval_cluster.fowlkes_mallows_score(label, pred)
    print('prec: {}, recall: {}, fmi: {}'.format(prec, recall, fmi))
