# Graph Toolbox.
# By Xiaohang Zhan (zhanxiaohang@sensetime.com)

import pdb
import pickle
import numpy as np
import time

class Data(object):
    def __init__(self, name):
        self.__name = name
        self.__links = set()

    @property
    def name(self):
        return self.__name

    @property
    def links(self):
        return set(self.__links)

    def add_link(self, other, score):
        self.__links.add(other)
        other.__links.add(self)

def connected_components(nodes):
    '''
    conventional connected components searching
    '''
    result = []
    nodes = set(nodes)
    while nodes:
        n = nodes.pop()
        group = {n}
        queue = [n]
        while queue:
            n = queue.pop(0)
            neighbors = n.links
            neighbors.difference_update(group)
            nodes.difference_update(neighbors)
            group.update(neighbors)
            queue.extend(neighbors)
        result.append(group)
    return result

def connected_components_constraint(nodes, max_sz, score_dict=None, th=None):
    '''
    only use edges whose scores are above `th`
    if a component is larger than `max_sz`, all the nodes in this component are added into `remain` and returned for next iteration.
    '''
    result = []
    remain = set()
    nodes = set(nodes)
    while nodes:
        n = nodes.pop()
        group = {n}
        queue = [n]
        valid = True
        while queue:
            n = queue.pop(0)
            if th is not None:
                neighbors = {l for l in n.links if score_dict[tuple(sorted([n.name, l.name]))] >= th}
            else:
                neighbors = n.links
            neighbors.difference_update(group)
            nodes.difference_update(neighbors)
            group.update(neighbors)
            queue.extend(neighbors)
            if len(group) > max_sz or len(remain.intersection(neighbors)) > 0:
                # if this group is larger than `max_sz`, add the nodes into `remain`
                valid = False
                remain.update(group)
                break
        if valid: # if this group is smaller than or equal to `max_sz`, finalize it.
            result.append(group)
    return result, remain

def graph_propagation(edges, score, max_sz, step=0.1):

    edges = np.sort(edges, axis=1)
    th = score.min()

    # construct graph
    score_dict = {} # score lookup table
    for i,e in enumerate(edges):
        score_dict[e[0], e[1]] = score[i]

    nodes = np.sort(np.unique(edges.flatten()))
    mapping = -1 * np.ones((nodes.max()+1), dtype=np.int)
    mapping[nodes] = np.arange(nodes.shape[0])
    link_idx = mapping[edges]
    vertex = [Data(n) for n in nodes]
    for l, s in zip(link_idx, score):
        vertex[l[0]].add_link(vertex[l[1]], s)

    # first iteration
    comps, remain = connected_components_constraint(vertex, max_sz)

    # iteration
    components = comps[:]
    while remain:
        th = th + (1 - th) * step
        comps, remain = connected_components_constraint(remain, max_sz, score_dict, th)
        components.extend(comps)
    return components

def graph_propagation_soft(edges, score, max_sz, step=0.1, **kwargs):

    edges = np.sort(edges, axis=1)
    th = score.min()

    # construct graph
    score_dict = {} # score lookup table
    for i,e in enumerate(edges):
        score_dict[e[0], e[1]] = score[i]

    nodes = np.sort(np.unique(edges.flatten()))
    mapping = -1 * np.ones((nodes.max()+1), dtype=np.int)
    mapping[nodes] = np.arange(nodes.shape[0])
    link_idx = mapping[edges]
    vertex = [Data(n) for n in nodes]
    for l, s in zip(link_idx, score):
        vertex[l[0]].add_link(vertex[l[1]], s)

    # first iteration
    comps, remain = connected_components_constraint(vertex, max_sz)
    first_vertex_idx = np.array([mapping[n.name] for c in comps for n in c])
    fusion_vertex_idx = np.setdiff1d(np.arange(nodes.shape[0]), first_vertex_idx, assume_unique=True)
    # iteration
    components = comps[:]
    while remain:
        th = th + (1 - th) * step
        comps, remain = connected_components_constraint(remain, max_sz, score_dict, th)
        components.extend(comps)
    label_dict = {}
    for i,c in enumerate(components):
        for n in c:
            label_dict[n.name] = i
    print('Propagation ...')
    prop_vertex = [vertex[idx] for idx in fusion_vertex_idx]
    label, label_fusion = diffusion(prop_vertex, label_dict, score_dict, **kwargs)
    return label, label_fusion

def diffusion(vertex, label, score_dict, max_depth=5, weight_decay=0.6, normalize=True):
    class BFSNode():
        def __init__(self, node, depth, value):
            self.node = node
            self.depth = depth
            self.value = value
            
    label_fusion = {}
    for name in label.keys():
        label_fusion[name] = {label[name]: 1.0}
    prog = 0
    prog_step = len(vertex) // 20
    start = time.time()
    for root in vertex:
        if prog % prog_step == 0:
            print("progress: {} / {}, elapsed time: {}".format(prog, len(vertex), time.time() - start))
        prog += 1
        #queue = {[root, 0, 1.0]}
        queue = {BFSNode(root, 0, 1.0)}
        visited = [root.name]
        root_label = label[root.name]
        while queue:
            curr = queue.pop()
            if curr.depth >= max_depth: # pruning
                continue
            neighbors = curr.node.links
            tmp_value = []
            tmp_neighbor = []
            for n in neighbors:
                if n.name not in visited:
                    sub_value = score_dict[tuple(sorted([curr.node.name, n.name]))] * weight_decay * curr.value
                    tmp_value.append(sub_value)
                    tmp_neighbor.append(n)
                    if root_label not in label_fusion[n.name].keys():
                        label_fusion[n.name][root_label] = sub_value
                    else:
                        label_fusion[n.name][root_label] += sub_value
                    visited.append(n.name)
                    #queue.add([n, curr.depth+1, sub_value])
            sortidx = np.argsort(tmp_value)[::-1]
            for si in sortidx:
                queue.add(BFSNode(tmp_neighbor[si], curr.depth+1, tmp_value[si]))
    if normalize:
        for name in label_fusion.keys():
            summ = sum(label_fusion[name].values())
            for k in label_fusion[name].keys():
                label_fusion[name][k] /= summ
    return label, label_fusion
