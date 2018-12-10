import numpy as np
import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from .utils import AverageMeter, accuracy, load_state, save_state, log

import pdb

class MediatorNet(nn.Module):
    def __init__(self, input_dim):
        super(MediatorNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim * 3)
        self.fc2 = nn.Linear(input_dim * 3, input_dim * 3)
        self.fc_last = nn.Linear(input_dim * 3, 2)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc_last(x)
        return x

class MediatorDataset(Dataset):
    def __init__(self, featnames, labelname=None):
        self.feat = np.concatenate([np.load(fn).astype(np.float32) for fn in featnames], axis=1)
        if labelname is not None:
            self.label = np.squeeze(np.load(labelname).astype(np.int))
        else:
            self.label = None
    def __len__(self):
        return self.feat.shape[0]
    def __getitem__(self, idx):
        if self.label is None:
            label = 0
        else:
            label = self.label[idx]
        return self.feat[idx,:], label

def infer_dim(args):
    dims = 0
    if 'relationship' in args.mediator['input']:
        dims += len(args.committee)
    if 'affinity' in args.mediator['input']:
        dims += len(args.committee) + 1
    if 'distribution' in args.mediator['input']:
        dims += 4 * (len(args.committee) + 1)
    return dims

def train(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpus
    cfg.ngpu = len(cfg.gpus.split(','))
    assert(cfg.batch_size % cfg.ngpu == 0)
    save_path = cfg.model_name

    model = MediatorNet(cfg.input_dim)
    model = nn.DataParallel(model)
    model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), cfg.base_lr,
                                momentum=cfg.momentum,
                                weight_decay=cfg.weight_decay)

    start_epoch = 0
    step = 0
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg.lr_decay_steps, gamma=cfg.lr_decay_scale, last_epoch=start_epoch-1)
    cudnn.benchmark = True

    # data
    train_dataset = MediatorDataset(cfg.trainset_fn, cfg.trainset_label)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    if cfg.val:
        val_dataset = MediatorDataset(cfg.valset_fn, cfg.valset_label)
        val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    # train
    for epoch in range(start_epoch, cfg.max_epoch):
        lr_scheduler.step()
        losses = AverageMeter(cfg.average_stats)
        top1 = AverageMeter(cfg.average_stats)
        model.train()
        for i, input in enumerate(train_loader):
            feature_var = torch.autograd.Variable(input[0].cuda())
            label = input[1].cuda(async=True)
            label_var = torch.autograd.Variable(label)

            output = model(feature_var)

            loss = criterion(output, label_var)
            prec1 = accuracy(output.data, label, topk=(1,))

            losses.update(loss.mean().data.cpu()[0])
            top1.update(prec1[0].cpu()[0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % cfg.print_freq == 0:
                log('Epoch: [{0}][{1}/{2}][{3}]\t'
                    'Lr: {4:.2g}\t'
                    'Loss {loss.val:.4g} ({loss.avg:.4g})\t'
                    'Prec@1 {top1.val:.3g} ({top1.avg:.3g})'.format(
                    epoch, i, len(train_loader), step,
                    optimizer.param_groups[0]['lr'],
                    loss=losses, top1=top1))

            step += 1

        if cfg.val:
            validate(model, val_loader, val_dataset.label, epoch, cfg.th)
    save_state({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': step},
        save_path, epoch, is_last=(epoch==cfg.max_epoch-1))

def validate(model, val_loader, label, epoch, th):
    model.eval()
    prob = []
    softmax = nn.Softmax(dim=1)
    for i, input in enumerate(val_loader):
        feature_var = torch.autograd.Variable(input[0].cuda())
        output = softmax(model(feature_var))
        prob.append(output.data.cpu().numpy()[:,1])
    prob = np.concatenate(tuple(prob), axis=0)
    prob = prob[:len(label)]
    # evaluate
    pred = (prob > th).astype(np.int)
    acc, abs_recall, rel_recall, false_pos, false_neg = evaluate(pred, label)
    log('Val for epoch #{}. th: {:.4g}, acc: {:.4g}, abs_recall: {:.4g}, rel_recall: {:.4g}, false_pos: {:.4g}, false_neg: {:.4g}'.format(epoch, th, acc, abs_recall, rel_recall, false_pos, false_neg))
    
def evaluate(pred, label):
    num = label.shape[0]
    pos_num = (label == 1).sum()
    acc = (pred == label).sum() / float(num)
    abs_recall = (pred == 1).sum() / float(num) 
    rel_recall = (pred == 1).sum() / float(pos_num)
    false_pos = ((pred == 1) & (label == 0)).sum() / float((pred == 1).sum())
    false_neg = ((pred == 0) & (label == 1)).sum() / float((pred == 0).sum())
    return acc, abs_recall, rel_recall, false_pos, false_neg
    
def test(cfg):
    load_path = cfg.model_name
    assert os.path.isfile(load_path), "Model file not exist: {}".format(load_path)
    if not os.path.isdir(os.path.dirname(cfg.test_output)):
        os.makedirs(os.path.dirname(cfg.test_output))
    # model
    model = MediatorNet(cfg.input_dim)
    model = nn.DataParallel(model)
    model.cuda()
    load_state(load_path, model)
    cudnn.benchmark = True

    # data
    test_dataset = MediatorDataset(cfg.testset_fn, cfg.testset_label)
    total_size = test_dataset.__len__()
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    # extract
    model.eval()
    prob = []
    softmax = nn.Softmax(dim=1)
    for i, input in enumerate(test_loader):
        feature_var = torch.autograd.Variable(input[0].cuda())
        output = softmax(model(feature_var))
        prob.append(output.data.cpu().numpy()[:,1])

    prob = np.concatenate(tuple(prob), axis=0)
    prob = prob[:total_size]
    np.save(cfg.test_output, prob)
    label = test_dataset.label
    # accuracy
    if label is not None:
        pred = (prob > cfg.th).astype(np.int)
        acc, abs_recall, rel_recall, false_pos, false_neg = evaluate(pred, label)
        log('Testing set. acc: {:.4g}, abs_recall: {:.4g}, rel_recall: {:.4g}, false_pos: {:.4g}, false_neg: {:.4g}'.format(acc, abs_recall, rel_recall, false_pos, false_neg))

class Mediator(object):
    def __init__(self, args):
        cfg = ArgObj()
        cfg.exp_root = args.exp_root
        cfg.model_name = args.mediator['model_name']
        cfg.input_dim = infer_dim(args)
        cfg.trainset_fn = ["data/{}/pairset/k{}/{}.npy".format(args.mediator['train_data_name'], args.k, ip) for ip in args.mediator['input']]
        cfg.trainset_label = "data/{}/pairset/k{}/pair_label.npy".format(args.mediator['train_data_name'], args.k)
        cfg.testset_fn = ["{}/output/pairset/k{}/{}.npy".format(args.exp_root, args.k, ip) for ip in args.mediator['input']]
        if args.evaluation:
            cfg.testset_label = "{}/output/pairset/k{}/pair_label.npy".format(args.exp_root, args.k)
        else:
            cfg.testset_label = None

        cfg.test_output = "{}/output/pairset/k{}/pairs_pred.npy".format(args.exp_root, args.k)

        cfg.val = False

        cfg.gpus = args.mediator['gpus']
        cfg.th = args.mediator['threshold']
        mlp_cfg = {'base_lr': 0.001, 'lr_decay_steps': [2], 'lr_decay_scale': 0.1, 
                'momentum': 0.9, 'weight_decay': 0.0001, 'batch_size': 1024,
                'max_epoch': 1, 'average_stats': 200, 'print_freq': 200}
        for k,v in mlp_cfg.items():
            setattr(cfg, k, v)

        self.cfg = cfg

    def train(self):
        train(self.cfg)

    def test(self):
        test(self.cfg)

class ArgObj(object):
    def __init__(self):
        pass
