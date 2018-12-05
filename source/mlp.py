import numpy as np
import os
from datetime import datetime
import logging
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils import AverageMeter, accuracy, load_state, save_state, log

import pdb

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str)
    parser.add_argument('--input-dim', type=int)
    parser.add_argument('--base-lr', type=float, default=0.05)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--load-path', type=str)
    parser.add_argument('--recover', action='store_true')
    parser.add_argument('--lr-decay-steps', nargs='+', type=int)
    parser.add_argument('--lr-decay-scale', type=float, default=0.1)
    parser.add_argument('--trainset-fn', type=str)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--max-epoch', type=int, default=3)
    parser.add_argument('--average-stats', type=int, default=100)
    parser.add_argument('--print-freq', type=int, default=20)
    parser.add_argument('--th', type=float, default=0.5)
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--valset-fn', type=str)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--testset-fn', type=str)
    parser.add_argument('--test-output', type=str)
    parser.add_argument('--ngpu', type=int)
    parser.add_argument('--port', type=str, default='23456')
    args = parser.parse_args()
    cfg = args
    return cfg

class MediatorNet(nn.Module):
    def __init__(self, input_dim):
        super(MediatorNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc_last = nn.Linear(50, 2)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc_last(x)
        return x

class MediatorDataset(Dataset):
    def __init__(self, filename):
        self.data = np.load(filename)
        self.feat = self.data[:,:-1].astype(np.float32)
        self.label = self.data[:,-1].astype(np.int)
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        return self.feat[idx,:], self.label[idx]

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
    devices = list(range(cfg.ngpu))
    assert(cfg.batch_size % cfg.ngpu == 0)
    cfg.batch_size = cfg.batch_size // cfg.ngpu
    save_path = cfg.exp_root + '/output/mlp/checkpoints/checkpoint'
    log_path = cfg.exp_root + '/output/mlp/log/'
    if not os.path.isdir(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    logging.basicConfig(filename=os.path.join(log_path, 'log-{}-{:02d}-{:02d}_{:02d}:{:02d}:{:02d}.txt'.format(
        datetime.today().year, datetime.today().month, datetime.today().day,
        datetime.today().hour, datetime.today().minute, datetime.today().second)),
        level=logging.INFO)

    model = MediatorNet(infer_dim(cfg))
    model = nn.DataParallel(model, devices=devices)
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
    train_dataset = MediatorDataset(cfg.trainset_fn)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    if cfg.val:
        val_dataset = MediatorDataset(cfg.valset_fn)
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

            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % cfg.print_freq == 0:
                log('Epoch: [{0}][{1}/{2}][{3}]\t'
                    'Lr: {4:.2g}\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
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
    print('Val for epoch #{}. th: {:.4g}, acc: {:.4g}, abs_recall: {:.4g}, rel_recall: {:.4g}, false_pos: {:.4g}, false_neg: {:.4g}'.format(epoch, th, acc, abs_recall, rel_recall, false_pos, false_neg))
    
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
    assert(os.path.isfile(cfg.load_path))
    if not os.path.isdir(os.path.dirname(cfg.test_output)):
        os.makedirs(os.path.dirname(cfg.test_output))
    # model
    model = MediatorNet(cfg.input_dim)
    model.cuda()
    load_state(cfg.load_path, model)
    cudnn.benchmark = True

    # data
    test_dataset = MediatorDataset(cfg.testset_fn)
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
    pred = (prob > cfg.th).astype(np.int)
    acc, abs_recall, rel_recall, false_pos, false_neg = evaluate(pred, label)
    print('Testing set. acc: {:.4g}, abs_recall: {:.4g}, rel_recall: {:.4g}, false_pos: {:.4g}, false_neg: {:.4g}'.format(acc, abs_recall, rel_recall, false_pos, false_neg))

def main():
    cfg = parse_args()
    if cfg.test:
        test(cfg)
    else:
        train(cfg)

def train_mediator(args):
    mlp_cfg = {'base_lr': 0.05, 'momentum': 0.9, 'weight-decay': 0.0001, 'batch_size': 1024,
        'max_epoch': 3, 'average_stats': 100, 'print_freq': 20, 'ngpu': 1}
    for k,v in mlp_cfg.items():
        setattr(args, k, v)
    train(args)

if __name__ == "__main__":
    main()
