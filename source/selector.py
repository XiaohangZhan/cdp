import numpy as np
import os
from datetime import datetime
import logging
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils.distributed_utils import dist_init, average_gradients, DistModule, DistributedSequentialSampler, gather_tensors
from utils.common_utils import AverageMeter, accuracy, load_state, save_state, log
from tensorboard import summary
from tensorboard import FileWriter

import pdb

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str)
    parser.add_argument('--input-dim', type=int)
    parser.add_argument('--base-lr', type=float)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--load-path', type=str)
    parser.add_argument('--recover', action='store_true')
    parser.add_argument('--lr-decay-steps', nargs='+', type=int)
    parser.add_argument('--lr-decay-scale', type=float, default=0.1)
    parser.add_argument('--trainset-fn', type=str)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--max-epoch', type=int)
    parser.add_argument('--average-stats', type=int, default=100)
    parser.add_argument('--print-freq', type=int, default=20)
    parser.add_argument('--th', type=float, default=0.5)
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--valset-fn', type=str)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--testset-fn', type=str)
    parser.add_argument('--test-output', type=str)
    parser.add_argument('--port', type=str, default='23456')
    args = parser.parse_args()
    cfg = args
    return cfg

class SelectorNet(nn.Module):
    def __init__(self, input_dim):
        super(SelectorNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 50)
        self.fc2 = nn.Linear(50, 50)
        #self.fc3 = nn.Linear(50, 50)
        #self.fc4 = nn.Linear(50, 50)
        self.fc_last = nn.Linear(50, 2)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        #x = self.fc3(x)
        #x = self.relu(x)
        #x = self.fc4(x)
        #x = self.relu(x)
        x = self.fc_last(x)
        return x

class SelectorDataset(Dataset):
    def __init__(self, filename):
        self.data = np.load(filename)
        self.feat = self.data[:,:-1].astype(np.float32)
        self.label = self.data[:,-1].astype(np.int)
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        return self.feat[idx,:], self.label[idx]

def train(cfg):
    rank, world_size = dist_init(cfg.port)
    assert(cfg.batch_size % world_size == 0)
    cfg.batch_size = cfg.batch_size // world_size
    save_path = 'snapshot/' + cfg.exp_name + '/checkpoint'
    log_path = 'log/' + cfg.exp_name
    if rank == 0 and not os.path.isdir(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    if rank == 0 and not os.path.isdir(log_path):
        os.makedirs(log_path)
    if rank == 0:
        logging.basicConfig(filename=os.path.join(log_path, 'log-{}-{:02d}-{:02d}_{:02d}:{:02d}:{:02d}.txt'.format(
            datetime.today().year, datetime.today().month, datetime.today().day,
            datetime.today().hour, datetime.today().minute, datetime.today().second)),
            level=logging.INFO)
        #loggers = FileWriter(log_path)
        loggers = None
    else:
        loggers = None

    model = SelectorNet(cfg.input_dim)
    if rank == 0:
        print(model)
    model.cuda()
    model = DistModule(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), cfg.base_lr,
                                momentum=cfg.momentum,
                                weight_decay=cfg.weight_decay)

    start_epoch = 0
    count = [0]
    if cfg.load_path:
        assert(os.path.isfile(cfg.load_path))
        if cfg.recover:
            checkpoint = load_state(cfg.load_path, model, optimizer)
            start_epoch = checkpoint['epoch']
            if 'count' in checkpoint.keys():
                count[0] = checkpoint['count']
        else:
            load_state(cfg.load_path, model)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg.lr_decay_steps, gamma=cfg.lr_decay_scale, last_epoch=start_epoch-1)
    cudnn.benchmark = True

    # data
    train_dataset = SelectorDataset(cfg.trainset_fn)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=False, sampler=train_sampler)
    if cfg.val:
        val_dataset = SelectorDataset(cfg.valset_fn)
        val_sampler = DistributedSequentialSampler(val_dataset)
        val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, sampler=val_sampler)

    # train
    for epoch in range(start_epoch, cfg.max_epoch):
        lr_scheduler.step()
        train_sampler.set_epoch(epoch)
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

            reduced_loss = loss.data.clone()
            reduced_prec1 = prec1[0].clone() / world_size

            dist.reduce(reduced_loss, 0)
            dist.reduce(reduced_prec1, 0)

            losses.update(reduced_loss[0])
            top1.update(reduced_prec1[0])

            optimizer.zero_grad()
            loss.backward()
            average_gradients(model)
            optimizer.step()

            if i % cfg.print_freq == 0 and rank == 0:
                log('Epoch: [{0}][{1}/{2}][{3}]    '
                    'Lr: {4:.2g}    '
                    'Loss {loss.val:.4f} ({loss.avg:.4f})    '
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch, i, len(train_loader), count[0],
                    optimizer.param_groups[0]['lr'],
                    loss=losses, top1=top1))
            if rank == 0 and loggers is not None:
                summary_loss = summary.scalar('loss', losses.val)
                summary_top1 = summary.scalar('top1', top1.val)
                loggers.add_summary(summary_loss, count[0])
                loggers.add_summary(summary_top1, count[0])

            count[0] += 1

        if cfg.val:
            validate(model, val_loader, val_dataset.label, epoch, rank, cfg.th)
        if rank == 0:
            save_state({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'count': count[0]},
                save_path, epoch, is_last=(epoch==cfg.max_epoch-1))

def validate(model, val_loader, label, epoch, rank, th):
    model.eval()
    prob = []
    softmax = nn.Softmax(dim=1)
    for i, input in enumerate(val_loader):
        feature_var = torch.autograd.Variable(input[0].cuda())
        output = softmax(model(feature_var))
        prob.append(output.data.cpu().numpy()[:,1])
    prob = np.concatenate(tuple(prob), axis=0)
    all_prob = gather_tensors(prob)
    if rank == 0:
        all_prob = np.concatenate(all_prob, axis=0)
        all_prob = all_prob[:len(label)]
        # evaluate
        all_pred = (all_prob > th).astype(np.int)
        acc, abs_recall, rel_recall, false_pos, false_neg = evaluate(all_pred, label)
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
    rank, world_size = dist_init(cfg.port)
    assert(cfg.batch_size % world_size == 0)
    assert(os.path.isfile(cfg.load_path))
    cfg.batch_size = cfg.batch_size // world_size
    if rank == 0:
        if not os.path.isdir(os.path.dirname(cfg.test_output)):
            os.makedirs(os.path.dirname(cfg.test_output))
    # model
    model = SelectorNet(cfg.input_dim)
    model.cuda()
    model = DistModule(model)
    load_state(cfg.load_path, model)
    cudnn.benchmark = True

    # data
    test_dataset = SelectorDataset(cfg.testset_fn)
    total_size = test_dataset.__len__()
    sampler = DistributedSequentialSampler(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, sampler=sampler)

    # extract
    model.eval()
    prob = []
    softmax = nn.Softmax(dim=1)
    for i, input in enumerate(test_loader):
        feature_var = torch.autograd.Variable(input[0].cuda())
        output = softmax(model(feature_var))
        prob.append(output.data.cpu().numpy()[:,1])

    prob = np.concatenate(tuple(prob), axis=0)
    all_prob = gather_tensors(prob)
    if rank == 0:
        all_prob = np.concatenate(all_prob, axis=0)
        all_prob = all_prob[:total_size]
        np.save(cfg.test_output, all_prob)
        label = test_dataset.label
        # accuracy
        all_pred = (all_prob > cfg.th).astype(np.int)
        acc, abs_recall, rel_recall, false_pos, false_neg = evaluate(all_pred, label)
        print('Testing set. acc: {:.4g}, abs_recall: {:.4g}, rel_recall: {:.4g}, false_pos: {:.4g}, false_neg: {:.4g}'.format(acc, abs_recall, rel_recall, false_pos, false_neg))

def main():
    cfg = parse_args()
    if cfg.test:
        test(cfg)
    else:
        train(cfg)

if __name__ == "__main__":
    main()
