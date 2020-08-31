from __future__ import print_function
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as data
from data import WiderFaceDetection, detection_collate, preproc, cfg_mnet, cfg_re50
from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
import time
import datetime
import math
import random
import numpy as np
from models.retinaface import RetinaFace

parser = argparse.ArgumentParser(description='Retinaface Training')
parser.add_argument('--training_dataset', default='./data/widerface/train/label.txt', help='Training dataset directory')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--save_folder', default='./weights/', help='Location to save checkpoint models')
parser.add_argument('--optim', default='SGD', type=str, help='name of the dataset to train')
parser.add_argument('--lr_type', default='cos', type=str, help='lr scheduler (exp/cos/step3/fixed)')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
cfg = None
if args.network == "mobile0.25":
    cfg = cfg_mnet
elif args.network == "resnet50":
    cfg = cfg_re50

rgb_mean = (104, 117, 123) # bgr order
num_classes = 2
img_dim = cfg['image_size']
num_gpu = cfg['ngpu']
batch_size = cfg['batch_size']
max_epoch = cfg['epoch']
gpu_train = cfg['gpu_train']

num_workers = args.num_workers
momentum = args.momentum
weight_decay = args.weight_decay
initial_lr = args.lr
gamma = args.gamma
training_dataset = args.training_dataset
save_folder = args.save_folder

net = RetinaFace(cfg=cfg)
print("Printing net...")
#print(net)

if args.resume_net is not None:
    print('Loading resume network...')
    state_dict = torch.load(args.resume_net)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

if num_gpu > 1 and gpu_train:
    net = torch.nn.DataParallel(net).cuda()
else:
    net = net.cuda()

cudnn.benchmark = True


optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)

priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
with torch.no_grad():
    priors = priorbox.forward()
    priors = priors.cuda()

def get_lrs(optimizer):
    lrs = []
    for pgs in optimizer.state_dict()['param_groups']:
        lrs.append(pgs['lr'])
    lrs = ['{:.6f}'.format(x) for x in lrs]
    return lrs

def train():
    net.train()
    epoch = 0 + args.resume_epoch
    print('Loading Dataset...')

    dataset = WiderFaceDetection( training_dataset,preproc(img_dim, rgb_mean))

    epoch_size = math.ceil(len(dataset) / batch_size)
    
    if args.optim == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    else:
        raise ValueError('wrong opitm')
    if args.lr_type == 'plateau':
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=0, min_lr=1e-4)
    elif args.lr_type == 'cos':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    else:
        lr_scheduler = None

    #from copy import deepcopy
    torch.multiprocessing.set_sharing_strategy('file_system')
    #train_loader = data.DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate, drop_last=True)
    #for images, targets in train_loader:
    #    print('batch')
    #print('done')
    train_loader = data.DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate)

    #for iteration in range(start_iter, max_iter):
    for epoch in range(max_epoch):
        load_t0 = time.time()
        for batch_idx, (images, targets) in enumerate(train_loader):    
            images = images.cuda()
            targets = [anno.cuda() for anno in targets]

            #mixup
            if random.random() < 0.4 and epoch >= 0:
                shuffle_indices = torch.randperm(images.size(0))
                indices = torch.arange(images.size(0))
                lam = np.clip(np.random.beta(1.0, 1.0), 0.35, 0.65)
                images = lam * images + (1-lam) * images[shuffle_indices, :]
                mix_targets = []
                for i, si in zip(indices, shuffle_indices):
                    #print(targets[i.item()].size())
                    if i.item() == si.item():
                        target = targets[i.item()]
                    else:
                        target = torch.cat([targets[i.item()], targets[si.item()]], 0)
                    mix_targets.append(target)
                #targets = torch.cat(mix_targets, 0)   
                #print('mix:', len(targets), targets[0].size())
            else:
                #print('norm:', len(targets), targets[0].size()) 
                pass

            # forward
            out = net(images)

            # backprop
            optimizer.zero_grad()
            loss_l, loss_c, loss_landm = criterion(out, priors, targets)
            loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
            loss.backward()
            optimizer.step()
            load_t1 = time.time()
            #batch_time = load_t1 - load_t0
            lr = get_lrs(optimizer)[0]
            print('Epoch:{}/{} || Iter: {}/{} || Loc: {:.4f} Cla: {:.4f} Landm: {:.4f} || LR: {} || {:.2f} min'
                .format(epoch, max_epoch, batch_idx, epoch_size, loss_l.item(), loss_c.item(), loss_landm.item(), lr, (load_t1-load_t0)/60), end='\r')

            
        #epoch done
        print("")
        torch.save(net.state_dict(), save_folder + cfg['name'] + '_latest.pth')
        lr_scheduler.step()


def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    warmup_epoch = -1
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr-1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__ == '__main__':
    train()
