import sys
import os

import warnings

from model import CSRNet

from utils import save_checkpoint

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

import numpy as np
import argparse
import json
import cv2
import dataset
import time
import random

parser = argparse.ArgumentParser(description='PyTorch CSRNet')

# parser.add_argument('train_json', metavar='TRAIN',
#                     help='path to train json')
# parser.add_argument('test_json', metavar='TEST',
#                     help='path to test json')

parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None,type=str,
                    help='path to the pretrained model')

# parser.add_argument('gpu',metavar='GPU', type=str,
#                     help='GPU id to use.')

# parser.add_argument('task',metavar='TASK', type=str,
#                     help='task id to use.')

# torch.backends.cudnn.benchmark = True

def main():
    
    global args,best_prec1
    
    best_prec1 = 1e6
    
    args = parser.parse_args()
    args.original_lr = 1e-6
    args.lr = 1e-7
    args.batch_size    = 1
    args.momentum      = 0.95
    args.decay         = 5*1e-4
    args.start_epoch   = 0
    args.epochs = 100
    args.steps         = [-1,1,100,150]
    args.scales        = [1,1,1,1]
    args.workers = 1
    args.seed = time.time()
    args.print_freq = 20
    # with open(args.train_json, 'r') as outfile:        
    #     train_list = json.load(outfile)
    # with open(args.test_json, 'r') as outfile:       
    #     val_list = json.load(outfile)
    
    # get the saved img from the bridge video
    val_dir = './test_image/'
    file_list = os.listdir(val_dir)
    print(file_list)
    val_list = []
    for item in file_list:
        val_list.append(val_dir+item)
    print(val_list)
    # sys.exit()

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # torch.cuda.manual_seed(args.seed)
    
    model = CSRNet()
    
    # model = model.cuda()
    
    # criterion = nn.MSELoss(size_average=False).cuda()
    criterion = nn.MSELoss(size_average=False)
    
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.decay)

    # load the trained model 
    if args.pre:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            checkpoint = torch.load(args.pre)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.pre, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pre))

    # predict
    prec1 = validate(val_list, model, criterion)
    print(prec1)
    
def validate(val_list, model, criterion):
    print ('begin test')
    test_loader = torch.utils.data.DataLoader(
    dataset.listDataset(val_list,
                   shuffle=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ]),  train=False, prediction=True),
    batch_size=args.batch_size)    
    
    model.eval()
    
    mae = 0
    
    for i,(img, img_path_returned) in enumerate(test_loader):
        # img = img.cuda()
        img = Variable(img)
        output = model(img)
        
        print(f'{img_path_returned}: prediction: {output.data.sum()}')

    return True  
        
    
if __name__ == '__main__':
    main()        