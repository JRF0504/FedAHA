import argparse
import collections
import copy
import os
from Dataset import NWPUDataset, load_yaml
import torch
import numpy as np
import yaml
from model.yolov5.models.yolo import Model
from model.yolov5.utils.dataloaders import LoadImagesAndLabels
from model.yolov5.utils.torch_utils import intersect_dicts
from torch.utils.data import Subset, DataLoader
from copy import deepcopy
import logging
import matplotlib.pyplot as plt
from trainer import  val
from trainer import train_model, val
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', default=True, help='pretrained')
    parser.add_argument('--weights', type=str, default='./model/yolov5/models/yolov5s.pt', help='initial weights path')
    parser.add_argument('--save_name', type=str, default='nwpu/Avg_w22', help='')
    parser.add_argument('--data_path', type=str, default='./model/yolov5/data/NWPU.yaml', help='data path')
    parser.add_argument('--hyp', type=str, default='./model/yolov5/data/hyps/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--cfg', type=str, default='./model/yolov5/models/yolov5s.yaml', help='model.yaml path')
    parser.add_argument('--batch_size', default=16, help='batch size')
    parser.add_argument('--single_cls', action='store_true', help='')
    parser.add_argument('--img_size', default=512, help='image size')
    parser.add_argument('--nc', default=10, help='number of classes')

    
    opt = parser.parse_args()
    return opt

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum()
    return X_exp / partition 

if __name__ == '__main__':
    opt = parse_opt()
    dataset = NWPUDataset(opt=opt)
    train_set, val_set = dataset.train_set, dataset.val_set
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if isinstance(opt.hyp, str):
        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    if opt.pretrained:
        ckpt = torch.load(opt.weights, map_location=device)  # load checkpoint
        model = Model(opt.cfg, ch=3, nc=opt.nc, anchors=hyp.get('anchors')).to(device)  # create
        exclude = ['anchor'] if (opt.cfg or hyp.get('anchors'))  else []  # exclude keys
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(csd, strict=False)  # load
        print(f'Transferred {len(csd)}/{len(model.state_dict())} items from {opt.weights}')
    else:
        model = Model(opt.cfg, ch=3, nc=opt.nc, anchors=hyp.get('anchors')).to(device)  # create
    model.nc = opt.nc
    model.hyp = hyp
    data_dict = load_yaml(opt.data_path)
    model.names = data_dict['names']
    val_loader=DataLoader(val_set, batch_size=opt.batch_size,collate_fn=LoadImagesAndLabels.collate_fn)
    model.load_state_dict(torch.load('pt/nwpu/Avg.pt'), strict=False)
    val(model, val_loader, device, opt, True)