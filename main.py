import argparse
import collections
import copy
import os
from Dataset import NWPUDataset, load_yaml
import torch
import numpy as np
import yaml
from FedAHA import train as FedAHA_train
from SCAFFOLD import main as SCAFFOLD_train
from model.yolov5.models.yolo import Model
from model.yolov5.utils.dataloaders import LoadImagesAndLabels
from model.yolov5.utils.torch_utils import intersect_dicts
from torch.utils.data import Subset, DataLoader
from model.yolov5.utils.torch_utils import de_parallel
from copy import deepcopy
import logging
from trainer import  val
from ALA import main as ALA_Train
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', default=True, help='pretrained')
    parser.add_argument('--weights', type=str, default='./model/yolov5/models/yolov5s.pt', help='initial weights path')
    parser.add_argument('--save_name', type=str, default='nwpu/avg_4', help='')
    parser.add_argument('--data_path', type=str, default='./model/yolov5/data/NWPU.yaml', help='data path')
    parser.add_argument('--hyp', type=str, default='./model/yolov5/data/hyps/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--cfg', type=str, default='./model/yolov5/models/yolov5s.yaml', help='model.yaml path')
    parser.add_argument('--batch_size', default=64, help='batch size')
    parser.add_argument('--clnt_num', default=4, help='num of clients')
    parser.add_argument('--comm_round', default=10, help='')
    parser.add_argument('--epoch', default=10, help='')
    parser.add_argument('--single_cls', action='store_true', help='')
    parser.add_argument('--img_size', default=512, help='image size')
    parser.add_argument('--nc', default=10, help='number of classes')

    parser.add_argument('--act_prob', default=1.0, help='SCAFFOLD act probability')
    parser.add_argument('-tp', "--t_layer_idx", type=int, default=6, help="ALA top layer_idx")
    parser.add_argument('-bp', "--b_layer_idx", type=int, default=48, help="ALA bottom layer_idx")
    # parser.add_argument('-p', "--layer_idx", type=int, default=2, help="ALA layer_idx")
    parser.add_argument('-et', "--eta", type=float, default=1.0, help='ALA eta')
    parser.add_argument('--num_pre_loss', default=10, help='ALA number of loss')
    parser.add_argument('--w_epoch', default=10, help='ALA number of loss')
    parser.add_argument('--threshold', default=1.0, help='ALA threshold')
    opt = parser.parse_args()
    return opt
if __name__ == '__main__':
    opt = parse_opt()
    logging.basicConfig(format='%(asctime)s - %(filename)s[line5 %(levelname)s: %(message)s',
                        level=logging.DEBUG,
                        filename=f'log/{opt.save_name}.log',
                        filemode='a')  # 配置输出格式、日志级别、存储文件及文件打开模式
    dataset = NWPUDataset(opt=opt)
    train_set, val_set = dataset.train_set, dataset.val_set
    data_num = len(train_set)
    print(data_num)
    # np.random.seed(1)
    # data_idx = np.random.permutation(data_num)
    data_idx = range(data_num)
    clnt_set = []
    for i in range(opt.clnt_num):
        s = int(i * data_num / opt.clnt_num)
        t = int( (i + 1)* data_num / opt.clnt_num)
        clnt_set .append(Subset(train_set, data_idx[s : t])) 
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if isinstance(opt.hyp, str):
        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    if opt.pretrained:
        ckpt = torch.load(opt.weights, map_location=device)  # load checkpoint
        model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=opt.nc, anchors=hyp.get('anchors')).to(device)  # create
        exclude = ['anchor'] if (opt.cfg or hyp.get('anchors'))  else []  # exclude keys
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(csd, strict=False)  # load
        # print(f'Transferred {len(csd)}/{len(model.state_dict())} items from {opt.weights}')
        logging.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {opt.weights}')
    else:
        model = Model(opt.cfg, ch=3, nc=opt.nc, anchors=hyp.get('anchors')).to(device)  # create
    nl = model.model[-1].nl
    hyp['box'] *= 3. / nl
    hyp['cls'] *= opt.nc / 80. * 3. / nl
    hyp['obj'] *= (opt.img_size / 640) ** 2 * 3. / nl
    hyp['label_smoothing'] = 0.0
    model.nc = opt.nc
    model.hyp = hyp
    data_dict = load_yaml(opt.data_path)
    model.names = data_dict['names']
    # model.load_state_dict(torch.load('pt/FedAvg.pt', map_location=device))
    # all_loader = DataLoader(train_set, batch_size=opt.batch_size, collate_fn=LoadImagesAndLabels.collate_fn)
    # val_loader = DataLoader(val_set, batch_size=opt.batch_size,collate_fn=LoadImagesAndLabels.collate_fn)
    # SCAFFOLD_train(model=model,clnt_set=clnt_set, opt=opt,device=device,hyp=hyp, val_set=val_set)
    # val(model=model, data_loader=val_loader, device=device, opt=opt, verbose = True)
    FedAHA_train(model=model, opt=opt,val_set=val_set, device=device, clnt_set=clnt_set, hyp=hyp)
    # ALA_Train(model, clnt_set, opt, device, hyp, val_loader)
    # AL(model, clnt_set, opt, device, hyp, val_loader)





