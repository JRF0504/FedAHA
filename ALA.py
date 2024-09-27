import random
from torch.utils.data import DataLoader
from typing import List, Tuple
import torch
import numpy as np
import logging
import copy
import yaml
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
from model.yolov5.utils.general import one_cycle
from model.yolov5.utils.loss import ComputeLoss
from torch.optim import lr_scheduler
from model.yolov5.utils.dataloaders import LoadImagesAndLabels
from torch.utils.data import DataLoader
from model.yolov5.utils.torch_utils import torch_distributed_zero_first
from Dataset import load_yaml
import os
import collections
from trainer import train_model, val
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))


def aggregatemodel(clnt_set, global_model, mdl_list, opt):
    total_data = 0
    for i in range(opt.clnt_num):
        total_data += len(clnt_set[i])
    worker_state_dict = [x.state_dict() for x in mdl_list]
    weight_keys = list(worker_state_dict[0].keys())
    global_model_state_dict = collections.OrderedDict()
    for key in weight_keys:
        key_sum = 0
        for i in range(len(mdl_list)):
            key_sum = key_sum + len(clnt_set[i]) * worker_state_dict[i][key]
        global_model_state_dict[key] = key_sum / total_data
    global_model.load_state_dict(global_model_state_dict)
    torch.save(global_model_state_dict, f'pt/{opt.save_name}.pt')

def initial_model(cid, clnt_model, w, global_model, dataset, opt, device, start_phase):
    loader = DataLoader(dataset, batch_size=opt.batch_size, collate_fn=LoadImagesAndLabels.collate_fn)
    params_g = list(global_model.parameters())
    params = list(clnt_model.parameters())
    # 第一轮不计算w
    if torch.sum(params_g[0] - params[0]) == 0:
        return
    for param, param_g in zip(params[:-opt.layer_idx], params_g[:-opt.layer_idx]):
        param.data = param_g.data.clone()
    # temp模型只用来计算梯度
    model_t = copy.deepcopy(clnt_model)
    params_t = list(model_t.parameters())
    compute_loss = ComputeLoss(model_t)
    # 只合并模型的高层
    params_p = params[-opt.layer_idx:]
    params_gp = params_g[-opt.layer_idx:]
    params_tp = params_t[-opt.layer_idx:]
    if w[cid] == None:
        w[cid] = [torch.ones_like(param.data).to(device) for param in params_p]
        start_phase[cid] = True
    else:
        start_phase[cid] = False
    # frozen the lower layers to reduce computational cost in Pytorch
    for param in params_t[:-opt.layer_idx]:
        param.requires_grad = False
    # used to obtain the gradient of higher layers
    # no need to use optimizer.step(), so lr=0
    optimizer = torch.optim.SGD(params_tp, lr=0)
    # initialize the higher layers in the temp local model
    for param_t, param, param_g, weight in zip(params_tp, params_p, params_gp, w[cid]):
        param_t.data = param + (param_g - param) * weight
    losses = []  # record losses
    cnt = 0  # weight training iteration counter
    model_t.to(device)
    model_t.train()
    while True:
        for x, y, paths, _ in loader:
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device)
            optimizer.zero_grad()
            output = model_t(x)
            loss_value , loss_items= compute_loss(output, y)  # modify according to the local objective
            loss_value.backward()

            # update weight in this batch
            for param_t, param, param_g, weight in zip(params_tp, params_p, params_gp, w[cid]):
                weight.data = torch.clamp(weight - opt.eta * (param_t.grad * (param_g - param)), 0, 1)

            # update temp local model in this batch
            for param_t, param, param_g, weight in zip(params_tp, params_p, params_gp, w[cid]):
                param_t.data = param + (param_g - param) * weight

        losses.append(loss_value.item())
        cnt += 1

        # only train one epoch in the subsequent iterations
        if not start_phase[cid]:
            break

        # train the weight until convergence
        if len(losses) > opt.num_pre_loss and np.std(losses[-opt.num_pre_loss:]) < opt.threshold:
            logging.info(f'Client:{cid}, Std: {np.std(losses[-opt.num_pre_loss:])}, ALA epochs: {cnt}')
            break


    # obtain initialized local model
    for param, param_t in zip(params_p, params_tp):
        param.data = param_t.data.clone()
def main(model, clnt_set, opt, device, hyp, val_set):
    start_phase = []
    val_loader = DataLoader(val_set, batch_size=opt.batch_size,
                            collate_fn=LoadImagesAndLabels.collate_fn)
    for i in range(opt.clnt_num):
        start_phase.append(False)
    w = []
    for i in range(opt.clnt_num):
        w.append(None)
    global_model = model
    clnt_model = []
    for i in range(opt.clnt_num):
        clnt_model.append(copy.deepcopy(global_model))
    for i in range(opt.comm_round):
        print(f'---------------round={i}:')
        logging.info(f'---------------round={i}:')
        for clnt in range(opt.clnt_num):
            print(f'===========client:{clnt}')
            logging.info(f'===========client:{clnt}')
            print('Initial Model...')
            logging.info('Initial Model...')
            initial_model(clnt, clnt_model[clnt], w, global_model, clnt_set[clnt], opt, device, start_phase)
            print('Train Model...')
            logging.info('Train Model...')
            train_model(model=clnt_model[clnt], train_loader=DataLoader(clnt_set[clnt], batch_size=opt.batch_size,
                                                                     collate_fn=LoadImagesAndLabels.collate_fn),
                        val_loader=val_loader, device=device, hyp=hyp, opt=opt)
        aggregatemodel(clnt_set, global_model, clnt_model, opt)
        logging.info('global model:')
        val(model=global_model, data_loader=val_loader, device=device, opt=opt, verbose=True)
        torch.save(global_model.state_dict(), f'pt/{opt.save_name}.pt')
