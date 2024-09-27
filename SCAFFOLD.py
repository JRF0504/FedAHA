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
from model.yolov5.val import run as run_val
from trainer import val
from torch.optim import SGD
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))

def get_mdl_params(model_list, n_par=None):
    if n_par == None:
        exp_mdl = model_list[0]
        n_par = 0
        for name, param in exp_mdl.named_parameters():
            n_par += len(param.data.reshape(-1))

    param_mat = np.zeros((len(model_list), n_par)).astype('float32')
    for i, mdl in enumerate(model_list):
        idx = 0
        for name, param in mdl.named_parameters():
            temp = param.data.cpu().numpy().reshape(-1)
            param_mat[i, idx:idx + len(temp)] = temp
            idx += len(temp)
    return np.copy(param_mat)
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum()
    return X_exp / partition 

def aggregatemodel(clnt_set, global_model, mdl_list, opt, device, val_set):
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

def main(model, clnt_set, opt, device, hyp, val_set):
    global_model = model
    global_c = [torch.zeros_like(param).to(device) for param in global_model.parameters()]
    clnt_c = list(range(opt.clnt_num))
    for i in range(opt.clnt_num):
        clnt_c[i] = copy.deepcopy(global_c)
    clnt_model = list(range(opt.clnt_num))
    for i in range(opt.comm_round):
        inc_seed = 0
        logging.info(f'---------------round={i}:')
        while(True):
            np.random.seed(i + inc_seed)
            act_list = np.random.uniform(size=opt.clnt_num)
            act_clients = act_list <= opt.act_prob
            selected_clnts = np.sort(np.where(act_clients)[0])
            inc_seed += 1
            if len(selected_clnts) != 0:
                break
        dy = [torch.zeros_like(param).to(device) for param in global_model.parameters()]
        dc = [torch.zeros_like(param).to(device) for param in global_model.parameters()]
        for clnt in selected_clnts:
            logging.info(f'===========client:{clnt}')
            clnt_model[clnt] = copy.deepcopy(global_model)
            delta_c, delta_y = train(global_model, clnt_model[clnt], val_set, opt, clnt_set[clnt], hyp, device, global_c, clnt_c[clnt])
            for _,__ in zip(dc, delta_c):
                _ += 1 / opt.clnt_num * __

            for _,__ in zip(dy, delta_y):
                _ += 1 / opt.clnt_num * __
            # dc += 1.0 / float(opt.clnt_num) * delta_c
            # dy += 1.0 / float(opt.clnt_num) * delta_y

        # for para, d_y in zip(global_model.parameters(), dy):
        #     para = para + global_rate * d_y
        aggregatemodel(clnt_set, global_model, clnt_model, opt, device, val_set)
        # global_rate *= 0.9
        for g_c , d_c in zip(global_c, dc):
            g_c += d_c
        val(global_model, DataLoader(val_set, batch_size=opt.batch_size,collate_fn=LoadImagesAndLabels.collate_fn), device ,opt, True)

def train(global_model, model, val_set, opt, dataset, hyp, device, global_c, clnt_c):
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay

    optimizer = optim.SGD(
            pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True
        )

    optimizer.add_param_group(
        {"params": pg1, "weight_decay": hyp['weight_decay']}
    )  # add pg1 with weight_decay
    optimizer.add_param_group({"params": pg2})  # add pg2 (biases)
    del pg0, pg1, pg2
    freeze = []  # parameter names to freeze (full or partial)
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print("freezing %s" % k)
            v.requires_grad = False

    lf = one_cycle(1, hyp['lrf'], opt.epoch)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    model.to(device)
    model.train()
    compute_loss = ComputeLoss(model)
    cuda = device.type != 'cpu'
    scaler = amp.GradScaler(enabled=cuda)
    train_loader = DataLoader(dataset, batch_size=opt.batch_size ,collate_fn=LoadImagesAndLabels.collate_fn)
    c_diff = []
    for c_l, c_g in zip(clnt_c, global_c):
        c_diff.append(-c_l + c_g)
    for e in range(opt.epoch):
        model.train()
        for (batch_idx, batch) in enumerate(train_loader):
            imgs, targets, paths, _ = batch
            imgs = imgs.to(device, non_blocking=True).float() / 256.0 - 0.5

            with amp.autocast(enabled=cuda):
                pred = model(imgs)
                loss, loss_items = compute_loss(pred, targets.to(device))

            scaler.scale(loss).backward()
            for param, c_d in zip(model.parameters(), c_diff):
                param.grad += c_d.data
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        scheduler.step()

    y_delta = []
    c_plus = []
    c_delta = []

    # compute y_delta (difference of model before and after training)
    for param_l, param_g in zip(model.parameters(), global_model.parameters()):
        y_delta.append(param_l - param_g)

    # compute c_plus
    # local_lr = 0.01
    coef = 1 / (optimizer.state_dict()['param_groups'][0]['lr'])
    for c_l, c_g, diff in zip(clnt_c, global_c, y_delta):
        c_plus.append(c_l - c_g - coef * diff)

    # compute c_delta
    for c_p, c_l in zip(c_plus, clnt_c):
        c_delta.append(c_p - c_l)

    clnt_c = c_plus


    val(model, DataLoader(val_set, batch_size=opt.batch_size,collate_fn=LoadImagesAndLabels.collate_fn), device, opt)
    return c_delta, y_delta

