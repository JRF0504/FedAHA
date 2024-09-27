import logging
import math
import copy
import numpy as np
import torch
import yaml
import torch.optim as optim
import torch.nn as nn
from torch.cuda import amp
from torch.optim import lr_scheduler
from Dataset import load_yaml
from model.yolov5.utils.general import one_cycle
from model.yolov5.utils.loss import ComputeLoss
from model.yolov5.utils.torch_utils import torch_distributed_zero_first
from model.yolov5.val import run as run_val
import os
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))

def train_model(model, train_loader, val_loader, device, hyp, opt):
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

    # Freeze
    freeze = []  # parameter names to freeze (full or partial)
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print("freezing %s" % k)
            v.requires_grad = False

    lf = one_cycle(1, hyp['lrf'],  opt.epoch)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)


    model.to(device)
    model.train()
    # model.eval()
    compute_loss = ComputeLoss(model)
    cuda = device.type != 'cpu'
    scaler = amp.GradScaler(enabled=cuda)
    for e in range(opt.epoch):
        # print(f'============epoch:{e}============')
        # print(optimizer.state_dict()['param_groups'][0]['lr'])
        # model.train()

        for (batch_idx, batch) in enumerate(train_loader):
            imgs, targets, paths, _ = batch
            imgs = imgs.to(device, non_blocking=True).float() / 256.0 - 0.5


            with amp.autocast(enabled=cuda):
                pred = model(imgs)
                loss, loss_items = compute_loss(pred, targets.to(device))
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()


        scheduler.step()
        if opt.clnt_num == 1:
            logging.info(f'#########eoch:{e}')
            val(model, val_loader, device, opt)
            print(optimizer.state_dict()['param_groups'][0]['lr'])
    val(model, val_loader, device, opt)
    return model


def val(model, data_loader, device, opt, verbose = False):
    print("=" * 30)

    model.eval()
    model.to(device)
    compute_loss = ComputeLoss(model)

    # val
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = load_yaml(opt.data_path)  # check if None

    results, maps, _ = run_val(data_dict, batch_size=opt.batch_size, imgsz=opt.img_size, dataloader=data_loader, 
                               model=model, single_cls=opt.single_cls, plots=False, compute_loss=compute_loss, verbose=verbose)
    #
    # save_dir=args.save_dir,
    mp, mr, map50, map, box_loss, obj_loss, cls_loss = results
    print((f'map: {map}, map_50: {map50}, box_loss: {box_loss}, obj_loss: {obj_loss}, cls_loss: {cls_loss}'))
    logging.info(f'map: {map}, map_50: {map50}, box_loss: {box_loss}, obj_loss: {obj_loss}, cls_loss: {cls_loss}')
    return map50

