import logging
import copy
from trainer import train_model, val
from model.yolov5.utils.dataloaders import LoadImagesAndLabels
from torch.utils.data import DataLoader
import torch
import collections
import torch.nn as nn
from torch.optim import SGD
from sklearn.metrics import normalized_mutual_info_score
import pytorch_ssim
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum()
    return X_exp / partition 


def train(model, opt, val_set, device, clnt_set, hyp):
    best = 0
    global_model = model
    val_loader=DataLoader(val_set, batch_size=opt.batch_size,collate_fn=LoadImagesAndLabels.collate_fn)
    clnt_model = []
    for cr in range(opt.comm_round):
        print(f'---------------round={cr}:')
        logging.info(f'---------------round={cr}:')
        
        for j in range(opt.clnt_num):
            print(f'=============clint[{j}]:')
            logging.info(f'=============clint[{j}]:')
            if(cr == 0):
                clnt_model.append(copy.deepcopy(global_model))
                clnt_model[j] = train_model(model=clnt_model[j], train_loader=DataLoader(clnt_set[j], batch_size=opt.batch_size,
                                                                     collate_fn=LoadImagesAndLabels.collate_fn),
                            val_loader=val_loader, device=device, hyp=hyp, opt=opt)
            else:
                clnt_model[j] = copy.deepcopy(global_model)
                clnt_model[j] = train_model(model=clnt_model[j], train_loader=DataLoader(clnt_set[j], batch_size=opt.batch_size,
                                                                     collate_fn=LoadImagesAndLabels.collate_fn),
                            val_loader=val_loader, device=device, hyp=hyp, opt=opt)

        total_data_num = 0
        for i in range(len(clnt_set)):
            total_data_num += len(clnt_set[i])

        w = torch.zeros(opt.clnt_num)
        loader = DataLoader(val_set, batch_size=opt.batch_size,collate_fn=LoadImagesAndLabels.collate_fn)
        for batch_i, (im, targets, paths, shapes) in enumerate(loader):
            a = []
            im = im.float().to(device, non_blocking=True)
            O = global_model(im, step=2)
            O = torch.max(O, dim=1)[0]
            O = O.detach().cpu().numpy().flatten()
            for c in range(opt.clnt_num):
                o = clnt_model[c](im, step=2)
                o = torch.max(o, dim=1)[0]
                o = o.detach().cpu().numpy().flatten()
                MI_score = normalized_mutual_info_score(O, o)
                ssim_value = pytorch_ssim.ssim(O, o)
                a.append(MI_score * ssim_value)
            w += a
        w /= len(val_set)
        for i in range(opt.clnt_num):
            w[i] = w[i] * len(clnt_set[i])
        w = softmax(w)
        

        worker_state_dict = [x.state_dict() for x in clnt_model]
        weight_keys = list(worker_state_dict[0].keys())
        global_model_state_dict = collections.OrderedDict()
        for key in weight_keys:
            key_sum = 0
            for i in range(len(clnt_model)):
                key_sum = key_sum +  w[i] * worker_state_dict[i][key]
               
            global_model_state_dict[key] = key_sum
        global_model.load_state_dict(global_model_state_dict)


        loss_f = nn.SmoothL1Loss()
        for k, v in global_model.named_parameters():
            v.requires_grad = True
        global_model.eval()
        for i in range(len(clnt_model)):
            clnt_model[i].eval()
        optimizer = SGD([p for p in global_model.parameters() if p.requires_grad], lr=0.0001)
        loader = DataLoader(val_set, batch_size=1,collate_fn=LoadImagesAndLabels.collate_fn)
        for batch_i, (im, targets, paths, shapes) in enumerate(loader):
            im = im.float().to(device, non_blocking=True)
            output = [0, 0, 0]
            for i in range(len(clnt_model)):
                p = clnt_model[i](im)[1]
                output[0] += w[i] * p[0]
                output[1] += w[i] * p[1]
                output[2] += w[i] * p[2]
            t = global_model(im)[1]
            loss = loss_f(output[0], t[0]) + loss_f(output[1], t[1]) + loss_f(output[2], t[2])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        r = val(model=global_model, data_loader=val_loader, device=device, opt=opt, verbose=True)
        if(r > best):
            r = best
            torch.save(global_model_state_dict, f'pt/{opt.save_name}.pt')