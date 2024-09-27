import os
import torch
from model.yolov5.utils.dataloaders import create_dataloader
# from model.yolov5.utils.dataloaders import create_dataloader
from torch.utils.data import DataLoader, Dataset, distributed, random_split, Subset
from model.yolov5.utils.torch_utils import torch_distributed_zero_first
from pathlib import Path
from zipfile import ZipFile
from model.yolov5.utils.general import colorstr
import numpy as np
import yaml
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))

# class DatasetObject:
#     def __init__(self, dataset, n_client, rule, unbalanced_sgm=0, rule_arg=''):
#         self.dataset = dataset
#         self.n_client = n_client
#         self.config_path = f'./data_config/{dataset}.yaml'
#         self.rule = rule
#         self.rule_arg = rule_arg
#         rule_arg_str = rule_arg if isinstance(rule_arg, str) else '%.3f' % rule_arg
#         self.name = "%s_%d_%s_%s"%(self.dataset,self.n_client,self.rule,rule_arg_str)
#         self.name += '_%f' %unbalanced_sgm if unbalanced_sgm!=0 else ''
#         self.set_data()
#
#     def set_data(self):
#         if not os.path.exists('%s/%s' %(self.data_path,self.name)):
#             pass

def load_yaml(data, autodownload=True):
    # Download and/or unzip dataset if not found locally
    # Usage: https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128_with_yaml.zip

    # Download (optional)
    extract_dir = ''

    # Read yaml (optional)
    if isinstance(data, (str, Path)):
        with open(data, errors='ignore') as f:
            data = yaml.safe_load(f)  # dictionary

    # Parse yaml
    path = extract_dir or Path(data.get('path') or '')  # optional 'path' default to '.'
    for k in 'train', 'val', 'test':
        if data.get(k):  # prepend path
            data[k] = str(path / data[k]) if isinstance(data[k], str) else [str(path / x) for x in data[k]]

    assert 'nc' in data, "Dataset 'nc' key missing."
    if 'names' not in data:
        data['names'] = [f'class{i}' for i in range(data['nc'])]  # assign class names if missing
    train, val, test, s = [data.get(x) for x in ('train', 'val', 'test', 'download')]
    if val:
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]  # val path
        if not all(x.exists() for x in val):
            print('\nWARNING: Dataset not found, nonexistent paths: %s' % [str(x) for x in val if not x.exists()])
            if s and autodownload:  # download script
                root = path.parent if 'path' in data else '..'  # unzip directory i.e. '../'
                if s.startswith('http') and s.endswith('.zip'):  # URL
                    f = Path(s).name  # filename
                    print(f'Downloading {s} to {f}...')
                    torch.hub.download_url_to_file(s, f)
                    Path(root).mkdir(parents=True, exist_ok=True)  # create root
                    ZipFile(f).extractall(path=root)  # unzip
                    Path(f).unlink()  # remove zip
                    r = None  # success
                elif s.startswith('bash '):  # bash script
                    print(f'Running {s} ...')
                    r = os.system(s)
                else:  # python script
                    r = exec(s, {'yaml': data})  # return None
                print(f"Dataset autodownload {f'success, saved to {root}' if r in (0, None) else 'failure'}\n")
            else:
                raise Exception('Dataset not found.')

    return data  # dictionary
class NWPUDataset(Dataset):
    def __init__(self, opt):
        with torch_distributed_zero_first(LOCAL_RANK):
            data_dict = load_yaml(opt.data_path)  # check if None
        if isinstance(opt.hyp, str):
            with open(opt.hyp, errors='ignore') as f:
                hyp = yaml.safe_load(f)  # load hyps dict
        train_path, val_path = data_dict['train'], data_dict['val']
        train_set = create_dataloader(path=train_path, imgsz=opt.img_size, batch_size=opt.batch_size, stride=32,
                                              hyp=hyp, augment=True, cache=False, rect=False, rank=LOCAL_RANK,
                                              workers=0, prefix=colorstr('train: '))[1]
        val_set = create_dataloader(path=val_path, imgsz=opt.img_size, batch_size=opt.batch_size, stride=32,
                                      hyp=hyp,  cache=False, rect=True, rank=-1, pad=0.5,
                                      workers=0, prefix=colorstr('val: '))[1]
        self.train_set = train_set
        self.val_set = val_set
        
class ClientSet(Dataset):
    def __init__(self, opt, path):
        # with torch_distributed_zero_first(LOCAL_RANK):
        #     data_dict = load_yaml(opt.data_path)  # check if None
        if isinstance(opt.hyp, str):
            with open(opt.hyp, errors='ignore') as f:
                hyp = yaml.safe_load(f)  # load hyps dict
        train_set = create_dataloader(path=path, imgsz=opt.img_size, batch_size=opt.batch_size, stride=32,
                                              hyp=hyp, augment=True, cache=False, rect=False, rank=LOCAL_RANK,
                                              workers=0, prefix=colorstr('train: '))[1]
        # self.loader = train_loader
        self.train_set = train_set

        
if __name__ == '__main__':
    # data_dict = load_yaml('./data_config/NWPU.yaml')
    # print(data_dict['train'], data_dict['val'])
    dataset = NWPUDataset(data_path='model/yolov5/data/NWPU.yaml', hyp='hyp.scratch.yaml')
    # print(dataset.set[0][0].size())
    # clnt_num = np.ones(10,dtype=int) * 39
    # clnt_set = random_split(dataset.set, clnt_num)
    sub = Subset(dataset.set, [1,1,2,3,4])
    sub_loader = DataLoader(sub, batch_size=1)
    for x, y, _, __ in sub_loader:
        print(x, y)
    print('done')