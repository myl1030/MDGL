# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 17:23:51 2022

@author: 86178
"""

import os
from model import *
import util
import scipy.io
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch
import numpy as np
from random import randrange 
from einops import repeat
from sklearn.model_selection import StratifiedKFold
from random import shuffle
import random
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchvision.utils import make_grid
from torch.utils.data import Dataset
from torch import tensor, float32, save, load



class DatasetASD(Dataset):
    def __init__(self, k_fold=None):
        super().__init__()
        
        # AAL Spatial Scale
        site1 = scipy.io.loadmat('F:\\code\\xin\\NYU116.mat')
        bold1 =site1['AAL']
        A1 =bold1[0]
        series1=[]
        for i in range(len(A1)):
            signal1=A1[i]
            series1.append(signal1)
        sample=len(series1)
        numbers = [int(x) for x in range(sample)]
        d1=zip(numbers,series1)
        self.timeseries_dict1=dict(d1)
        self.num_timepoints1, self.num_nodes1 = list(self.timeseries_dict1.values())[0].shape
        
        # CC200 Spatial Scale
        site2 = scipy.io.loadmat('F:\\code\\xin\\NYU200.mat')
        bold2 =site2['AAL2']
        A2 =bold2[0]
        series2=[]
        for i in range(len(A2)): 
            signal2=A2[i]
            series2.append(signal2)
        d2=zip(numbers,series2)
        self.timeseries_dict2=dict(d2)
        self.num_timepoints2, self.num_nodes2 = list(self.timeseries_dict2.values())[0].shape
        
        self.full_subject_list = list(self.timeseries_dict2.keys())
        if k_fold is None:
            self.subject_list = self.full_subject_list
        else:
            self.k_fold = StratifiedKFold(k_fold, shuffle=True, random_state=0) if k_fold is not None else None
            self.k = None

        # label
        y=site1['lab']
        y= np.squeeze(y)
        y=y.tolist()
        dy=zip(numbers,y)
        self.behavioral_dict=dict(dy)
        self.full_label_list = [self.behavioral_dict[int(subject)] for subject in self.full_subject_list]


    def __len__(self):
        return len(self.subject_list) if self.k is not None else len(self.full_subject_list)


    def set_fold(self, fold, train=True):
        assert self.k_fold is not None
        self.k = fold
        train_idx, test_idx = list(self.k_fold.split(self.full_subject_list, self.full_label_list))[fold]
        if train: shuffle(train_idx)
        self.subject_list = [self.full_subject_list[idx] for idx in train_idx] if train else [self.full_subject_list[idx] for idx in test_idx]


    def __getitem__(self, idx):
        subject = self.subject_list[idx]
        timeseries1 = self.timeseries_dict1[subject]
        timeseries2 = self.timeseries_dict2[subject]
        timeseries1 = (timeseries1 - np.mean(timeseries1, axis=0, keepdims=True)) / np.std(timeseries1, axis=0, keepdims=True)
        timeseries2 = (timeseries2 - np.mean(timeseries2, axis=0, keepdims=True)) / np.std(timeseries2, axis=0, keepdims=True)
        label = self.behavioral_dict[int(subject)]

        if label==0:
            label = tensor(0)
        elif label==1:
            label = tensor(1)
        else:
            raise

        return {'id': subject, 'timeseries1': tensor(timeseries1, dtype=float32),'timeseries2': tensor(timeseries2, dtype=float32), 'label': label}
    
    
def step(model, criterion, dyn_v1, dyn_a1, sampling_endpoints1, t1, dyn_v2, dyn_a2, sampling_endpoints2, t2, label,reg_lambda,clip_grad=0.0, device='cpu', optimizer=None):
    if optimizer is None: model.eval()
    else: model.train()

    # run model
    logit,reg_ortho1,reg_ortho2 = model(dyn_v1.to(device), dyn_a1.to(device), t1.to(device), sampling_endpoints1, dyn_v2.to(device), dyn_a2.to(device), t2.to(device), sampling_endpoints2)
    loss = criterion(logit, label.to(device))
    reg_ortho1 *= reg_lambda
    reg_ortho2 *= reg_lambda
    loss += reg_ortho1+reg_ortho2

    # optimize model
    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        if clip_grad > 0.0: torch.nn.utils.clip_grad_value_(model.parameters(), clip_grad)
        optimizer.step()
        
    return logit

#### train

# make directories
os.makedirs(os.path.join('result', 'model'), exist_ok=True)
os.makedirs(os.path.join('result', 'summary'), exist_ok=True)

# set seed and device
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.manual_seed_all(0)
else:
    device = torch.device("cpu")

# define dataset
dataset = DatasetASD(k_fold=5)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)  

# resume checkpoint if file exists
if os.path.isfile(os.path.join('result', 'checkpoint.pth')):
    print('resuming checkpoint experiment')
    checkpoint = torch.load(os.path.join('result', 'checkpoint.pth'), map_location=device)
else:
    checkpoint = {
        'fold': 0,
        'epoch': 0,
        'model': None,
        'optimizer': None}

# start experiment
for k in range(checkpoint['fold'],5):
    # make directories per fold
    os.makedirs(os.path.join('result', 'model', str(k)), exist_ok=True)

    # set dataloader
    dataset.set_fold(k, train=True)

    # define model
    model = Fusion(
        input_dim1=116, # number of brain regions at AAL spatial scale
        input_dim2=200, # number of brain regions at CC200 spatial scale
        hidden_dim=64,
        num_classes=2,
        num_heads=1,
        num_layers=2,
        sparsity=30,
        dropout=0.5,
        cls_token='sum',
        readout='sero')
    model.to(device)
    if checkpoint['model'] is not None: model.load_state_dict(checkpoint['model'])
    criterion = torch.nn.CrossEntropyLoss()
    
    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    if checkpoint['optimizer'] is not None: optimizer.load_state_dict(checkpoint['optimizer'])
    
    # define logging objects
    summary_writer = SummaryWriter(os.path.join('result', 'summary', str(k), 'train'), )
    summary_writer_val = SummaryWriter(os.path.join('result', 'summary', str(k), 'val'), )
    logger = util.logger.LoggerMDGL(5, 2)

    # start training
    for epoch in range(checkpoint['epoch'],40):
        logger.initialize(k)
        dataset.set_fold(k, train=True)
        loss_accumulate = 0.0
        reg_ortho_accumulate = 0.0
        for i, x in enumerate(tqdm(dataloader, ncols=60, desc=f'k:{k} e:{epoch}')):
            # process input data
            dyn_a1, sampling_points1 = util.bold.process_dynamic_fc(x['timeseries1'], 40,3,175) # time series segmentation with sliding windows(AAL Spatial Scale)
            sampling_endpoints1 = [p+40 for p in sampling_points1]
            dyn_v1 =dyn_a1 
            t1 = x['timeseries1'].permute(1,0,2)
            dyn_a2, sampling_points2 = util.bold.process_dynamic_fc(x['timeseries2'],40,3,175) # time series segmentation with sliding windows(CC200 Spatial Scale)
            sampling_endpoints2 = [p+40 for p in sampling_points2]
            dyn_v2 =dyn_a2
            t2 = x['timeseries2'].permute(1,0,2)
            label = x['label']
            
            logit = step(
                model=model,
                criterion=criterion,
                dyn_v1=dyn_v1,
                dyn_a1=dyn_a1,
                sampling_endpoints1=sampling_endpoints1,
                t1=t1,
                dyn_v2=dyn_v2,
                dyn_a2=dyn_a2,
                sampling_endpoints2=sampling_endpoints2,
                t2=t2,
                label=label,
                reg_lambda=0.000001,
                clip_grad=0.0,
                device=device,
                optimizer=optimizer)
    
            pred = logit.argmax(1)
            prob = logit.softmax(1)
            logger.add(k=k, pred=pred.detach().cpu().numpy(), true=label.detach().cpu().numpy(), prob=prob.detach().cpu().numpy())
          
            
        # summarize results
        samples = logger.get(k)
        metrics = logger.evaluate(k)
        print(metrics)

        # save checkpoint
        torch.save({
            'fold': k,
            'epoch': epoch+1,
            'model': model.state_dict(), 
            'optimizer': optimizer.state_dict()},
            os.path.join('result', 'checkpoint.pth'))

    # finalize fold
    torch.save(model.state_dict(), os.path.join('result', 'model', str(k), 'model.pth'))
    checkpoint.update({'epoch': 0, 'model': None, 'optimizer': None})

os.remove(os.path.join('result', 'checkpoint.pth')) # delete a file in the specified path



#### test

os.makedirs(os.path.join('result', 'attention'), exist_ok=True)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# define dataset
dataset = DatasetASD(k_fold=5)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)
logger = util.logger.LoggerMDGL(5, 2)

for k in range(5):

    os.makedirs(os.path.join('result', 'attention', str(k)), exist_ok=True)
    model = Fusion(
        input_dim1=116, # number of brain regions at AAL spatial scale
        input_dim2=200, # number of brain regions at CC200 spatial scale
        hidden_dim=64,
        num_classes=2,
        num_heads=1,
        num_layers=2,
        sparsity=30,
        dropout=0.5,
        cls_token='sum',
        readout='sero')
    model.to(device)
    model.load_state_dict(torch.load(os.path.join('result', 'model', str(k), 'model.pth')))
    criterion = torch.nn.CrossEntropyLoss()

    # define logging objects
    fold_attention = {'node_attention': [], 'time_attention': []}
    summary_writer = SummaryWriter(os.path.join('result', 'summary', str(k), 'test'))

    logger.initialize(k)
    dataset.set_fold(k, train=False)
    loss_accumulate = 0.0
    reg_ortho_accumulate = 0.0
    latent_accumulate = []
    for i, x in enumerate(tqdm(dataloader, ncols=60, desc=f'k:{k}')):
        with torch.no_grad():
            # process input data
            dyn_a1, sampling_points1 = util.bold.process_dynamic_fc(x['timeseries1'],40,3,175) # time series segmentation with sliding windows(AAL Spatial Scale)
            sampling_endpoints1 = [p+40 for p in sampling_points1]
            dyn_v1 =dyn_a1
            t1 = x['timeseries1'].permute(1,0,2)
            dyn_a2, sampling_points2 = util.bold.process_dynamic_fc(x['timeseries2'],40,3,175) # time series segmentation with sliding windows(CC200 Spatial Scale)
            sampling_endpoints2 = [p+40 for p in sampling_points2]
            dyn_v2 =dyn_a2
            t2 = x['timeseries2'].permute(1,0,2)
            label = x['label']
            
            logit = step(
                model=model,
                criterion=criterion,
                dyn_v1=dyn_v1,
                dyn_a1=dyn_a1,
                sampling_endpoints1=sampling_endpoints1,
                t1=t1,
                dyn_v2=dyn_v2,
                dyn_a2=dyn_a2,
                sampling_endpoints2=sampling_endpoints2,
                t2=t2,
                label=label,
                reg_lambda=0.000001,
                clip_grad=0.0,
                device=device,
                optimizer=None)
            pred = logit.argmax(1)
            prob = logit.softmax(1)
            logger.add(k=k, pred=pred.detach().cpu().numpy(), true=label.detach().cpu().numpy(), prob=prob.detach().cpu().numpy())


    # summarize results
    samples = logger.get(k)
    metrics = logger.evaluate(k)
    print(metrics)


# finalize experiment
logger.to_csv('result')
final_metrics = logger.evaluate() 
print(final_metrics)
summary_writer.close()
torch.save(logger.get(), os.path.join('result', 'samples.pkl')) 