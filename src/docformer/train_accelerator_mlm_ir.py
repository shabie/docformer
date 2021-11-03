# -*- coding: utf-8 -*-



## Dependencies

import pickle
import os
from accelerate import Accelerator
import accelerate
import pytesseract
import torchmetrics
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import json
import numpy as np
from tqdm.auto import tqdm
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
from einops import rearrange
from einops import rearrange as rearr
from PIL import Image
from sklearn.model_selection import train_test_split as tts
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from modeling import *

batch_size = 9


weights = {'mlm':5,'ir':1,'tdi':5}

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        
    @property
    def avg(self):
        return (self.sum / self.count) if self.count>0 else 0

## Loggers
class Logger:
    def __init__(self, filename, format='csv'):
        self.filename = filename + '.' + format
        self._log = []
        self.format = format

    def save(self, log, epoch=None):
        log['epoch'] = epoch + 1
        self._log.append(log)
        if self.format == 'json':
            with open(self.filename, 'w') as f:
                json.dump(self._log, f)
        else:
            pd.DataFrame(self._log).to_csv(self.filename, index=False)



def train_fn(data_loader, model, criterion1,criterion2, optimizer, epoch, device, scheduler=None,weights=weights):
    model.train()
    accelerator = Accelerator()
    model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)
    loop = tqdm(data_loader, leave=True)
    log = None
    train_acc = torchmetrics.Accuracy()
    loop = tqdm(data_loader)
    
    for batch in loop:

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels1 = batch["mlm_labels"].to(device)
        labels2 = batch['resized_image'].to(device)

        # process
        outputs = model(batch)
        ce_loss = criterion1(outputs['mlm_labels'].transpose(1,2), labels1)
        ir_loss = criterion2(outputs['ir'],labels2)

        if log is None:
            log = {}
            log["ce_loss"] = AverageMeter()
            log['accuracy'] = AverageMeter()
            log['ir_loss'] = AverageMeter()
            log['total_loss'] = AverageMeter()

        total_loss = weights['mlm']*ce_loss + weights['ir']*ir_loss
        optimizer.zero_grad()
        accelerator.backward(total_loss)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        log['accuracy'].update(train_acc(labels1.cpu(),torch.argmax(outputs['mlm_labels'],-1).cpu()).item(),batch_size)
        log['ce_loss'].update(ce_loss.item(),batch_size)
        log['ir_loss'].update(ir_loss.item(),batch_size)
        log['total_loss'].update(total_loss.item(),batch_size)
        loop.set_postfix({k: v.avg for k, v in log.items()})

    return log


# Function for the validation data loader
def eval_fn(data_loader, model, criterion1,criterion2, device,weights=weights):
    model.eval()
    log = None
    val_acc = torchmetrics.Accuracy()       
    
    
    with torch.no_grad():
        loop = tqdm(data_loader, total=len(data_loader), leave=True)
        for batch in loop:

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels1 = batch["mlm_labels"].to(device)
            labels2 = batch['resized_image'].to(device)


            output = model(batch)
            ce_loss = criterion1(output['mlm_labels'].transpose(1,2), labels1)
            ir_loss = criterion2(output['ir'],labels2)
            total_loss = weights['mlm']*ce_loss + weights['ir']*ir_loss
            if log is None:
                log = {}
                log["ce_loss"] = AverageMeter()
                log['accuracy'] = AverageMeter()
                log['ir_loss'] = AverageMeter()
                log['total_loss'] = AverageMeter()

            log['accuracy'].update(val_acc(labels1.cpu(),torch.argmax(output['mlm_labels'],-1).cpu()).item(),batch_size)
            log['ce_loss'].update(ce_loss.item(),batch_size)
            log['ir_loss'].update(ir_loss.item(),batch_size)
            log['total_loss'].update(total_loss.item(),batch_size)
            loop.set_postfix({k: v.avg for k, v in log.items()})

    return log  # ['total_loss']

date = '26Oct'


def run(config,train_dataloader,val_dataloader,device,epochs,path,classes,lr = 5e-5,weights=weights):
    logger = Logger(f"{path}/logs")
    model = DocFormer_For_IR(config,classes).to(device)
    criterion1 = nn.CrossEntropyLoss().to(device)
    criterion2 =  torch.nn.L1Loss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    best_val_loss = 1e9
    header_printed = False
    batch_size = config['batch_size']
    for epoch in range(epochs):
        print("Training the model.....")
        train_log = train_fn(
            train_dataloader, model, criterion1,criterion2, optimizer, epoch, device, scheduler=None
        )
        
        print("Validating the model.....")
        valid_log = eval_fn(val_dataloader, model, criterion1,criterion2, device)
        log = {k: v.avg for k, v in train_log.items()}
        log.update({"V/" + k: v.avg for k, v in valid_log.items()})
        logger.save(log, epoch)
        keys = sorted(log.keys())
        if not header_printed:
            print(" ".join(map(lambda k: f"{k[:8]:8}", keys)))
            header_printed = True
        print(" ".join(map(lambda k: f"{log[k]:8.3f}"[:8], keys)))
        if log["V/total_loss"] < best_val_loss:
            best_val_loss = log["V/total_loss"]
            print("Best model found at epoch {}".format(epoch + 1))
            torch.save(model.state_dict(), f"{path}/docformer_best_{epoch}_{date}.pth")
