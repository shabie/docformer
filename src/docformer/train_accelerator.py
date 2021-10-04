## Dependencies

from accelerate import Accelerator
import pytesseract
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import numpy as np
from torchvision.transforms import ToTensor
import torch.nn.functional as F

import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from collections import OrderedDict
import torch.nn.functional as F
from einops import rearrange
from einops import rearrange as rearr

from sklearn.model_selection import train_test_split as tts
import warnings
warnings.filterwarnings("ignore")

# Integrating with Hugging face Accelerator

accelerator = Accelerator()


## Loggers
class Logger:
    def __init__(self,filename,format='csv'):
        self.filename = filename + '.' + format
        self._log = []
        self.format = format
    def save(self,log,epoch=None):
        log['epoch'] = epoch+1
        self._log.append(log)
        if self.format == 'json':
            with open(self.filename,'w') as f:
                json.dump(self._log,f)
        else:
            pd.DataFrame(self._log).to_csv(self.filename,index=False)

# Function for the training data loader
def train_fn(data_loader,model,criterion,optimizer,epoch,device,scheduler = None):
    model.train()
    model,optimizer,data_loader = accelerator.prepare(model,optimizer,data_loader)
    loop = tqdm(data_loader, leave=True)
    log = None
    for batch in loop:
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # process
        outputs = model(batch)
        ce_loss = criterion(outputs,labels)
        
        if log is None:
            log = {'ce_loss':ce_loss}
            log['total_loss'] = AverageMeter()
        
        total_loss = ce_loss
        optimizer.zero_grad()
        accelerate.backward(total_loss)
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        log['total_loss'].update(total_loss.item(),batch_size)
        loop.set_postfix({k:v.avg for k,v in log.items()}) 
        
    return log

# Function for the validation data loader
def eval_fn(data_loader, model,criterion, device):
    model.eval()
    log = None
    with torch.no_grad():
        loop = tqdm(data_loader, total=len(data_loader),leave=True)
        for batch in loop:
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            output = model(batch)
            ce_loss = criterion(output, labels)
        
            if log is None:
                log = {'ce_loss':ce_loss}
                log['total_loss'] = AverageMeter()
            
            for k,v in loss_dict.items():
                log[k].update(v.item(),batch_size)
        
            total_loss = ce_loss
            log['total_loss'].update(total_loss.item(),batch_size)
            loop.set_postfix({k:v.avg for k,v in log.items()}) 
    return log #['total_loss']

## Combining everything
date = ''
def run(model,train_dataloader,valid_dataloader,device,epochs,path,lr = 5e-5):
    logger = Logger(f'{path}/logs')
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr = lr)
    best_val_loss = 1e9
    header_printed = False
    for epoch in range(epochs):
        train_log = train_fn(train_dataloader, model,criterion, optimizer,epoch,device,scheduler=None)
        valid_log = eval_fn(valid_dataloader, model,criterion, device)
        log = {k:v.avg for k,v in train_log.items()}
        log.update({'V/'+k:v.avg for k,v in valid_log.items()})
        logger.save(log,epoch)
        keys = sorted(log.keys())
        if not header_printed:
            print(' '.join(map(lambda k: f'{k[:8]:8}',keys)))
            header_printed = True
        print(' '.join(map(lambda k: f'{log[k]:8.3f}'[:8],keys)))
        if log['V/total_loss'] > best_val_loss:
            best_val_loss = log['V/total_loss']
            print('Best model found at epoch {}'.format(epoch+1))
            torch.save(model.state_dict(), f'{path}/docformer_best_{epoch}_{date}.pth')
