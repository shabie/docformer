# -*- coding: utf-8 -*-

import math
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
from sklearn.model_selection import train_test_split as tts
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from transformers import AutoModel, AutoTokenizer
from modelling import *

"""## Base Dataset"""

device = "cuda" if torch.cuda.is_available() else "cpu"

"""## Base Model"""

config = {
    "coordinate_size": 96,
    "hidden_dropout_prob": 0.1,
    "hidden_size": 768,
    "image_feature_pool_shape": [7, 7, 256],
    "intermediate_ff_size_factor": 3,  # default ought to be 4
    "max_2d_position_embeddings": 1024,
    "max_position_embeddings": 512,
    "max_relative_positions": 8,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "pad_token_id": 0,
    "shape_size": 96,
    "vocab_size": 30522,
    "layer_norm_eps": 1e-12,
    "batch_size":9
}


class Model(pl.LightningModule):
    
  def __init__(self,config,num_classes,lr = 5e-5):
    
    super().__init__()
    self.save_hyperparameters()
    self.docformer = DocFormerForClassification(config,num_classes)

  def forward(self,x):
    return self.docformer(x)

  def training_step(self,batch,batch_idx):
        
    # For the purpose of pretraining, there could be multiple target outputs, so therefore we need to add additional loss function, as for an image_fp, if the MLM + IR is to be done
    # then, there could be a dictionary as an output, and then we need to define two criterion as CrossEntropy and L1 loss, and add the weighted sum of them as the total loss
    # and proceed forward, and for the whole process, only the final head of the DocFormer encoder needs to be changed, and thats it
    
    
    # Currently, we are performing only the MLM Part
    logits = self.forward(batch)
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(logits.transpose(2,1), batch["mlm_labels"].long())
    self.log("train_loss",loss,prog_bar = True)

  def validation_step(self, batch, batch_idx):
    
    logits = self.forward(batch)
    b,size,classes = logits.shape
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(logits.transpose(2,1), batch["mlm_labels"].long())
    val_acc = 100*(torch.argmax(logits,dim = -1)==batch["mlm_labels"].long()).float().sum()/(logits.shape[0]*logits.shape[1])
    val_acc = torch.tensor(val_acc)
    self.log("val_loss", loss, prog_bar=True)
    self.log("val_acc", val_acc, prog_bar=True)

  def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams["lr"])

"""## Examples"""

# pathToPickleFile = 'RVL-CDIP-PickleFiles/'
# entries = os.listdir(pathToPickleFile)
# data = DataModule(train_entries,val_entries,pathToPickleFile)
# model = Model(config,num_classes= 30522).to(device)
# trainer = pl.Trainer(gpus=(1 if torch.cuda.is_available() else 0),
#     max_epochs=10,
#     fast_dev_run=False,
#     logger=pl.loggers.TensorBoardLogger("logs/", name="rvl-cdip", version=1),)
# trainer.fit(model,data)



