import math
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pickle
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
import pickle
import pytorch_lightning as pl
from transformers import AutoTokenizer,AutoModel
from sklearn.model_selection import train_test_split as tts
import warnings

# pathToPickleFile = 'RVL-CDIP-PickleFiles/'
# entries = os.listdir(pathToPickleFile)

"""## Base Dataset"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'
class RVLCDIPDatset(Dataset):
    def __init__(
        self,
        entries,pathToPickleFile
        ):
        self.pathToPickleFile = pathToPickleFile
        self.entries = entries
 
    def __len__(self):
      return len(self.entries)

    def __getitem__(self, index):
        imageName = self.entries[index]
        encoding = os.path.join(self.pathToPickleFile,imageName)
        
        with open(encoding,"rb") as sample:
          encoding = pickle.load(sample)
        for i in list(encoding.keys()):
            encoding[i] = encoding[i].to(device)
        return encoding

# train_entries,val_entries = tts(entries,test_size = 0.2)
# train_dataset = RVLCDIPDatset(train_entries,pathToPickleFile)
# val_dataset = RVLCDIPDatset(val_entries,pathToPickleFile)

"""## DataModules"""

class DataModule(pl.LightningDataModule):
      def __init__(self,train_entries,val_entries,pathToPickleFile,batch_size = 4):
          super().__init__()
          self.batch_size = batch_size
          self.train_entries = train_entries
          self.val_entries = val_entries

      def setup(self,stage = None):
        self.train_dataset = RVLCDIPDatset(self.train_entries,pathToPickleFile)
        self.val_dataset  =  RVLCDIPDatset(self.val_entries,pathToPickleFile)

      def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset,batch_size = self.batch_size,shuffle=True)

      def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset,batch_size = self.batch_size,shuffle=False)
