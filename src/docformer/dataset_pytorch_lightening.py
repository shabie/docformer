import os
import pickle

import numpy as np
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

# pathToPickleFile = 'RVL-CDIP-PickleFiles/'
# entries = os.listdir(pathToPickleFile)

"""## Base Dataset"""

device = "cuda" if torch.cuda.is_available() else "cpu"


class RVLCDIPDatset(Dataset):
    def __init__(self, entries, pathToPickleFile,colab= False):
        self.pathToPickleFile = pathToPickleFile
        self.entries = entries
        self.colab = colab

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        imageName = self.entries[index]
        encoding = os.path.join(self.pathToPickleFile, imageName)

        with open(encoding, "rb") as sample:
            encoding = pickle.load(sample)
        if not self.colab:  # Using for the pretraining task
            del encoding['category_labels']
            del encoding['numeric_labels']
            del encoding['target_bbox']
            del encoding['resized_and_aligned_target_bbox']
        for i in list(encoding.keys()):
            encoding[i] = encoding[i].to(device)
            
        encoding['x_features'][:,3:] = torch.clamp(encoding['x_features'][:,3:],-1024,1024)
        encoding['x_features'][:,3:] +=1024
        encoding['y_features'][:,3:]  = torch.clamp(encoding['y_features'][:,3:],-1024,1024)
        encoding['y_features'][:,3:] +=1024
        ## The image stored in the pickle file is not nromalized, so normalizing it
        return encoding


# train_entries,val_entries = tts(entries,test_size = 0.2)
# train_dataset = RVLCDIPDatset(train_entries,pathToPickleFile)
# val_dataset = RVLCDIPDatset(val_entries,pathToPickleFile)
