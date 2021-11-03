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

"""## Base Dataset"""

device = "cuda" if torch.cuda.is_available() else "cpu"

class DocumentDatset(Dataset):
    def __init__(self, entries, pathToPickleFile,pretrain= True):
        self.pathToPickleFile = pathToPickleFile
        self.entries = entries
        self.pretrain = pretrain

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        
        imageName = self.entries[index]
        encoding = os.path.join(self.pathToPickleFile, imageName)

        with open(encoding, "rb") as sample:
            encoding = pickle.load(sample)
            
        if self.pretrain:  
            
            # If the model is used for the purpose of pretraining, then there is no need for the other entries, since there would be some errors, while training
            
            del encoding['category_labels']                   # Error would be created, because category label cannot be stored in the pytorch tensor
            del encoding['numeric_labels']                    # Removed it, but this can be used for the purpose of the segmenting (as for an example, in the FUNSD Dataset)
            del encoding['target_bbox']                       # For the purpose of segmenting the different text in the image
            del encoding['resized_and_aligned_target_bbox']   # Resized version of the above bounding box, for 224x224 image
            
        for i in list(encoding.keys()):
            encoding[i] = encoding[i].to(device)
            
        # Since, we had taken the absolute value of the relative distance, we don't need to add any offset, and hence we can proceed with the model training
        return encoding


# pathToPickleFile = 'RVL-CDIP-PickleFiles/'
# entries = os.listdir(pathToPickleFile)
# train_entries,val_entries = tts(entries,test_size = 0.2)
# train_dataset = DocumentDatset(train_entries,pathToPickleFile)
# val_dataset = DocumentDatset(val_entries,pathToPickleFile)
=======
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

device = "cuda"


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
        if self.colab:  # Using for the pretraining task
            if 'category_labels' in encoding.keys():
                del encoding['category_labels']
            if 'numeric_labels' in encoding.keys():
                del encoding['numeric_labels']
            if 'target_bbox' in encoding.keys():
                del encoding['target_bbox']
            if 'resized_and_aligned_target_bbox' in encoding.keys():
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

