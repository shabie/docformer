# -*- coding: utf-8 -*-


# from google.colab import drive
# drive.mount('/content/drive',force_remount=True)
# ! pip install -q albumentations==0.4.6
# ! sudo apt install tesseract-ocr
# ! pip -q install pytesseract
# ! pip install -q transformers
# ! pip install albumentations==0.4.6

import pytesseract
import os
import numpy as np
import pandas as pd
from datetime import datetime
import time
import random
from tqdm.auto import tqdm

## Libraries
import os
import transformers
from PIL import Image
import pytesseract
import numpy as np
import torch
from torchvision.transforms import ToTensor
import pickle

import warnings
warnings.simplefilter(action='ignore',category = FutureWarning)

#Torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader

#sklearn
from sklearn.model_selection import StratifiedKFold

#CV
import cv2
from PIL import *

#Albumenatations
import albumentations as A
import matplotlib.pyplot as plt
from albumentations.pytorch.transforms import ToTensorV2

#Glob
from glob import glob

import sys
sys.path.append('docformer/src/docformer/')
from dataset import *    # Includes all the utility functions

seed = 42
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(seed)
marking = pd.read_csv('new_publaynet.csv')     # A CSV file, which contains the bbox for each image and its category
bboxs = np.stack(marking['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))
for i, column in enumerate(['x1', 'y1', 'x2', 'y2']):
   marking[column] = bboxs[:,i]

marking.drop(columns=['bbox'], inplace=True)
marking = marking[(marking['height']>0) & (marking['width']>0) & (marking['x1']>0) & (marking['y1']>0)]

actual_image = []
shapes = []
count=0

for j,i in tqdm(enumerate(marking['image_id'].values)):
    if os.path.exists(i):
     actual_image.append(i)
    else:
     count+=1
print("Total non existing images:", count)
marking = marking[marking['image_id'].isin(actual_image)]

image_data = marking.groupby('image_id')
images = marking['image_id'].unique()

def get_data(img_id):
    if img_id not in image_data.groups:
        return dict(image_id=img_id, source='', boxes=list(),labels = ())
    data  = image_data.get_group(img_id)
    boxes = data[['x1','y1','x2','y2']].values
    labels = data['category_id'].values
    return dict(image_id = img_id, boxes = boxes,labels = labels)
print("Getting the image list")
image_list = [get_data(img_id) for img_id in tqdm(images)]

print(f'total number of images: {len(image_list)}, images with bboxes: {len(image_data)}')
null_images=[x['image_id'] for x in image_list if len(x['boxes'])==0]
len(null_images)

# Define color code
colors = {'title': (255, 0, 0),
          'text': (0, 255, 0),
          'figure': (0, 0, 255),
          'table': (255, 255, 0),
          'list': (0, 255, 255)}
labels = list(colors.keys())
idx2label = {v: k for v, k in enumerate(labels)}
label2idx = {k: v for v, k in enumerate(labels)}

import math

def createPickelFile(image,pathToSave,tokenizer=None,target_size = 224,max_seq_length = 512,pretrain = True):
    
        if not os.path.exists(pathToSave):
          os.mkdir(pathToSave)

        if tokenizer==None:
            from transformers import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        pad_token_box = [0, 0, 0, 0]
        original_image = Image.open(image['image_id']).convert("RGB")
        entries = apply_ocr(image['image_id'])
        resized_image = original_image.resize((target_size,target_size))
        unnormalized_word_boxes = entries['bbox']
        words = entries['words']

        width, height = original_image.size
        normalized_word_boxes = [normalize_box(bbox, width, height) for bbox in unnormalized_word_boxes]
        assert len(words) == len(normalized_word_boxes)

        token_boxes, unnormalized_token_boxes,final_word_tokens = get_tokens(tokenizer,words, unnormalized_word_boxes, normalized_word_boxes)

        # Truncation of token_boxes + token_labels
        special_tokens_count = 2
        if len(token_boxes) > max_seq_length - special_tokens_count:
            token_boxes = token_boxes[: (max_seq_length - special_tokens_count)]
            unnormalized_token_boxes = unnormalized_token_boxes[: (max_seq_length - special_tokens_count)]

        # add bounding boxes and labels of cls + sep tokens
        token_boxes = [[0, 0, 0, 0]] + token_boxes + [[0,0, 0, 0]]
        unnormalized_token_boxes = [[0, 0, 0, 0]] + unnormalized_token_boxes + [[0, 0, 0, 0]]

        encoding = tokenizer(' '.join(words), padding='max_length', truncation=True)

        # Padding of token_boxes up the bounding boxes to the sequence length.
        input_ids = tokenizer(' '.join(words), truncation=True)["input_ids"]
        assert len(input_ids) == len(token_boxes)  # check if number of tokens match
        padding_length = max_seq_length - len(input_ids)
        token_boxes += [pad_token_box] * padding_length
        unnormalized_token_boxes += [pad_token_box] * padding_length
        encoding['bbox'] = token_boxes


        encoding['mlm_labels'] = encoding['input_ids']
        assert len(encoding['mlm_labels']) == max_seq_length

        encoding['category_labels'] = image['labels']
        encoding['numeric_labels'] = [label2idx[x] for x in image['labels']]
        assert len(encoding['input_ids']) == max_seq_length
        assert len(encoding['attention_mask']) == max_seq_length
        assert len(encoding['token_type_ids']) == max_seq_length
        assert len(encoding['bbox']) == max_seq_length

        encoding['resized_image'] = ToTensor()(resized_image)

        ## Applying mask for the sake of pre-training
        encoding['input_ids'] = apply_mask(encoding['input_ids'])


        # rescale and align the bounding boxes to match the resized image size (typically 224x224)
        encoding['resized_and_aligned_bounding_boxes_for_words'] = [resize_and_align_bounding_box(bbox, original_image, target_size)
                                                          for bbox in unnormalized_token_boxes]
        encoding['resized_and_aligned_target_bbox'] = [resize_and_align_bounding_box(bbox, original_image, target_size)
                                                          for bbox in image['boxes']]

        index = -1
        # getting the index of the last word (using the input id)
        for i in range(len(encoding['input_ids']) - 1):
          if encoding['input_ids'][i + 1] == 0:
            index = i
            break

        # adding the relative position as well
        actual_bbox = encoding['resized_and_aligned_bounding_boxes_for_words']

        # Calculating the centroid
        centroid = get_centroid(actual_bbox)

        a_rel_x, a_rel_y = get_relative_distance(actual_bbox, centroid, index)
        encoding['unnormalized_token_boxes'] = unnormalized_token_boxes

        encoding['input_ids']*=(encoding['input_ids']!=102)    # Remvoing the SEP token
        encoding['mlm_labels']*=(encoding['mlm_labels']!=102)
        encoding['target_bbox'] = image['boxes']

        # finally, convert everything to PyTorch tensors
        for k, v in encoding.items():
          try:
              encoding[k] = torch.as_tensor(encoding[k])
          except:
            continue

        encoding.update({"x_features": torch.as_tensor(a_rel_x), "y_features": torch.as_tensor(a_rel_y)})
        imageName = image['image_id'].split('/')[-1]
        with open(f'{pathToSave}{imageName}.pickle', 'wb') as f:
              pickle.dump(encoding, f)
        return encoding

print("Saving the pickle files")
pathToSave = ''

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
encoding = [createPickelFile(x,pathToSave,tokenizer) for x in tqdm(image_list)]


print("Executed Successfully!!!")