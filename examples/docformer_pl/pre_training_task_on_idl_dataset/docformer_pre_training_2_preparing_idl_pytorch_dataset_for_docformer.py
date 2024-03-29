# -*- coding: utf-8 -*-
"""DocFormer Pre-training : 2. Preparing IDL PyTorch Dataset for DocFormer

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dDMABz_EunCdg3S1neJ5fMPkRSmTcAfd
"""

## Refer here for the dataset: https://github.com/furkanbiten/idl_data 
# (IDL dataset was also used in the pre-training of LaTr), might take time to download the dataset

# !wget http://datasets.cvc.uab.es/UCSF_IDL/Samples/ocr_imgs_sample.zip
# !unzip /content/ocr_imgs_sample.zip
# !rm /content/ocr_imgs_sample.zip

# Commented out IPython magic to ensure Python compatibility.
# ## Installing the dependencies (might take some time)
# 
# %%capture
# !pip install pytesseract
# !sudo apt install tesseract-ocr
# !pip install transformers
# !pip install pytorch-lightning
# !pip install einops
# !pip install tqdm
# !pip install 'Pillow==7.1.2'
# !pip install PyPDF2

## Cloning the repository
# !git clone https://github.com/uakarsh/docformer.git

## Getting the JSON File
import json

## For reading the PDFs
from PyPDF2 import PdfReader
import io
from PIL import Image, ImageDraw

## A bit of code taken from here : https://www.kaggle.com/code/akarshu121/docformer-for-token-classification-on-funsd/notebook
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

## PyTorch Libraries
import torch
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn

## Adding the path of docformer to system path
import sys
sys.path.append('./docformer/src/docformer/')

## Importing the functions from the DocFormer Repo
from dataset import resize_align_bbox, get_centroid, get_pad_token_id_start_index, get_relative_distance
from modeling import DocFormerEncoder,ResNetFeatureExtractor,DocFormerEmbeddings,LanguageFeatureExtractor

## Transformer librarues
from transformers import BertTokenizerFast

## PyTorch Lightning library
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

pdf_path = "./sample/pdfs"
ocr_path = "./sample/OCR"

## Image property

resize_scale = (500, 500)

from typing import List

def normalize_box(box : List[int], width : int, height : int, size : tuple = resize_scale):
    """
    Takes a bounding box and normalizes it to a thousand pixels. If you notice it is
    just like calculating percentage except takes 1000 instead of 100.
    """
    return [
        int(size[0] * (box[0] / width)),
        int(size[1] * (box[1] / height)),
        int(size[0] * (box[2] / width)),
        int(size[1] * (box[3] / height)),
    ]

## Function to get the images from the PDFs as well as the OCRs for the corresponding images

def get_image_ocrs_from_path(pdf_file_path : str, ocr_file_path : str, resize_scale = resize_scale):

  ## Getting the image list, since the pdfs can contain many image
  reader = PdfReader(pdf_file_path)
  img_list = []
  for i in range(len(reader.pages)):
    page = reader.pages[i]
    for image_file_object in page.images:

      stream = io.BytesIO(image_file_object.data)
      img = Image.open(stream).convert("RGB").resize(resize_scale)
      img_list.append(img)

  json_entry = json.load(open(ocr_file_path))[1]
  json_entry =[x for x in json_entry["Blocks"] if "Text" in x]

  pages = [x["Page"] for x in json_entry]
  ocrs = {pg : [] for pg in set(pages)}

  for entry in json_entry:
    bbox = entry["Geometry"]["BoundingBox"]
    x, y, w, h = bbox['Left'], bbox['Top'], bbox["Width"], bbox["Height"]
    bbox = [x, y, x + w, y + h]
    bbox = normalize_box(bbox, width = 1, height = 1, size = resize_scale)
    ocrs[entry["Page"]].append({"word" : entry["Text"], "bbox" : bbox})

  return img_list, ocrs

# sample_pdf_folder = os.path.join(pdf_path, sorted(os.listdir(pdf_path))[0])
# sample_ocr_folder = os.path.join(ocr_path, sorted(os.listdir(ocr_path))[0])

# sample_pdf = os.path.join(sample_pdf_folder, sample_pdf_folder.split("/")[-1] + ".pdf")
# sample_ocr = os.path.join(sample_ocr_folder, os.listdir(sample_ocr_folder)[0])

# img_list, ocrs = get_image_ocrs_from_path(sample_pdf, sample_ocr)

"""## Preparing the Pytorch Dataset"""

from tqdm.auto import tqdm

img_list = []
ocr_list = []

pdf_files = sorted(os.listdir(pdf_path))[:30] ## Using only 30 since, google session gets crashed
ocr_files = sorted(os.listdir(ocr_path))[:30] 

for pdf, ocr in tqdm(zip(pdf_files, ocr_files), total = len(pdf_files)):
  pdf = os.path.join(pdf_path, pdf, pdf + '.pdf')
  ocr = os.path.join(ocr_path, ocr)
  ocr = os.path.join(ocr, os.listdir(ocr)[0])
  img, ocrs = get_image_ocrs_from_path(pdf, ocr)

  for i in range(len(img)):
    img_list.append(img[i])
    ocr_list.append(ocrs[i+1])  ## Pages are 1, 2, 3 hence 0 + 1, 1 + 1, 2 + 1

"""## Visualizing the OCRs"""

index = 17
curr_img = img_list[index]
curr_ocr = ocr_list[index]

# create rectangle image
draw_on_img = ImageDraw.Draw(curr_img)  

for it in curr_ocr:
  box = it["bbox"]
  draw_on_img.rectangle(box, outline ="violet")

curr_img

"""## Creating features for DocFormer"""

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def get_tokens_with_boxes(unnormalized_word_boxes, word_ids,max_seq_len = 512, pad_token_box = [0, 0, 0, 0]):

    # assert len(unnormalized_word_boxes) == len(word_ids), this should not be applied, since word_ids may have higher 
    # length and the bbox corresponding to them may not exist

    unnormalized_token_boxes = []
    
    i = 0
    for word_idx in word_ids:
        if word_idx is None:
            break
        unnormalized_token_boxes.append(unnormalized_word_boxes[word_idx])
        i+=1

    # all remaining are padding tokens so why add them in a loop one by one
    num_pad_tokens = len(word_ids) - i - 1
    if num_pad_tokens > 0:
        unnormalized_token_boxes.extend([pad_token_box] * num_pad_tokens)


    if len(unnormalized_token_boxes)<max_seq_len:
        unnormalized_token_boxes.extend([pad_token_box] * (max_seq_len-len(unnormalized_token_boxes)))

    return unnormalized_token_boxes

def create_features_for_cls(image,
        tokenizer = tokenizer,
        target_size=(500,384),  # This was the resolution used by the authors
        max_seq_length=512,
        bounding_box = None,
        words = None):
    
    
    CLS_TOKEN_BOX = [0, 0, *resize_scale]    # Can be variable, but as per the paper, they have mentioned that it covers the whole image
    # step 2: resize image
    resized_image = image.resize(target_size)
    
    # step 4: tokenize words and get their bounding boxes (one word may split into multiple tokens)
    encoding = tokenizer(words,
                         padding="max_length",
                         max_length=max_seq_length,
                         is_split_into_words=True,
                         truncation=True,
                         add_special_tokens=False)
    
    unnormalized_token_boxes = get_tokens_with_boxes(unnormalized_word_boxes = bounding_box,
                                                     word_ids = encoding.word_ids())
    
    # step 5: add special tokens and truncate seq. to maximum length
    unnormalized_token_boxes = [CLS_TOKEN_BOX] + unnormalized_token_boxes[:-1]
    # add CLS token manually to avoid autom. addition of SEP too (as in the paper)
    encoding["input_ids"] = [tokenizer.cls_token_id] + encoding["input_ids"][:-1]
    
    # step 6: Add bounding boxes to the encoding dict
    encoding["unnormalized_token_boxes"] = unnormalized_token_boxes
    
    # step 8: normalize the image
    encoding["resized_scaled_img"] = ToTensor()(resized_image)
    
    # step 10: rescale and align the bounding boxes to match the resized image size (typically 224x224)
    resized_and_aligned_bboxes = []

    for bbox in unnormalized_token_boxes:
        # performing the normalization of the bounding box
        resized_and_aligned_bboxes.append(resize_align_bbox(tuple(bbox), *resize_scale, *target_size)) ## The bbox are resized to (500, 500)

    encoding["resized_and_aligned_bounding_boxes"] = resized_and_aligned_bboxes

    # step 11: add the relative distances in the normalized grid
    bboxes_centroids = get_centroid(resized_and_aligned_bboxes)
    pad_token_start_index = get_pad_token_id_start_index(words, encoding, tokenizer)
    a_rel_x, a_rel_y = get_relative_distance(resized_and_aligned_bboxes, bboxes_centroids, pad_token_start_index)

    # step 12: convert all to tensors
    for k, v in encoding.items():
        encoding[k] = torch.as_tensor(encoding[k])

    encoding.update({
        "x_features": torch.tensor(a_rel_x).int(),
        "y_features": torch.tensor(a_rel_y).int(),
        })

    
    # step 16: keys to keep, resized_and_aligned_bounding_boxes have been added for the purpose to test if the bounding boxes are drawn correctly or not, it maybe removed
    keys = ['resized_scaled_img', 'x_features','y_features','input_ids','resized_and_aligned_bounding_boxes']

    final_encoding = {k:encoding[k] for k in keys}

    del encoding
    return final_encoding

"""## PyTorch Dataset"""

class MyDataset(Dataset):

  def __init__(self, img_list, ocr_list):
    self.img_list = img_list
    self.ocr_list = ocr_list

  def __len__(self):
    return len(self.img_list)

  def __getitem__(self, idx):

    curr_img = self.img_list[idx]
    curr_ocr = self.ocr_list[idx]

    words = [x['word'] for x in curr_ocr]
    bbox = [x["bbox"] for x in curr_ocr]

    encoding = create_features_for_cls(
    image = curr_img,
    bounding_box = bbox,
    words = words)   

    return encoding

"""## MLM from Transformer"""

## https://github.com/huggingface/transformers/blob/d316037ad71f8748aac9045ffd96970826456a04/src/transformers/data/data_collator.py#L750

def mask_tokens(inputs, mask_token_id = 103, special_tokens_mask = None):

        """
        FOR A BATCH
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        labels = inputs.clone()

        probability_matrix = torch.full(labels.shape, 0.15)
        if special_tokens_mask is None:
            special_tokens_mask = [
                tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

mlm_function = mask_tokens

def collate_fn(data_bunch):
  final_dict = {}
  for item in data_bunch:
    for key, value in item.items():
      if key not in final_dict:
        final_dict[key] = []
      final_dict[key].append(value)

  for key in list(final_dict.keys()):
    final_dict[key] = torch.stack(final_dict[key], axis = 0)

  inputs, labels = mlm_function(final_dict["input_ids"])
  final_dict["input_ids"] = inputs
  final_dict["labels"] = labels
  return final_dict

ds = MyDataset(img_list, ocr_list)
dl = DataLoader(ds, batch_size = 2, collate_fn = collate_fn)

