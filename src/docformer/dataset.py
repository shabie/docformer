import os
# !sudo apt install tesseract-ocr
# !pip install -q pytesseract
# !pip install -q transformers
# !pip install -q einops

## Libraries
import os
import transformers
from PIL import Image
import pytesseract
import numpy as np
import torch
from torchvision.transforms import ToTensor
import pickle


pathToSave = ''                                   ## eg: Dataset-PickleFiles/

# Extracting some of the images, for the purpose of demonstrations
from tqdm import tqdm
count = 0;
label2idx = {}
idx2label = []
images = []
labels = []
basePath=""                                     # Path where the image is located
for number, label in tqdm(enumerate(os.listdir(basePath))):
      label2idx[label] = number
      idx2label.append(label)
      for image in os.listdir(basePath+label):
        images.append(basePath+label+'/'+image)                
        labels.append(number)

import math
def relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=1000):

    """
    Adapted from Mesh Tensorflow:
    https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
    Translate relative position to a bucket number for relative attention. The relative position is defined as
    memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
    position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for small
    absolute relative_position and larger buckets for larger absolute relative_positions. All relative positions
    >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket. This should
    allow for more graceful generalization to longer sequences than the model has been trained on.
    Args:
        relative_position: an int32 Tensor
        bidirectional: a boolean - whether the attention is bidirectional
        num_buckets: an integer
        max_distance: an integer
    Returns:
        a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
    """

    ret = 0
    if bidirectional:
        num_buckets //= 2
        ret += (relative_position > 0).long() * num_buckets
        n = torch.abs(relative_position)
    else:
        n = torch.max(-relative_position, torch.zeros_like(relative_position))
    # now n is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = n < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    val_if_large = max_exact + (
        torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
    ).to(torch.int)
    # print("val_if_large:",val_if_large)
    val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
    # print("val_if_large:",val_if_large)
    # print("ret:",ret)
    ret += torch.where(is_small, n, val_if_large)
    # print("ret:",ret)
    return ret

## Utility function
def normalize_box(box, width, height):
        return [
            int(1000 * (box[0] / width)),
            int(1000 * (box[1] / height)),
            int(1000 * (box[2] / width)),
            int(1000 * (box[3] / height)),
        ]
 
def resize_and_align_bounding_box( bbox, original_image, target_size):

      x_, y_ = original_image.size
      x_scale = target_size / x_ 
      y_scale = target_size / y_
      origLeft, origTop, origRight, origBottom = tuple(bbox)
      x = int(np.round(origLeft * x_scale))
      y = int(np.round(origTop * y_scale))
      xmax = int(np.round(origRight * x_scale))
      ymax = int(np.round(origBottom * y_scale)) 
  
      return [x, y , xmax , ymax]

def apply_ocr( example):
            
            # get the image
            image = Image.open(example)
            width, height = image.size
            
            # apply ocr to the image 
            ocr_df = pytesseract.image_to_data(image, output_type='data.frame')
            float_cols = ocr_df.select_dtypes('float').columns
            ocr_df = ocr_df.dropna().reset_index(drop=True)
            ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int)
            ocr_df = ocr_df.replace(r'^\s*$', np.nan, regex=True)
            ocr_df = ocr_df.dropna().reset_index(drop=True)
            ocr_df = ocr_df.sort_values(by = ['left', 'top'])   ## Sorting the values, on the basis of the coordinate space

            # get the words and actual (unnormalized) bounding boxes
            # words = [word for word in ocr_df.text if str(word) != 'nan'])

            words = list(ocr_df.text)
            words = [str(w) for w in words]
            coordinates = ocr_df[['left', 'top', 'width', 'height']]
            actual_boxes = []
            for idx, row in coordinates.iterrows():
                x, y, w, h = tuple(row)           # the row comes in (left, top, width, height) format
                actual_box = [x, y, x + w, y + h] # we turn it into (left, top, left+width, top+height) to get the actual box 
                actual_boxes.append(actual_box)
            
            # add as extra columns 
            assert len(words) == len(actual_boxes)
            return {'words':words,'bbox':actual_boxes}
 
def get_tokens(tokenizer,words, unnormalized_word_boxes, normalized_word_boxes):
        token_boxes = []
        final_word_tokens =[]
        unnormalized_token_boxes = []
        for word, unnormalized_box, box in zip(words, unnormalized_word_boxes, normalized_word_boxes):
            word_tokens = tokenizer.tokenize(word)
            final_word_tokens.extend(word_tokens)
            unnormalized_token_boxes.extend(unnormalized_box for _ in range(len(word_tokens)))
            token_boxes.extend(box for _ in range(len(word_tokens)))
 
        return token_boxes, unnormalized_token_boxes,final_word_tokens # TODO: why are word_tokens not returned?
 
def get_centroid( actual_bbox): 
        centroid = []
        for i in actual_bbox:
          width = i[2] - i[0]
          height = i[3] - i[1]
          centroid.append([i[0] + width/2, i[1] + height/2])
        return centroid
 
def get_relative_distance(actual_bbox, centroid, index):
 
        a_rel_x = []
        a_rel_y = []
 
        for i in range(0, len(actual_bbox) - 1):
              if (i != index):
                prev = actual_bbox[i]
                curr = actual_bbox[i + 1]
                a_rel_x.append(
                    [
                      prev[0],            # top left x            
                      prev[2],            # bottom right x
                      prev[2] - prev[0],  # width
                      curr[0] - prev[0],  # diff top left x
                      curr[0] - prev[0],  # diff bottom left x
                      curr[2] - prev[2],  # diff top right x
                      curr[2] - prev[2],  # diff bottom right x
                      centroid[i + 1][0] - centroid[i][0],
                     ]
                )
                a_rel_y.append(
                    [
                      prev[1],            # top left y           
                      prev[3],            # bottom right y
                      prev[3] - prev[1],  # height
                      curr[1] - prev[1],  # diff top left y
                      curr[3] - prev[3],  # diff bottom left y
                      curr[1] - prev[1],  # diff top right y
                      curr[3] - prev[3],  # diff bottom right y
                      centroid[i + 1][1] - centroid[i][1],
                     ]
              )
              else:
                  a_rel_x.append([0] * 8)  # For the actual last word
                  a_rel_y.append([0] * 8)  # For the actual last word
 
        a_rel_x.append([0] * 8)  # For the last word
        a_rel_y.append([0] * 8)  # For the last word

        ## New changes

        def func(coordinates):
            return relative_position_bucket(torch.as_tensor(coordinates).type(torch.IntTensor)).tolist()

        a_rel_x = list(map(func,a_rel_x))
        a_rel_y = list(map(func,a_rel_y))
        return a_rel_x,a_rel_y
 
def resize_and_align_bounding_box(bbox, original_image, target_size):
      x_, y_ = original_image.size
      x_scale = target_size / x_ 
      y_scale = target_size / y_
      
      origLeft, origTop, origRight, origBottom = tuple(bbox)
      
      x = int(np.round(origLeft * x_scale))
      y = int(np.round(origTop * y_scale))
      xmax = int(np.round(origRight * x_scale))
      ymax = int(np.round(origBottom * y_scale)) 
      
      return [x, y, xmax, ymax]

def apply_mask(inputs):
            inputs = torch.as_tensor(inputs)
            # create random array of floats in equal dimension to input_ids
            rand = torch.rand(inputs.shape)
            # where the random array is less than 0.15, we set true
            mask_arr = (rand < 0.15) * (inputs != 101) * (inputs != 102)
            # create selection from mask_arr
            selection = torch.flatten((mask_arr).nonzero()).tolist()
            # apply selection index to inputs.input_ids, adding MASK tokens
            inputs[selection] = 103
            return inputs

def createPickelFile(image,pathToSave,tokenizer=None,target_size = 224,max_seq_length = 512,pretrain = True):

        if not os.path.exists(pathToSave):
          os.mkdir(pathToSave)

        if tokenizer==None:
            from transformers import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        pad_token_box = [0, 0, 0, 0]
        original_image = Image.open(image).convert("RGB")
        entries = apply_ocr(image)
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
        token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]
        unnormalized_token_boxes = [[0, 0, 0, 0]] + unnormalized_token_boxes + [[1000, 1000, 1000, 1000]]
 
        encoding = tokenizer(' '.join(words), padding='max_length', truncation=True)

        # Padding of token_boxes up the bounding boxes to the sequence length.
        input_ids = tokenizer(' '.join(words), truncation=True)["input_ids"]
        assert len(input_ids) == len(token_boxes)  # check if number of tokens match
        padding_length = max_seq_length - len(input_ids)
        token_boxes += [pad_token_box] * padding_length
        unnormalized_token_boxes += [pad_token_box] * padding_length
        encoding['bbox'] = token_boxes

        if pretrain:
            encoding['labels'] = encoding['input_ids']
            assert len(encoding['labels']) == max_seq_length
        
        else:
            encoding['labels'] = [label]
            assert len(encoding['labels']) == 1

        assert len(encoding['input_ids']) == max_seq_length
        assert len(encoding['attention_mask']) == max_seq_length
        assert len(encoding['token_type_ids']) == max_seq_length
        assert len(encoding['bbox']) == max_seq_length
        
        encoding['resized_image'] = ToTensor()(resized_image)

        ## Applying mask for the sake of pre-training
        encoding['input_ids'] = apply_mask(encoding['input_ids'])


        # rescale and align the bounding boxes to match the resized image size (typically 224x224) 
        encoding['resized_and_aligned_bounding_boxes'] = [resize_and_align_bounding_box(bbox, original_image, target_size) 
                                                          for bbox in unnormalized_token_boxes]
 
        index = -1
        # getting the index of the last word (using the input id)
        for i in range(len(encoding['input_ids']) - 1):
          if encoding['input_ids'][i + 1] == 0:
            index = i
            break

        # adding the relative position as well
        actual_bbox = encoding['resized_and_aligned_bounding_boxes']
        
        # Calculating the centroid
        centroid = get_centroid(actual_bbox)
 
        a_rel_x, a_rel_y = get_relative_distance(actual_bbox, centroid, index)
        encoding['unnormalized_token_boxes'] = unnormalized_token_boxes
        
        # finally, convert everything to PyTorch tensors 
        for k, v in encoding.items():
              encoding[k] = torch.as_tensor(encoding[k])
        encoding.update({"x_features": torch.as_tensor(a_rel_x), "y_features": torch.as_tensor(a_rel_y)})
        imageName = image.split('/')[-1]
        with open(f'{pathToSave}{imageName}.pickle', 'wb') as f:
              pickle.dump(encoding, f)
        return encoding


## images -> list, which contains the address of each image
#output = list(map(lambda x:createPickelFile(x,pathToSave),images))
