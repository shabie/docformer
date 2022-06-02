'''
Utility File

Created basically for the purpose of defining the labels for the unsupervised task of "Text Describe Image", as in the paper DocFormer
'''

import numpy as np
import random
import json
import os

def labels_for_tdi(length,sample_ratio=0.05):

      '''

      Function for providing the labels for the task of
      Text Describes Image

      Input: 
      * Length of the dataset, i.e the total number of data points
      * sample_ratio: The percentage of the total length, which you want to shuffle

      Output: 
      * d_arr (dummy array): The array which contains the indexes of the images
      * Labels: Whether the image has been shuffled with some other image or not


      Example:
      d_arr,labels = labels_for_tdi(100)
      
      Explanation:
      Suppose, the array is [1,2,3,4,5], so, the steps are as follows:
      
      * Choose some arbitrary values, (refer the samples_to_be_changed variable) (let us suppose [2,4])
      * Generate the permutation of the same, and replace the arbitary values with their permutations (one permutation can be [4,2], and hence the array becomes
      [1,4,3,2,5]
      * And then, if the original arr and the d_arr's arguments matches, put a label of 1, else put 0, hence the labels array becomes [1,0,1,0,1]
      
      
      The purpose of returning d_arr is, because the d_arr[i] would be the argument, which is responsible for becoming the d_resized_scaled_img of ith encoding dictinary vector
      
      i.e if d_arr[i] == i (means not shuffled), the d_resized_scaled_img of ith entry would be same else resized_scaled_img, 
      else d_sized_scaled_img[i] = resized_scaled_img[d_arr[i]] 
      
      '''
      samples_to_be_changed = int(sample_ratio*length)
      arr = np.arange(length)
      d_arr = arr.copy()
      labels = np.ones(length)
      sample_id = np.array(random.sample(list(arr), samples_to_be_changed))
      new_sample_id = np.random.permutation(sample_id)
      d_arr[sample_id]=new_sample_id
      labels = (arr==d_arr).astype(int)

      return d_arr,labels


## Purpose: Reading the json file from the path and return the dictionary
def load_json_file(file_path):
  with open(file_path, 'r') as f:
    data = json.load(f)
  return data

## Purpose: Getting the address of specific file type, eg: .pdf, .tif, so and so
def get_specific_file(path, last_entry = 'tif'):
  base_path = path
  for i in os.listdir(path):
    if i.endswith(last_entry):
      return os.path.join(base_path, i)

  return '-1'

