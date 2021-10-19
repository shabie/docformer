# -*- coding: utf-8 -*-
# !git clone https://github.com/uakarsh/docformer.git
import sys
sys.path.extend(['docformer/'])
sys.path.extend(['docformer/src/docformer/'])

config = {
  "coordinate_size": 96,
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "image_feature_pool_shape": [
    7,
    7,
    256
  ],
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


from dataset_pytorch_lightening import *
pathToPickle = ''  # Path to the pickle file
entries = os.listdir(pathToPickle)
train_entries,val_entries = tts(entries,test_size = 0.2)
train_dataset = RVLCDIPDatset(train_entries,pathToPickle)
val_dataset = RVLCDIPDatset(val_entries,pathToPickle)

train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size = config['batch_size'],shuffle = True)

val_dataloader = torch.utils.data.DataLoader(val_dataset,batch_size = config['batch_size'],shuffle = True)

path = 'DocFormer_Oct16'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 5
from train_accelerator import run
run(config,train_dataloader,val_dataloader,device,epochs,path,classes = 30522)
