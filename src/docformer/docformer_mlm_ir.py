

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
  "batch_size":1
}

from modeling import *
import torch
import torch.nn as nn


import torch.nn as nn
import torch
class ShallowDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        # decoder 
        self.linear1 = nn.Linear(in_features = 768,out_features = 512)                        # Making the image to be symmetric
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 3,kernel_size = 3,stride = 1)
        self.conv2 = nn.Conv2d(in_channels = 3, out_channels = 3,kernel_size = 3,stride = 1)

        self.conv3 = nn.Conv2d(in_channels = 3, out_channels = 3,kernel_size = 5,stride = 1)
        self.conv4 = nn.Conv2d(in_channels = 3, out_channels = 3,kernel_size = 5,stride = 2)

        self.conv5 = nn.Conv2d(in_channels = 3, out_channels = 3,kernel_size = 7)
        self.conv6 = nn.Conv2d(in_channels = 3, out_channels = 3,kernel_size = 7)

        self.conv7 = nn.Conv2d(in_channels = 3, out_channels = 3,kernel_size = 7)
        self.conv8 = nn.Conv2d(in_channels = 3, out_channels = 3,kernel_size = 7)

        self.conv9 = nn.Conv2d(in_channels = 3, out_channels = 3,kernel_size = 3,stride = 1)

    def forward(self, x):
          x = x.unsqueeze(1).cuda()
          x = nn.ReLU()(torch.nn.BatchNorm2d(num_features=1,device = 'cuda')(self.linear1(x)))
          x = nn.ReLU()(torch.nn.BatchNorm2d(num_features=3,device = 'cuda')(self.conv1(x)))
          x = nn.ReLU()(torch.nn.BatchNorm2d(num_features=3,device = 'cuda')(self.conv2(x)))
          x = nn.ReLU()(torch.nn.BatchNorm2d(num_features=3,device = 'cuda')(self.conv3(x)))
          x = nn.ReLU()(torch.nn.BatchNorm2d(num_features=3,device = 'cuda')(self.conv4(x)))
          x = nn.ReLU()(torch.nn.BatchNorm2d(num_features=3,device = 'cuda')(self.conv5(x)))
          x = nn.ReLU()(torch.nn.BatchNorm2d(num_features=3,device = 'cuda')(self.conv6(x)))
          x = nn.ReLU()(torch.nn.BatchNorm2d(num_features=3,device = 'cuda')(self.conv7(x)))
          x = nn.ReLU()(torch.nn.BatchNorm2d(num_features=3,device = 'cuda')(self.conv8(x)))
          x = nn.ReLU()(torch.nn.BatchNorm2d(num_features=3,device = 'cuda')(self.conv9(x)))
          return torch.sigmoid(x)

class DocFormer_For_IR(nn.Module):
    def __init__(self, config,num_classes= 30522):
        super().__init__()
        self.config = config
        self.extract_feature = ExtractFeatures(config)
        self.encoder = DocFormerEncoder(config)
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])
        self.classifier = nn.Linear(in_features = 768,out_features = num_classes)
        self.decoder = ShallowDecoder().cuda()

    def forward(self, x):
        v_bar, t_bar, v_bar_s, t_bar_s = self.extract_feature(x)
        features = {'v_bar': v_bar, 't_bar': t_bar, 'v_bar_s': v_bar_s, 't_bar_s': t_bar_s}
        for f in features:
            features[f] = features[f]
        output = self.encoder(features['t_bar'], features['v_bar'], features['t_bar_s'], features['v_bar_s'])
        output = self.dropout(output)
        output_mlm = self.classifier(output)
        output_ir = self.decoder(output) 
        return {'mlm_labels':output_mlm,'ir':output_ir}
# model = DocFormer_For_IR(config).cuda()

# import pickle
# import os
# path = 'drive/MyDrive/RVL-CDIP-PickleFiles/'
# entry = os.listdir(path)[0]
# entry = pickle.load(open(path+entry,'rb'))
# for i in list(entry.keys()):
#   entry[i] = entry[i].unsqueeze(0).cuda()

# out = model(entry)

