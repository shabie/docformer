import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision.models as models

import math
from einops import rearrange


class RelPosEmb1D(nn.Module):
    def __init__(self, tokens, head_dim):
        """
        Output: [batch head tokens tokens]
        Args:
            tokens: the number of the tokens of the seq
            head_dim: the size of the last dimension of q
        """
        super().__init__()
        scale = head_dim ** -0.5
        self.rel_pos_emb = nn.Parameter(torch.randn(2 * tokens - 1, head_dim) * scale)

    def forward(self, q):
        return torch.einsum('b h t d, r d -> b h t r', q, self.rel_pos_emb)


class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_size = (3, 224, 224)

        # Making the resnet 50 model, which was used in the DocFormer for the purpose of visual feature extraction

        resnet50 = models.resnet50(pretrained=False)
        modules = list(resnet50.children())[:-2]
        self.resnet50 = nn.Sequential(*modules)  # TODO: check if this is correct
        self.conv = nn.Conv2d(2048, 768, 1)
        self.relu = F.relu
        self.linear = nn.Linear(49, 512)

    def forward(self, x):
        x = self.resnet50(x)
        x = self.conv(x)
        x = self.relu(x)
        x = rearrange(
            x, "b e w h -> b e (w h)"
        )  # b -> batch, e -> embedding dim, w -> width, h -> height
        x = self.linear(x)
        x = rearrange(
            x, "b e s -> b s e"
        )  # b -> batch, e -> embedding dim, s -> sequence length
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        self.d_model = d_model
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[: x.size(1)]
        return self.dropout(x)


config = {
    "attention_probs_dropout_prob": 0.1,
    "coordinate_size": 64,
    "fast_qkv": True,
    "gradient_checkpointing": False,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 768,
    "image_feature_pool_shape": [7, 7, 256],
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "layer_norm_eps": 1e-12,
    "max_2d_positions": 1024,
    "rel_positions_attn_span": 8,
    "max_position_embeddings": 512,
    "model_type": "docformer",
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "output_past": True,
    "pad_token_id": 0,
    "shape_size": 64,
    "vocab_size": 30522,
}


class DocFormerEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super(DocFormerEmbeddings, self).__init__()

        self.config = config
        self.word_embeddings = nn.Embedding(
            config["vocab_size"],
            config["hidden_size"],
            padding_idx=config["pad_token_id"],
        )

        self.position_embeddings_v = PositionalEncoding(
            d_model=config["hidden_size"],
            dropout=0.1,
            max_len=config["max_position_embeddings"],
        )

        max_2d_pos = config["max_2d_positions"]

        self.x_topleft_position_embeddings_v = nn.Embedding(max_2d_pos, config["coordinate_size"])
        self.x_bottomright_position_embeddings_v = nn.Embedding(max_2d_pos, config["coordinate_size"])
        self.w_position_embeddings_v = nn.Embedding(max_2d_pos, config["shape_size"])
        self.x_topleft_distance_to_prev_embeddings_v = nn.Embedding(max_2d_pos, config["shape_size"])
        self.x_bottomleft_distance_to_prev_embeddings_v = nn.Embedding(max_2d_pos, config["shape_size"])
        self.x_topright_distance_to_prev_embeddings_v = nn.Embedding(max_2d_pos, config["shape_size"])
        self.x_bottomright_distance_to_prev_embeddings_v = nn.Embedding(max_2d_pos, config["shape_size"])
        self.x_centroid_distance_to_prev_embeddings_v = nn.Embedding(max_2d_pos, config["shape_size"])

        self.y_topleft_position_embeddings_v = nn.Embedding(max_2d_pos, config["coordinate_size"])
        self.y_bottomright_position_embeddings_v = nn.Embedding(max_2d_pos, config["coordinate_size"])
        self.h_position_embeddings_v = nn.Embedding(max_2d_pos, config["shape_size"])
        self.y_topleft_distance_to_prev_embeddings_v = nn.Embedding(max_2d_pos, config["shape_size"])
        self.y_bottomleft_distance_to_prev_embeddings_v = nn.Embedding(max_2d_pos, config["shape_size"])
        self.y_topright_distance_to_prev_embeddings_v = nn.Embedding(max_2d_pos, config["shape_size"])
        self.y_bottomright_distance_to_prev_embeddings_v = nn.Embedding(max_2d_pos, config["shape_size"])
        self.y_centroid_distance_to_prev_embeddings_v = nn.Embedding(max_2d_pos, config["shape_size"])

        self.position_embeddings_t = PositionalEncoding(
            d_model=config["hidden_size"],
            dropout=0.1,
            max_len=config["max_position_embeddings"],
        )

        self.x_topleft_position_embeddings_t = nn.Embedding(max_2d_pos, config["coordinate_size"])
        self.x_bottomright_position_embeddings_t = nn.Embedding(max_2d_pos, config["coordinate_size"])
        self.w_position_embeddings_t = nn.Embedding(max_2d_pos, config["shape_size"])
        self.x_topleft_distance_to_prev_embeddings_t = nn.Embedding(max_2d_pos, config["shape_size"])
        self.x_bottomleft_distance_to_prev_embeddings_t = nn.Embedding(max_2d_pos, config["shape_size"])
        self.x_topright_distance_to_prev_embeddings_t = nn.Embedding(max_2d_pos, config["shape_size"])
        self.x_bottomright_distance_to_prev_embeddings_t = nn.Embedding(max_2d_pos, config["shape_size"])
        self.x_centroid_distance_to_prev_embeddings_t = nn.Embedding(max_2d_pos, config["shape_size"])

        self.y_topleft_position_embeddings_t = nn.Embedding(max_2d_pos, config["coordinate_size"])
        self.y_bottomright_position_embeddings_t = nn.Embedding(max_2d_pos, config["coordinate_size"])
        self.h_position_embeddings_t = nn.Embedding(max_2d_pos, config["shape_size"])
        self.y_topleft_distance_to_prev_embeddings_t = nn.Embedding(max_2d_pos, config["shape_size"])
        self.y_bottomleft_distance_to_prev_embeddings_t = nn.Embedding(max_2d_pos, config["shape_size"])
        self.y_topright_distance_to_prev_embeddings_t = nn.Embedding(max_2d_pos, config["shape_size"])
        self.y_bottomright_distance_to_prev_embeddings_t = nn.Embedding(max_2d_pos, config["shape_size"])
        self.y_centroid_distance_to_prev_embeddings_t = nn.Embedding(max_2d_pos, config["shape_size"])

        self.LayerNorm = nn.LayerNorm(config["hidden_size"], eps=config["layer_norm_eps"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x_feature, y_feature):

        """
        Arguments:
        x_features of shape, (batch size, seq_len, 8)
        y_features of shape, (batch size, seq_len, 8)

        Outputs:

        (V-bar-s, T-bar-s) of shape (batch size, 512,768),(batch size, 512,768)

        What are the features:

        0 -> top left x/y
        1 -> bottom right x/y
        2 -> width/height
        3 -> diff top left x/y
        4 -> diff bottom left x/y
        5 -> diff top right x/y
        6 -> diff bottom right x/y
        7 -> centroid diff x/y
        """

        """
  
        Calculating V-bar-s,
  
        Steps:  
        1. Apply embedding to each of the coordinate, and then concatenate it
        2. Apply Absolute Positional Encoding
        """
        x_embedding_v = [
            self.x_topleft_position_embeddings_v,
            self.x_bottomright_position_embeddings_v,
            self.w_position_embeddings_v,
            self.x_topleft_relative_position_embeddings_v,
            self.x_bottomleft_relative_position_embeddings_v,
            self.x_topright_relative_position_embeddings_v,
            self.x_bottomright_relative_position_embeddings_v,
            self.x_centroid_relative_position_embeddings_v,
        ]

        y_embedding_v = [
            self.y_topleft_position_embeddings_v,
            self.y_bottomright_position_embeddings_v,
            self.h_position_embeddings_v,
            self.y_topleft_relative_position_embeddings_v,
            self.y_bottomleft_relative_position_embeddings_v,
            self.y_topright_relative_position_embeddings_v,
            self.y_bottomright_relative_position_embeddings_v,
            self.y_centroid_relative_position_embeddings_v,
        ]

        x_calculated_embedding_v = []
        y_calculated_embedding_v = []

        for i in range(8):
            if i < 3:

                # For the normal coordinates, just pass it through the embedding layers

                temp_x = x_feature[:, :, i]  # Shape (batch_size,seq_len)
                x_calculated_embedding_v.append(x_embedding_v[i](temp_x.long()))
                temp_y = y_feature[:, :, i]
                y_calculated_embedding_v.append(y_embedding_v[i](temp_y.long()))

            else:
                # Else, follow this https://github.com/evelinehong/Transformer_Relative_Position_PyTorch/blob/master/relative_position.py

                temp_x = x_feature[:, :, i]  # Shape (batch_size,seq_len)
                temp_x = torch.clamp(
                    temp_x,
                    -self.config["max_2d_position_embeddings"],
                    self.config["max_2d_position_embeddings"],
                )
                temp_x = temp_x + self.config["max_2d_position_embeddings"]
                x_calculated_embedding_v.append(x_embedding_v[i](temp_x.long()))

                temp_y = y_feature[:, :, i]  # Shape (batch_size,seq_len,1)
                temp_y = torch.clamp(
                    temp_y,
                    -self.config["max_2d_position_embeddings"],
                    self.config["max_2d_position_embeddings"],
                )
                temp_y = temp_y + self.config["max_2d_position_embeddings"]
                y_calculated_embedding_v.append(y_embedding_v[i](temp_y.long()))

        x_calculated_embedding_v = torch.cat(x_calculated_embedding_v, dim=-1)
        y_calculated_embedding_v = torch.cat(y_calculated_embedding_v, dim=-1)

        ## Adding the positional encoding and the calculated_embedding_v
        v_bar_s = (
                x_calculated_embedding_v
                + y_calculated_embedding_v
                + self.position_embeddings_v()
        )

        """
  
        Calculating t-bar-s,
  
        Steps:  
        1. Apply embedding to each of the coordinate, and then concatenate it
        2. Apply Absolute Positional Encoding
        """
        x_embedding_t = [
            self.x_topleft_position_embeddings_t,
            self.x_bottomright_position_embeddings_t,
            self.w_position_embeddings_t,
            self.x_topleft_relative_position_embeddings_t,
            self.x_bottomleft_relative_position_embeddings_t,
            self.x_topright_relative_position_embeddings_t,
            self.x_bottomright_relative_position_embeddings_t,
            self.x_centroid_relative_position_embeddings_t,
        ]

        y_embedding_t = [
            self.y_topleft_position_embeddings_t,
            self.y_bottomright_position_embeddings_t,
            self.h_position_embeddings_t,
            self.y_topleft_relative_position_embeddings_t,
            self.y_bottomleft_relative_position_embeddings_t,
            self.y_topright_relative_position_embeddings_t,
            self.y_bottomright_relative_position_embeddings_t,
            self.y_centroid_relative_position_embeddings_t,
        ]

        x_calculated_embedding_t = []
        y_calculated_embedding_t = []

        for i in range(8):
            if i < 3:

                # For the normal coordinates, just pass it through the embedding layers

                temp_x = x_feature[:, :, i]  # Shape (batch_size,seq_len)
                x_calculated_embedding_t.append(x_embedding_t[i](temp_x.long()))
                temp_y = y_feature[:, :, i]
                y_calculated_embedding_t.append(y_embedding_t[i](temp_y.long()))

            else:
                # Else, follow this https://github.com/evelinehong/Transformer_Relative_Position_PyTorch/blob/master/relative_position.py

                temp_x = x_feature[:, :, i]  # Shape (batch_size,seq_len)
                temp_x = torch.clamp(
                    temp_x,
                    -self.config["max_2d_position_embeddings"],
                    self.config["max_2d_position_embeddings"],
                )
                temp_x = temp_x + self.config["max_2d_position_embeddings"]
                x_calculated_embedding_t.append(x_embedding_t[i](temp_x.long()))

                temp_y = y_feature[:, :, i]  # Shape (batch_size,seq_len,1)
                temp_y = torch.clamp(
                    temp_y,
                    -self.config["max_2d_position_embeddings"],
                    self.config["max_2d_position_embeddings"],
                )
                temp_y = temp_y + self.config["max_2d_position_embeddings"]
                y_calculated_embedding_t.append(y_embedding_t[i](temp_y.long()))

        x_calculated_embedding_t = torch.cat(x_calculated_embedding_t, dim=-1)
        y_calculated_embedding_t = torch.cat(y_calculated_embedding_t, dim=-1)

        ## Adding the positional encoding and the calculated_embedding_t
        t_bar_s = (
                x_calculated_embedding_t
                + y_calculated_embedding_t
                + self.position_embeddings_t()
        )

        return v_bar_s, t_bar_s
