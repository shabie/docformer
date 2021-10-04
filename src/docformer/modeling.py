import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision.models as models

import math
from einops import rearrange


device = 'cuda'

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
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

    def forward(self) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size,seq_len, embedding_dim]
        """
        x = self.pe[0, : self.max_len]
        return self.dropout(x).unsqueeze(0)


class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_size = (3, 224, 224)
        resnet50 = models.resnet50(pretrained=False)
        modules = list(resnet50.children())[:-2]
        self.resnet50 = nn.Sequential(*modules)
        self.conv1 = nn.Conv2d(2048, 768, 1)
        self.relu1 = F.relu
        self.linear1 = nn.Linear(49, 512)

    def forward(self, x):
        x = self.resnet50(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = rearrange(x, 'b e width height -> b e (width height)')
        x = self.linear1(x)
        x = rearrange(x, 'b e seqlen -> b seqlen e')
        return x    


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

        self.x_topleft_position_embeddings_v = nn.Embedding(
            config["max_2d_position_embeddings"], config["coordinate_size"]
        )
        self.x_bottomright_position_embeddings_v = nn.Embedding(
            config["max_2d_position_embeddings"], config["coordinate_size"]
        )
        self.w_position_embeddings_v = nn.Embedding(
            config["max_2d_position_embeddings"], config["shape_size"]
        )
        self.x_topleft_distance_to_prev_embeddings_v = nn.Embedding(
            config["max_2d_position_embeddings"], config["shape_size"]
        )
        self.x_bottomleft_distance_to_prev_embeddings_v = nn.Embedding(
            config["max_2d_position_embeddings"], config["shape_size"]
        )
        self.x_topright_distance_to_prev_embeddings_v = nn.Embedding(
            config["max_2d_position_embeddings"], config["shape_size"]
        )
        self.x_bottomright_distance_to_prev_embeddings_v = nn.Embedding(
            config["max_2d_position_embeddings"], config["shape_size"]
        )
        self.x_centroid_distance_to_prev_embeddings_v = nn.Embedding(
            config["max_2d_position_embeddings"], config["shape_size"]
        )

        self.y_topleft_position_embeddings_v = nn.Embedding(
            config["max_2d_position_embeddings"], config["coordinate_size"]
        )
        self.y_bottomright_position_embeddings_v = nn.Embedding(
            config["max_2d_position_embeddings"], config["coordinate_size"]
        )
        self.h_position_embeddings_v = nn.Embedding(
            config["max_2d_position_embeddings"], config["shape_size"]
        )
        self.y_topleft_distance_to_prev_embeddings_v = nn.Embedding(
            2 * config["max_2d_position_embeddings"] + 1, config["shape_size"]
        )
        self.y_bottomleft_distance_to_prev_embeddings_v = nn.Embedding(
            2 * config["max_2d_position_embeddings"] + 1, config["shape_size"]
        )
        self.y_topright_distance_to_prev_embeddings_v = nn.Embedding(
            2 * config["max_2d_position_embeddings"] + 1, config["shape_size"]
        )
        self.y_bottomright_distance_to_prev_embeddings_v = nn.Embedding(
            2 * config["max_2d_position_embeddings"] + 1, config["shape_size"]
        )
        self.y_centroid_distance_to_prev_embeddings_v = nn.Embedding(
            2 * config["max_2d_position_embeddings"] + 1, config["shape_size"]
        )

        self.position_embeddings_t = PositionalEncoding(
            d_model=config["hidden_size"],
            dropout=0.1,
            max_len=config["max_position_embeddings"],
        )

        self.x_topleft_position_embeddings_t = nn.Embedding(
            config["max_2d_position_embeddings"], config["coordinate_size"]
        )
        self.x_bottomright_position_embeddings_t = nn.Embedding(
            config["max_2d_position_embeddings"], config["coordinate_size"]
        )
        self.w_position_embeddings_t = nn.Embedding(
            config["max_2d_position_embeddings"], config["shape_size"]
        )
        self.x_topleft_distance_to_prev_embeddings_t = nn.Embedding(
            2 * config["max_2d_position_embeddings"] + 1, config["shape_size"]
        )
        self.x_bottomleft_distance_to_prev_embeddings_t = nn.Embedding(
            2 * config["max_2d_position_embeddings"] + 1, config["shape_size"]
        )
        self.x_topright_distance_to_prev_embeddings_t = nn.Embedding(
            2 * config["max_2d_position_embeddings"] + 1, config["shape_size"]
        )
        self.x_bottomright_distance_to_prev_embeddings_t = nn.Embedding(
            2 * config["max_2d_position_embeddings"] + 1, config["shape_size"]
        )
        self.x_centroid_distance_to_prev_embeddings_t = nn.Embedding(
            2 * config["max_2d_position_embeddings"] + 1, config["shape_size"]
        )

        self.y_topleft_position_embeddings_t = nn.Embedding(
            config["max_2d_position_embeddings"], config["coordinate_size"]
        )
        self.y_bottomright_position_embeddings_t = nn.Embedding(
            config["max_2d_position_embeddings"], config["coordinate_size"]
        )
        self.h_position_embeddings_t = nn.Embedding(
            config["max_2d_position_embeddings"], config["shape_size"]
        )
        self.y_topleft_distance_to_prev_embeddings_t = nn.Embedding(
            2 * config["max_2d_position_embeddings"] + 1, config["shape_size"]
        )
        self.y_bottomleft_distance_to_prev_embeddings_t = nn.Embedding(
            2 * config["max_2d_position_embeddings"] + 1, config["shape_size"]
        )
        self.y_topright_distance_to_prev_embeddings_t = nn.Embedding(
            2 * config["max_2d_position_embeddings"] + 1, config["shape_size"]
        )
        self.y_bottomright_distance_to_prev_embeddings_t = nn.Embedding(
            2 * config["max_2d_position_embeddings"] + 1, config["shape_size"]
        )
        self.y_centroid_distance_to_prev_embeddings_t = nn.Embedding(
            2 * config["max_2d_position_embeddings"] + 1, config["shape_size"]
        )

        self.LayerNorm = nn.LayerNorm(
            config["hidden_size"], eps=config["layer_norm_eps"]
        )
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x_feature, y_feature):

        """
        Arguments:
        x_features of shape, (batch size, seq_len, 8)
        y_features of shape, (batch size, seq_len, 8)

        Outputs:

        (V-bar-s, T-bar-s) of shape (batch size, 512, 768), (batch size, 512, 768)

        What are the features:

        0 -> top left x|y
        1 -> bottom right x|y
        2 -> width|height
        3 -> diff top left x|y
        4 -> diff bottom left x|y
        5 -> diff top right x|y
        6 -> diff bottom right x|y
        7 -> centroid diff x|y
        """

        x_embedding_v = [
            self.x_topleft_position_embeddings_v,
            self.x_bottomright_position_embeddings_v,
            self.w_position_embeddings_v,
            self.x_topleft_distance_to_prev_embeddings_v,
            self.x_bottomleft_distance_to_prev_embeddings_v,
            self.x_topright_distance_to_prev_embeddings_v,
            self.x_bottomright_distance_to_prev_embeddings_v,
            self.x_centroid_distance_to_prev_embeddings_v,
        ]

        y_embedding_v = [
            self.y_topleft_position_embeddings_v,
            self.y_bottomright_position_embeddings_v,
            self.h_position_embeddings_v,
            self.y_topleft_distance_to_prev_embeddings_v,
            self.y_bottomleft_distance_to_prev_embeddings_v,
            self.y_topright_distance_to_prev_embeddings_v,
            self.y_bottomright_distance_to_prev_embeddings_v,
            self.y_centroid_distance_to_prev_embeddings_v,
        ]

        x_calculated_embedding_v = []
        y_calculated_embedding_v = []

        for i in range(8):

            temp_x = x_feature[:, :, i]  # Shape (batch_size, seq_len)
            x_calculated_embedding_v.append(x_embedding_v[i](temp_x.long()))
            temp_y = y_feature[:, :, i]
            y_calculated_embedding_v.append(y_embedding_v[i](temp_y.long()))

        x_calculated_embedding_v = torch.cat(x_calculated_embedding_v, dim=-1)
        y_calculated_embedding_v = torch.cat(y_calculated_embedding_v, dim=-1)

        ## Adding the positional encoding and the calculated_embedding_v
        v_bar_s = (
            x_calculated_embedding_v
            + y_calculated_embedding_v
            + self.position_embeddings_v()
        )

        x_embedding_t = [
            self.x_topleft_position_embeddings_t,
            self.x_bottomright_position_embeddings_t,
            self.w_position_embeddings_t,
            self.x_topleft_distance_to_prev_embeddings_t,
            self.x_bottomleft_distance_to_prev_embeddings_t,
            self.x_topright_distance_to_prev_embeddings_t,
            self.x_bottomright_distance_to_prev_embeddings_t,
            self.x_centroid_distance_to_prev_embeddings_t,
        ]

        y_embedding_t = [
            self.y_topleft_position_embeddings_t,
            self.y_bottomright_position_embeddings_t,
            self.h_position_embeddings_t,
            self.y_topleft_distance_to_prev_embeddings_t,
            self.y_bottomleft_distance_to_prev_embeddings_t,
            self.y_topright_distance_to_prev_embeddings_t,
            self.y_bottomright_distance_to_prev_embeddings_t,
            self.y_centroid_distance_to_prev_embeddings_t,
        ]

        x_calculated_embedding_t = []
        y_calculated_embedding_t = []

        for i in range(8):

            temp_x = x_feature[:, :, i]  # (batch_size, seq_len)
            x_calculated_embedding_t.append(x_embedding_t[i](temp_x.long()))
            temp_y = y_feature[:, :, i]
            y_calculated_embedding_t.append(y_embedding_t[i](temp_y.long()))

        x_calculated_embedding_t = torch.cat(x_calculated_embedding_t, dim=-1)
        y_calculated_embedding_t = torch.cat(y_calculated_embedding_t, dim=-1)

        t_bar_s = (
            x_calculated_embedding_t
            + y_calculated_embedding_t
            + self.position_embeddings_t()
        )

        return v_bar_s, t_bar_s


# fmt: off
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        
        # Fig 1: http://proceedings.mlr.press/v119/xiong20b/xiong20b.pdf
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class PreNormAttn(nn.Module):
    def __init__(self, dim, fn):
        
        # Fig 1: http://proceedings.mlr.press/v119/xiong20b/xiong20b.pdf
        super().__init__()
        
        self.norm_t_bar = nn.LayerNorm(dim)
        self.norm_v_bar = nn.LayerNorm(dim)
        self.norm_t_bar_s = nn.LayerNorm(dim)
        self.norm_v_bar_s = nn.LayerNorm(dim)
        self.fn = fn
        
    def forward(self, t_bar, v_bar, t_bar_s, v_bar_s, **kwargs):
        return self.fn(self.norm_t_bar(t_bar),
                       self.norm_v_bar(v_bar),
                       self.norm_t_bar_s(t_bar_s),
                       self.norm_v_bar_s(v_bar_s), **kwargs)
    

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class RelativePosition(nn.Module):

    def __init__(self, num_units, max_relative_position, max_seq_length):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        self.max_length = max_seq_length
        range_vec_q = torch.arange(max_seq_length)
        range_vec_k = torch.arange(max_seq_length)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        self.final_mat = torch.LongTensor(final_mat)
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        embeddings = self.embeddings_table[self.final_mat[:length_q, :length_k]]
        return embeddings


class MultiModalAttentionLayer(nn.Module):
    def __init__(self, embed_dim, n_heads, max_relative_position, max_seq_length, dropout):
        super().__init__()
        assert embed_dim % n_heads == 0

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        self.relative_positions_text = RelativePosition(self.head_dim, max_relative_position, max_seq_length)
        self.relative_positions_img = RelativePosition(self.head_dim, max_relative_position, max_seq_length)

        # text qkv embeddings
        self.fc_k_text = nn.Linear(embed_dim, embed_dim)
        self.fc_q_text = nn.Linear(embed_dim, embed_dim)
        self.fc_v_text = nn.Linear(embed_dim, embed_dim)

        # image qkv embeddings
        self.fc_k_img = nn.Linear(embed_dim, embed_dim)
        self.fc_q_img = nn.Linear(embed_dim, embed_dim)
        self.fc_v_img = nn.Linear(embed_dim, embed_dim)

        # spatial qk embeddings (shared for visual and text)
        self.fc_k_spatial = nn.Linear(embed_dim, embed_dim)
        self.fc_q_spatial = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)

        self.to_out = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.scale = torch.sqrt(torch.FloatTensor([embed_dim]))

    def forward(self, text_feat, img_feat, text_spatial_feat, img_spatial_feat):
        text_feat = text_feat
        img_feat = img_feat
        text_spatial_feat = text_spatial_feat
        img_spatial_feat = img_spatial_feat
        seq_length = text_feat.shape[1]

        # self attention of text
        # b -> batch, t -> time steps (l -> length has same meaning), head -> # of heads, k -> head dim.
        key_text_nh = rearrange(self.fc_k_text(text_feat), 'b t (head k) -> head b t k', head=self.n_heads).to(device)
        query_text_nh = rearrange(self.fc_q_text(text_feat), 'b l (head k) -> head b l k', head=self.n_heads).to(device)
        value_text_nh = rearrange(self.fc_v_text(text_feat), 'b t (head k) -> head b t k', head=self.n_heads).to(device)
        dots_text = torch.einsum('hblk,hbtk->hblt', query_text_nh, key_text_nh) / self.scale.to(device)

        # 1D relative positions (query, key)
        rel_pos_embed_text = self.relative_positions_text(seq_length, seq_length)
        rel_pos_key_text = torch.einsum('bhrd,lrd->bhlr', key_text_nh, rel_pos_embed_text)
        rel_pos_query_text = torch.einsum('bhld,lrd->bhlr', query_text_nh, rel_pos_embed_text)

        # shared spatial <-> text hidden features
        key_spatial_text = self.fc_k_spatial(text_spatial_feat)
        query_spatial_text = self.fc_q_spatial(text_spatial_feat)
        key_spatial_text_nh = rearrange(key_spatial_text, 'b t (head k) -> head b t k', head=self.n_heads).to(device)
        query_spatial_text_nh = rearrange(query_spatial_text, 'b l (head k) -> head b l k', head=self.n_heads).to(device)
        dots_text_spatial = torch.einsum('hblk,hbtk->hblt',  query_spatial_text_nh, key_spatial_text_nh) / self.scale.to(device)

        # Line 38 of pseudo-code
        text_attn_scores = dots_text + rel_pos_key_text + rel_pos_query_text + dots_text_spatial

        # self-attention of image
        key_img_nh = rearrange(self.fc_k_img(img_feat), 'b t (head k) -> head b t k', head=self.n_heads).to(device)
        query_img_nh = rearrange(self.fc_q_img(img_feat), 'b l (head k) -> head b l k', head=self.n_heads).to(device)
        value_img_nh = rearrange(self.fc_v_img(img_feat), 'b t (head k) -> head b t k', head=self.n_heads).to(device)
        dots_img = torch.einsum('hblk,hbtk->hblt', query_img_nh, key_img_nh) / self.scale.to(device)

        # 1D relative positions (query, key)
        rel_pos_embed_img = self.relative_positions_img(seq_length, seq_length)
        rel_pos_key_img = torch.einsum('bhrd,lrd->bhlr', key_img_nh, rel_pos_embed_text)
        rel_pos_query_img = torch.einsum('bhld,lrd->bhlr', query_img_nh, rel_pos_embed_text)

        # shared spatial <-> image features
        key_spatial_img = self.fc_k_spatial(img_spatial_feat)
        query_spatial_img = self.fc_q_spatial(img_spatial_feat)
        key_spatial_img_nh = rearrange(key_spatial_img, 'b t (head k) -> head b t k', head=self.n_heads).to(device)
        query_spatial_img_nh = rearrange(query_spatial_img, 'b l (head k) -> head b l k', head=self.n_heads).to(device)
        dots_img_spatial = torch.einsum('hblk,hbtk->hblt',  query_spatial_img_nh, key_spatial_img_nh) / self.scale.to(device)

        # Line 59 of pseudo-code
        img_attn_scores = dots_img + rel_pos_key_img + rel_pos_query_img + dots_img_spatial

        text_attn_probs = self.dropout(torch.softmax(text_attn_scores, dim=-1))
        img_attn_probs = self.dropout(torch.softmax(img_attn_scores, dim=-1))

        text_context = torch.einsum('hblt,hbtv->hblv', text_attn_probs, value_text_nh)
        img_context = torch.einsum('hblt,hbtv->hblv', img_attn_probs, value_img_nh)

        context = text_context + img_context

        embeddings = rearrange(context, 'head b t d -> b t (head d)')
        return self.to_out(embeddings)
    


class DocFormerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([])
        for _ in range(config['num_hidden_layers']):
            encoder_block = nn.ModuleList([
                    PreNormAttn(config['hidden_size'], 
                            MultiModalAttentionLayer(config['hidden_size'],
                                                     config['num_attention_heads'],
                                                     config['max_relative_positions'],
                                                     config['max_position_embeddings'],
                                                     config['hidden_dropout_prob'],
                                                    )
                           ),
                    PreNorm(config['hidden_size'],
                            FeedForward(config['hidden_size'],
                                        config['hidden_size'] * config['intermediate_ff_size_factor'],
                                        dropout = config['hidden_dropout_prob']))
            ])
            self.layers.append(encoder_block)

    def forward(
        self, 
        text_feat,  # text feat or output from last encoder block
        img_feat,
        text_spatial_feat,
        img_spatial_feat,
    ):
        # Fig 1 encoder part (skip conn for both attn & FF): https://arxiv.org/abs/1706.03762
        # TODO: ensure 1st skip conn (var "skip") in such a multimodal setting makes sense (most likely does)
        for attn, ff in self.layers:
                skip = text_feat + img_feat + text_spatial_feat + img_spatial_feat
                x = attn(text_feat, img_feat, text_spatial_feat, img_spatial_feat) + skip
                x = ff(x) + x
                text_feat  = x
        return x   

    
class LanguageFeatureExtractor(nn.Module):
    def __init__(self, max_vocab_size, hidden_dim):
        super().__init__()
        self.embedding_vector = nn.Embedding(max_vocab_size,hidden_dim)
    def forward(self, x):
        return self.embedding_vector(x)


class ExtractFeatures(nn.Module):

    '''
    Inputs: dictionary
    Output: v_bar, t_bar, v_bar_s, t_bar_s

    '''

    def __init__(self,config):
        super().__init__()
        self.visual_feature = ResNetFeatureExtractor()
        self.language_feature = LanguageFeatureExtractor(config['vocab_size'], config['hidden_size'])
        self.spatial_feature = DocFormerEmbeddings(config)

    def forward(self,encoding):
        image = encoding['resized_image']
        language = encoding['input_ids']
        x_feature = encoding['x_features']
        y_feature = encoding['y_features']

        v_bar = self.visual_feature(image)
        t_bar = self.language_feature(language)
        v_bar_s, t_bar_s = self.spatial_feature(x_feature, y_feature)
        return v_bar, t_bar, v_bar_s, t_bar_s


class DocFormer(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        self.config = config
        self.extract_feature = ExtractFeatures(config)
        self.encoder = DocFormerEncoder(config)
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])
        self.classifier = nn.Linear(config['hidden_size'], num_classes)


    def forward(self, x):
        v_bar, t_bar, v_bar_s, t_bar_s = self.extract_feature(x)
        features = {'v_bar':v_bar, 't_bar':t_bar,'v_bar_s':v_bar_s,'t_bar_s':t_bar_s}
        for f in features:
            features[f] = features[f]
        output = self.encoder(features['t_bar'], features['v_bar'], features['t_bar_s'], features['v_bar_s'])
        output = self.dropout(output)
        output = self.classifier(output)
        return output
