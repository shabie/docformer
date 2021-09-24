from torch import nn
import torch
from einops import rearrange


class RelativePosition(nn.Module):

    def __init__(self, head_dim, max_relative_position):
        super().__init__()
        self.head_dim = head_dim
        self.max_relative_position = max_relative_position
        # best explanation of nn.Parameters: https://github.com/lucidrains/vit-pytorch/issues/60
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, head_dim))  # +1 for own pos
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q).unsqueeze(0)
        range_vec_k = torch.arange(length_k).unsqueeze(1)
        distance_mat = range_vec_k - range_vec_q
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat)
        embeddings = self.embeddings_table[final_mat]

        return embeddings


class MultiModalAttentionLayer(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout):
        super().__init__()
        assert embed_dim % n_heads == 0

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.max_relative_positions = 16

        self.relative_positions_text = RelativePosition(self.head_dim, self.max_relative_positions)
        self.relative_positions_img = RelativePosition(self.head_dim, self.max_relative_positions)

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

        self.fc_o = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([embed_dim]))

    def forward(self, text_feat, img_feat, text_spatial_feat, img_spatial_feat):

        seq_length = text_feat.shape[1]

        # self attention of text
        # b -> batch, t -> time steps (l -> length has same meaning), head -> # of heads, k -> head dim.
        key_text_nh = rearrange(self.fc_k_text(text_feat), 'b t (head k) -> head b t k', head=self.n_heads)
        query_text_nh = rearrange(self.fc_q_text(text_feat), 'b l (head k) -> head b l k', head=self.n_heads)
        value_text_nh = rearrange(self.fc_v_text(text_feat), 'b t (head k) -> head b t k', head=self.n_heads)
        dots_text = torch.einsum('hblk,hbtk->hblt', query_text_nh, key_text_nh) / self.scale

        # 1D relative positions (query, key)
        rel_pos_embed_text = self.relative_positions_text(seq_length, seq_length)
        rel_pos_key_text = torch.einsum('bhrd,lrd->bhlr', key_text_nh, rel_pos_embed_text)
        rel_pos_query_text = torch.einsum('bhld,lrd->bhlr', query_text_nh, rel_pos_embed_text)

        # shared spatial <-> text hidden features
        key_spatial_text = self.fc_k_spatial(text_spatial_feat)
        query_spatial_text = self.fc_q_spatial(text_spatial_feat)
        key_spatial_text_nh = rearrange(key_spatial_text, 'b t (head k) -> head b t k', head=self.n_heads)
        query_spatial_text_nh = rearrange(query_spatial_text, 'b l (head k) -> head b l k', head=self.n_heads)
        dots_text_spatial = torch.einsum('hblk,hbtk->hblt',  query_spatial_text_nh, key_spatial_text_nh) / self.scale

        # Line 38 of pseudo-code
        text_attn_scores = dots_text + rel_pos_key_text + rel_pos_query_text + dots_text_spatial

        # self-attention of image
        key_img_nh = rearrange(self.fc_k_img(img_feat), 'b t (head k) -> head b t k', head=self.n_heads)
        query_img_nh = rearrange(self.fc_q_img(img_feat), 'b l (head k) -> head b l k', head=self.n_heads)
        value_img_nh = rearrange(self.fc_v_img(img_feat), 'b t (head k) -> head b t k', head=self.n_heads)
        dots_img = torch.einsum('hblk,hbtk->hblt', query_img_nh, key_img_nh) / self.scale

        # 1D relative positions (query, key)
        rel_pos_embed_img = self.relative_positions_img(seq_length, seq_length)
        rel_pos_key_img = torch.einsum('bhrd,lrd->bhlr', key_img_nh, rel_pos_embed_text)
        rel_pos_query_img = torch.einsum('bhld,lrd->bhlr', query_img_nh, rel_pos_embed_text)

        # shared spatial <-> image features
        key_spatial_img = self.fc_k_spatial(img_spatial_feat)
        query_spatial_img = self.fc_q_spatial(img_spatial_feat)
        key_spatial_img_nh = rearrange(key_spatial_img, 'b t (head k) -> head b t k', head=self.n_heads)
        query_spatial_img_nh = rearrange(query_spatial_img, 'b l (head k) -> head b l k', head=self.n_heads)
        dots_img_spatial = torch.einsum('hblk,hbtk->hblt',  query_spatial_img_nh, key_spatial_img_nh) / self.scale

        # Line 59 of pseudo-code
        img_attn_scores = dots_img + rel_pos_key_img + rel_pos_query_img + dots_img_spatial

        text_attn_probs = self.dropout(torch.softmax(text_attn_scores, dim=-1))
        img_attn_probs = self.dropout(torch.softmax(img_attn_scores, dim=-1))

        text_context = torch.einsum('hblt,hbtv->hblv', text_attn_probs, value_text_nh)
        img_context = torch.einsum('hblt,hbtv->hblv', img_attn_probs, value_img_nh)

        context = text_context + img_context

        embeddings = rearrange(context, 'head b t d -> b t (head d)')
        return embeddings


if __name__ == "__main__":
    print(torch.cuda.is_available())
    device = torch.device('cpu')
    text_embeddings = torch.randn(8, 512, 768)            # T-bar
    img_embeddings = torch.randn(8, 512, 768)             # V-bar
    text_spatial_embeddings = torch.randn(8, 512, 768)    # T-bar-s
    img_spatial_embeddings = torch.randn(8, 512, 768)     # V-bar-s
    mmha = MultiModalAttentionLayer(768, 12, 0.)
    mmha.to(device)
    output = mmha(text_embeddings, img_embeddings, text_spatial_embeddings, img_spatial_embeddings)
    print(output.shape)
