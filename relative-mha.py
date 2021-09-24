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


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout, device):
        super().__init__()
        
        assert embed_dim % n_heads == 0
        
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.max_relative_position = 2

        self.relative_position_k = RelativePosition(self.head_dim, self.max_relative_position)
        self.relative_position_v = RelativePosition(self.head_dim, self.max_relative_position)

        self.fc_q = nn.Linear(embed_dim, embed_dim)
        self.fc_k = nn.Linear(embed_dim, embed_dim)
        self.fc_v = nn.Linear(embed_dim, embed_dim)
        
        self.fc_o = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask=None):
        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]
        batch_size = query.shape[0]
        len_k, len_q, len_v = key.shape[1], query.shape[1], value.shape[1]

        # linear transformation for all heads at once
        query = self.fc_q(query)
        key = self.fc_k(key)
        value = self.fc_v(value)

        # b -> batch, s -> sequence length, nh -> number of heads, d -> attention head dimension
        query_nh = rearrange(query, 'b s (nh d) -> b nh s d', nh=self.n_heads)
        key_nh = rearrange(query, 'b s (nh d) -> b nh d s', nh=self.n_heads)
        dots = torch.einsum('ijkl,ijlm->ijkm', query_nh, key_nh)

        # position embeddings are shared across heads and sequences so "merge" heads with batch into a single dimension
        query_rp = rearrange(query, 'b s (nh d) -> s (b nh) d', nh=self.n_heads).contiguous()
        key_rp = rearrange(self.relative_position_k(len_q, len_k), 's1 s2 d -> s1 d s2')
        rp_dots = rearrange(torch.matmul(query_rp, key_rp), 's1 (b nh) s2 -> b nh s1 s2', nh=self.n_heads).contiguous()

        attn = (dots + rp_dots) / self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e10)
        attn = self.dropout(torch.softmax(attn, dim=-1))

        value_nh = rearrange(value, 'b s (nh d) -> b nh s d', nh=self.n_heads)
        weight1 = torch.matmul(attn, value_nh)

        value_rp = self.relative_position_v(len_q, len_v)  # returns 512 x 512 x 64
        attn = rearrange(attn, 'b nh s1 s2 -> s1 (b nh) s2').contiguous()
        weight2 = rearrange(torch.matmul(attn, value_rp), 's (b nh) d -> b nh s d', nh=self.n_heads).contiguous()

        embeddings = weight1 + weight2
        
        # embeddings = [batch size, n heads, query len, head dim]
        embeddings = rearrange(embeddings, 'b nh s d -> b s (nh d)').contiguous()

        # embeddings = [batch size, query len, hid dim]
        embeddings = self.fc_o(embeddings)
        
        # embeddings = [batch size, query len, hid dim]
        return embeddings


if __name__ == "__main__":
    print(torch.cuda.is_available())
    device = torch.device('cpu')
    v = torch.randn(8, 512, 768)
    v.to(device)
    mha = MultiHeadAttentionLayer(768, 12, 0, device)
    mha.to(device)
    output = mha(v, v, v)
    print(output.shape)
