import math
import torch
import torch.nn as nn

class Patchembedding(nn.Module):
    def __init__(self, res = 28, patch_size = 4, in_channel = 1):
        """
        Patching the data through 2D conv

        2d conv : (batch_size, in_channel, res, res) -> (batch_size, embed_size, ((res - 4) / 4), (res - 4) /4)
        reshape : (batch_size, embed_size, ((res - 4) / 4 + 1) , (res - 4) /4 + 1) -> (batch_size, ((res - 4) / 4 + 1)^2. embed_size)

        (batch_size, in_channel, res, res) -> (batch_size, patch_num, embed_size)

        ex) MNIST : (batch_size, 1, 28, 28) -> (batch_size, 49, 16)
        """
        super().__init__()
        self.patch_size = patch_size
        self.in_channel = in_channel
        self.res = res
        self.embed_size = self.in_channel * patch_size * patch_size
        self.patch_num = int((self.res * self.res) / (self.patch_size ** 2))
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_size))

        self.embedding_layer = nn.Conv2d(in_channels = self.in_channel, out_channels = self.embed_size,
                                         kernel_size = self.patch_size, stride = self.patch_size)
        
    def forward(self, x):
        x = self.embedding_layer(x)
        ## (Batch_size, embed_size, sqrt(patch_num), sqrt(patch_num))
        x = x.view(-1, self.patch_num, self.embed_size)
        repeat_cls = self.cls_token.repeat(x.size()[0], 1, 1)
        x = torch.cat([repeat_cls, x], dim = 1)
        return x


class add_norm(nn.Module):
    def __init__(self, embed_dim = 16, p_dropout = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, func, x, *args):
        return self.norm(x + self.dropout(func(x, *args)))
    

class Encoder_block(nn.Module):
    def __init__(self, embed_dim = 16, head_num = 8, p_dropout = 0.1, d_ff = 1024):
        super().__init__()
        self.MHA = MultiHeadAttention(embed_dim = embed_dim, head_num = head_num)
        self.ffn = position_FFN(d = embed_dim, d_ff = d_ff)
        self.add_norm_mha = add_norm(embed_dim = embed_dim, p_dropout = p_dropout)
        self.add_norm_ffn = add_norm(embed_dim = embed_dim, p_dropout = p_dropout)

    def forward(self, x):
        x = self.add_norm_mha(self.MHA, x, x, x)
        x = self.add_norm_ffn(self.ffn, x)
        return x
    
class PositionalEncoding(nn.Module):
    """
    Positional Encoding 부분에서 sin, cos 정의할 때 Overflow, Underflow를 조심해야 한다.
    Position 위치의 길이가 엄청나게 길면? -> 매우 위험하다
    따라서 log와 exp 트릭을 써서 안전하게 지수 값을 계산 가능하다.
    """
    def __init__(self, d, max_len=1000):
        super().__init__()

        pe = torch.zeros(max_len, d)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d, 2) *
                             -(math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        out = x + self.pe[:, :x.size(1)]
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim = 16, head_num = 8):
        super().__init__()
        self.head_num = head_num
        self.embed_dim = embed_dim
        assert embed_dim % head_num == 0
        self.h = embed_dim // head_num

        self.W_q = nn.LazyLinear(self.embed_dim)
        self.W_k = nn.LazyLinear(self.embed_dim)
        self.W_v = nn.LazyLinear(self.embed_dim)
        self.W_o = nn.LazyLinear(self.embed_dim)

    def attention(self, query, key, value, mask =None):
        d_k = query.size(-1)
        score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            score = score.masked_fill(mask==0, 1e-9)
        prob = score.softmax(dim = 1)
        attn = torch.matmul(prob, value)
        return attn, prob
    
    def forward(self, query, key, value, mask = None):
        batch_size = query.size(0)
        q, k, v = self.W_q(query), self.W_k(key), self.W_v(value)
        # (batch_size, seq_len, embed_dim)
        ## 아마 붙어있는 것을 나눌 때만 reshape가 가능해서, reshape로 가지 않을까..?
        q, k, v = q.view(batch_size, -1, self.head_num, self.h).transpose(1,2), k.view(batch_size, -1, self.head_num, self.h).transpose(1,2), v.view(batch_size, -1, self.head_num, self.h).transpose(1,2)
        output, self.prob = self.attention(q,k,v,mask)
        # (batch_size, head_num, seq_len, embed_dim)
        output = output.transpose(1,2).reshape(batch_size, -1, self.embed_dim)
        output = self.W_o(output)
        return output
    
class position_FFN(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.W1 = nn.LazyLinear(d_ff)
        self.W2 = nn.LazyLinear(d)

    def forward(self, x):
        x = self.W1(x)
        x = self.W2(x)
        return x