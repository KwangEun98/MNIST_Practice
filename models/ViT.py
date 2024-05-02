import torch
import torch.nn as nn

import sys
sys.path.append('C:/Users/DMQA/DMQA_documents/DMQA_documents/2024-1/Seminar/Mentoring/MNIST_practice')

from layers.transformer_layers import Patchembedding, PositionalEncoding, Encoder_block


class ViT(nn.Module):
    def __init__(self, res, num_classes, d_ff = 1024, dropout_rate = 0.1, num_layers = 12, num_head = 8, patch_size = 4, in_channel = 1):
        super().__init__()
        self.num_layers = num_layers
        self.patch_embed = Patchembedding(res, patch_size, in_channel)
        self.embed_size = in_channel * patch_size * patch_size
        self.pe = PositionalEncoding(d = self.embed_size)
        self.encoder_block = Encoder_block(embed_dim = self.embed_size, head_num = num_head, p_dropout = dropout_rate, d_ff = d_ff)

        self.encoder = nn.ModuleList([
            self.encoder_block
            for _ in range(num_layers)
        ])

        self.linear_head = nn.LazyLinear(num_classes)


    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pe(x)

        for layer_block in self.encoder:
            x = layer_block(x)

        x = self.linear_head(x[:, 0])

        return x