import torch
import torch.nn as nn

from model.utils.causal_conv import CausalConv1d
from model.utils.Embed import DataEmbedding


class FreqInceptionLayer(nn.Module):
    def __init__(self, in_channel, d_model, kernel_size=3, dropout=0.1, causal_conv=False):
        super().__init__()

        self.conv_in = conv_in = int(0.25*d_model)
        self.pool_in = pool_in = int(0.25*d_model)
        self.conv_dim = conv_dim = d_model
        self.pool_dim = pool_dim = d_model
        padding_val = int((pool_dim-pool_dim+kernel_size-1)/2) # make sure shape before conv = shape after conv
        
        # embedding
        self.data_embedding = DataEmbedding(in_channel, int(d_model*0.75))
        
        # pipeline 1
        if causal_conv is False:
            self.conv1 = nn.Conv1d(conv_in, conv_dim, kernel_size=1, stride=1, padding=padding_val)
            self.proj1 = nn.Conv1d(conv_dim, conv_dim, kernel_size=kernel_size, stride=1)
        else:
            self.conv1 = CausalConv1d(conv_in, conv_dim, kernel_size=1, stride=1)
            self.proj1 = CausalConv1d(conv_dim, conv_dim, kernel_size=kernel_size, stride=1)
        self.mid_gelu1 = nn.GELU()
        
        # pipeline 2
        self.Maxpool = nn.MaxPool1d(kernel_size, stride=1, padding=padding_val)
        if causal_conv is False:
            self.proj2 = nn.Conv1d(pool_in, pool_dim, kernel_size=1, stride=1)
        else:
            self.proj2 = CausalConv1d(pool_in, pool_dim, kernel_size=1, stride=1)
        self.mid_gelu2 = nn.GELU()
        
        # pipeline 3
        self.Avgpool = nn.AvgPool1d(kernel_size, stride=1, padding=padding_val)
        if causal_conv is False:
            self.proj3 = nn.Conv1d(pool_in, pool_dim, kernel_size=kernel_size, stride=1, padding=padding_val)
        else:
            self.proj3 = CausalConv1d(pool_in, pool_dim, kernel_size=kernel_size, stride=1)
        self.mid_gelu3 = nn.GELU()

        # conv_fuse
        self.bi_lstm = nn.LSTM(3*d_model, in_channel, batch_first=True, num_layers=2, bidirectional=True)
        self.conv_fuse = nn.Linear(2*in_channel, in_channel)
        self.layer_norm = nn.LayerNorm(in_channel, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.gelu4 = nn.GELU()
        self.dropout_2 = nn.Dropout(dropout)
        
        
        
    def forward(self, x):
        '''
        x [B, L, C]
        '''
        # embedding
        x = self.data_embedding(x)                     # [B, L, C]

        # pipeline 1
        x = x.permute(0, 2, 1)                         # [B, C, L]
        p1_x = x[:, :self.conv_in, :]                  # [B, C/3, L]
        p1_x = self.conv1(p1_x)                        # [B, C, L]
        p1_x = self.proj1(p1_x)                        # [B, C, L]
        p1_x = self.mid_gelu1(p1_x)                    # [B, C, L]
        
        # pipeline 2
        p2_x = x[:, self.conv_in:2*self.conv_in, :]    # [B, C/3, L]
        p2_x = self.Maxpool(p2_x)                      # [B, C, L]
        p2_x = self.proj2(p2_x)                        # [B, C, L]
        p2_x = self.mid_gelu2(p2_x)                    # [B, C, L]
        
        # pipeline 3
        p3_x = self.Avgpool(x[:, 2*self.conv_in:, :])  # [B, C/3, L]
        p3_x = self.proj3(p3_x)                        # [B, C, L]
        p3_x = self.mid_gelu3(p3_x)                    # [B, C, L]
        
        # conv fuse
        hx = torch.cat((p1_x, p2_x, p3_x), dim=1)      # [B, 3C, L]
        hx, _ = self.bi_lstm(hx.permute(0, 2, 1))      # [B, L, C]
        hx = self.conv_fuse(hx)                        # [B, L, C]
        hx = self.layer_norm(hx)                       # [B, L, C]
        hx = self.dropout_1(hx)                        # [B, L, C]
        hx = self.gelu4(hx.permute(0, 2, 1))           # [B, C, L]
        hx = self.dropout_2(hx)                        # [B, C, L]

        return hx.permute(0, 2, 1)                     # [B, L, C]
    
    
class FreqInceptionBlock(nn.Module):
    def __init__(self, freq_inception_layer, num_layer=2):
        super().__init__()
        self.layers = [freq_inception_layer for _ in range(num_layer)]
        
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        return x