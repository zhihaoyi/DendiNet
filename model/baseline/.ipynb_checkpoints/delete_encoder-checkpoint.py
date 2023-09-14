'''
reused code from:
https://github.com/mlpotter/Transformer_Time_Series/blob/master/Transformer_Decoder_nologsparse.ipynb
'''
import torch
import torch.nn.functional as F

from model.utils.positional_encoder import PositionalEncoder
from model.utils.ts_encoder_mask_generator import ts_encoder_mask_generator
from model.utils.causal_conv import CausalConv1d
from model.utils.high_mixer import HighMixer
from model.utils.low_mixer import LowMixer

class EncoderTimeSeries(torch.nn.Module):
    def __init__(self, enc_seq_len, trg_seq_len, in_channels, output_channels, kernel_size, stride=1, dilation=1, nhead=8, num_layers=3, inception=False):
        super(EncoderTimeSeries,self).__init__()
        self.enc_seq_len = enc_seq_len
        self.trg_seq_len = trg_seq_len
        self.output_channels = output_channels
        self.inception = inception
        
        # mask genereation [(enc_seq_len+trg_seq_len)x(enc_seq_len+trg_seq_len)]
        self.att_mask = ts_encoder_mask_generator(enc_seq_len, trg_seq_len)
        
        # layers
        #self.input_embedding = torch.nn.Linear(in_channels, output_channels)
        self.input_embedding = CausalConv1d(in_channels, output_channels, kernel_size, dilation=1)
        if inception is True:
            self.high_freq_embedding = HighMixer(output_channels//2, kernel_size=kernel_size, stride=stride, dilation=dilation)
            self.low_freq_embedding = LowMixer(output_channels//2, kernel_size=kernel_size, stride=stride, dilation=dilation)
            self.conv_fuse = CausalConv1d(int(1.5*output_channels), output_channels, kernel_size=3, stride=1)
        self.pos_embedding = PositionalEncoder(d_model=output_channels)
        self.decode_layer = torch.nn.TransformerEncoderLayer(d_model=output_channels, nhead=nhead)
        self.transformer_decoder = torch.nn.TransformerEncoder(self.decode_layer, num_layers=num_layers)
        self.fc = torch.nn.Linear(output_channels, 1)
        
        
    def forward(self, x, y, device=None):
        '''
        x [B, prev_time_stamp, C]
        y [B, target_time_stamp, C]
        '''
        z = torch.cat((x, y), 1) # (B, L, C)
        y = z[:, :, -1].unsqueeze(-1) # features: [B, L，1]
        x = z[:, :, :-1] # target: [B, L，C-1]
 
        #z_embedding = self.input_embedding(z) # [B, L, d]
        z_embedding = self.input_embedding(z.permute(0, 2, 1)) # [B, d, L]
        
        if self.inception is True:
            high_freq_embedding = self.high_freq_embedding(z_embedding[:, :self.output_channels//2, :]) # [B, d, L]
            low_freq_embedding = self.low_freq_embedding(z_embedding[:, self.output_channels//2:, :]) # [B, d/2, L]
            freq_embedding = torch.cat((high_freq_embedding, low_freq_embedding), 1) # [B, 1.5d, L]
            pos_embeddings = self.conv_fuse(freq_embedding)
        pos_embeddings = self.pos_embedding(z_embedding.permute(0, 2, 1)).permute(1, 0, 2) # [L, B, d]
        transformer_embedding = self.transformer_decoder(pos_embeddings, self.att_mask.to(device)) # [L, B, d]
        output = self.fc(transformer_embedding.permute(1, 0, 2)) # [B, L, 1]
        
        return output[:,-self.trg_seq_len:].squeeze(-1), y[:,-self.trg_seq_len:].squeeze(-1)