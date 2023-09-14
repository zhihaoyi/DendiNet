import torch
import torch.nn as nn
import copy
import os
import sys

from model.utils.ns_Transformer import NS_transformer
from model.utils.freq_inception import FreqInceptionLayer, FreqInceptionBlock


class DendiNet(nn.Module):
    def __init__(self, configs):
        super(DendiNet, self).__init__()
        self.configs = copy.deepcopy(configs)
        
        # NS transformer
        self.NS_transformer = NS_transformer(self.configs).to(self.configs.device)
        
        # inception
        self.FreqInceptionLayer = FreqInceptionLayer(self.configs.enc_in,
                                                     self.configs.d_model, 
                                                     kernel_size=self.configs.kernel_size,
                                                     dropout=self.configs.dropout,
                                                     causal_conv=self.configs.causal_conv)
        
        self.FreqInceptionBlock = FreqInceptionBlock(freq_inception_layer=self.FreqInceptionLayer, 
                                                     num_layer=self.configs.inception_layers)
       
        # add trainable weight to NS transformer and inception
        self.weight = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.weight.data.fill_(0)
        self.relu = nn.ReLU()
        

    def getInput4NS(self, seq, pred_len, device=None):
        assert seq.shape[1] > pred_len , 'Arg-2 predicted length must be less than Arg-1 sequence length'
        B, S, D = seq.shape[0], seq.shape[1], seq.shape[2]
        dec_input = torch.zeros(B, pred_len,D).float().to(device)
        dec_input = torch.cat([seq[:, -pred_len:, :], dec_input], dim=1).float()
        
        return seq.float(), dec_input.float()
    

    def forward(self, seq):    
        B, S, D = seq.shape[0], seq.shape[1], seq.shape[2]
        
        # transformer signals to frequency domain
        fourier = torch.fft.rfft(seq, dim=1)
        frequencies = torch.fft.rfftfreq(S)
        
        # split original signals into two parts
        fourier[:, frequencies>0.5, :] = 0              # cutoff shreshold = 0.5
        low_signal = torch.fft.irfft(fourier, dim=1)    # low frequency goes to NS transformer
        high_signal = seq - low_signal                  # high frequency goes to inception
        
        # for NS transforer, get encoder and decoder inputs
        enc_input, dec_input = self.getInput4NS(low_signal, self.configs.pred_len, device=self.configs.device)
        
        # pass inputs to NS transformer
        low_signal, attns = self.NS_transformer(enc_input, dec_input)
        
        # pass input to inception
        high_signal = self.FreqInceptionBlock(high_signal)[:, -self.configs.pred_len:, :] 
        
        # weight outputs from NS transformer and inception
        weight = self.relu(self.weight)
        out = (1 - weight) * low_signal + weight * high_signal
        
        return out[:, :, -1], attns # last feature row -> soc