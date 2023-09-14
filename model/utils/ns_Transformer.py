import torch
import torch.nn as nn
import numpy as np
import os
import sys
from statsmodels.tsa.stattools import adfuller

from model.utils.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from model.utils.SelfAttention_Family import DSAttention, AttentionLayer
from model.utils.Embed import DataEmbedding
from model.utils.projector import Projector


class NS_transformer(nn.Module):
    """
    Non-stationary Transformer
    """
    def __init__(self, configs):
        super(NS_transformer, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.output_attention = configs.output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, 
                                           configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model,
                                           configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        DSAttention(False, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        DSAttention(True, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        DSAttention(False, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

        self.tau_learner = Projector(enc_in=configs.enc_in, seq_len=configs.seq_len, hidden_dims=configs.p_hidden_dims, hidden_layers=configs.p_hidden_layers, output_dim=1)
        self.delta_learner = Projector(enc_in=configs.enc_in, seq_len=configs.seq_len, hidden_dims=configs.p_hidden_dims, hidden_layers=configs.p_hidden_layers, output_dim=configs.seq_len)

    def forward(self, x_enc, x_dec, adf=None, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        x_raw = x_enc.clone().detach() 
 
        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach() # [B,1,E]
        x_enc = x_enc - mean_enc # [B,S,E]
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach() # B x 1 x E
        x_enc = x_enc / std_enc # [B,S,E]
        
        tau = self.tau_learner(x_raw, std_enc.to('cuda')).exp()     # B x S x E, B x 1 x E -> B x 1, positive scalar    
        delta = self.delta_learner(x_raw, mean_enc.to('cuda'))      # B x S x E, B x 1 x E -> B x S

        # Model Inference
        enc_out = self.enc_embedding(x_enc) 
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask, tau=tau, delta=delta)
        
        dec_out = self.dec_embedding(x_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask, tau=tau, delta=delta)
       
        # De-normalization
        dec_out = dec_out * std_enc + mean_enc
      
        if self.output_attention is True:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :], None