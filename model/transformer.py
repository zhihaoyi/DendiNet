import torch
import torch.nn as nn
from model.utils.Transformer_EncDec_trans import Decoder, DecoderLayer, Encoder, EncoderLayer
from model.utils.SelfAttention_Family_trans import FullAttention, AttentionLayer
from model.utils.Embed import DataEmbedding


class Transformer(nn.Module):
    """
    Vanilla Transformer
    """
    def __init__(self, configs):
        super(Transformer, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.device = configs.device
        
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
                        FullAttention(False, attention_dropout=configs.dropout,
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
                        FullAttention(True, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, attention_dropout=configs.dropout, output_attention=False),
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
        
        
    def getInput4NS(self, seq, pred_len, device=None):
        assert seq.shape[1] > pred_len , 'Arg-2 predicted length must be less than Arg-1 sequence length'
        B, S, D = seq.shape[0], seq.shape[1], seq.shape[2]
        dec_input = torch.zeros(B, pred_len,D).float().to(device)
        dec_input = torch.cat([seq[:, -pred_len:, :], dec_input], dim=1).float()
        
        return seq.float(), dec_input.float()
    

    def forward(self, seq):
        enc_input, dec_input = self.getInput4NS(seq, self.pred_len, device=self.device)

        enc_out = self.enc_embedding(enc_input)
        enc_out, attns = self.encoder(enc_out)

        dec_out = self.dec_embedding(dec_input)
        dec_out = self.decoder(dec_out, enc_out)
        

        if self.output_attention:
            return dec_out[:, -self.pred_len:, -1], attns
        else:
            return dec_out[:, -self.pred_len:, -1], None  # [B, L, D]
