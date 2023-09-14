import torch

class Configs():
    def __init__(self):
        self.num_features = 3
        self.dec_seq_len = 8
        self.enc_seq_len = 32
        self.trg_seq_len = 8
        self.batch_first = False
        self.dim_val = 32
        self.n_encoder_layers = 3
        self.n_decoder_layers = 2
        self.n_heads = 2
        self.dropout_encoder = 0.2
        self.dropout_decoder = 0.2
        self.dropout_pos_enc = 0.1
        self.dim_feedforward_encoder = 64
        self.dim_feedforward_decoder = 64
        self.num_predicted_features = 1
        
        """
        Args:

            num_features: int, number of input variables. 1 if univariate.

            dec_seq_len: int, the length of the input sequence fed to the decoder

            dim_val: int, aka d_model. All sub-layers in the model produce 
                     outputs of dimension dim_val

            n_encoder_layers: int, number of stacked encoder layers in the encoder

            n_decoder_layers: int, number of stacked encoder layers in the decoder

            n_heads: int, the number of attention heads (aka parallel attention layers)

            dropout_encoder: float, the dropout rate of the encoder

            dropout_decoder: float, the dropout rate of the decoder

            dropout_pos_enc: float, the dropout rate of the positional encoder

            dim_feedforward_encoder: int, number of neurons in the linear layer 
                                     of the encoder

            dim_feedforward_decoder: int, number of neurons in the linear layer 
                                     of the decoder

            num_predicted_features: int, the number of features you want to predict.
                                    Most of the time, this will be 1 because we're
                                    only forecasting FCR-N prices in DK2, but in
                                    we wanted to also predict FCR-D with the same
                                    model, num_predicted_features should be 2.
        """