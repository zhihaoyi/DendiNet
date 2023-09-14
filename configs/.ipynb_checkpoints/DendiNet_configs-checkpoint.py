import torch

class Configs():
    def __init__(self):
        self.pred_len = 8               # predicted length
        self.seq_len = 32               # sequence length
        self.label_len = self.pred_len  # equal to pred_len - NOT CHANGEABLE
        self.output_attention = True    # whether return attention
        self.enc_in = 3                 # encoder channel: 'Voltage', 'Current', 'SOC' - NOT CHANGEABLE
        self.d_model = 32 
        self.dropout = 0.1
        self.dec_in = self.enc_in       # decoder channel - NOT CHANGEABLE
        self.n_heads = 2                # number of head
        self.d_ff = None                # fc neurons defualt to 4*d_model
        self.activation = 'relu'        # activation used in NS transformer default to GeLU
        self.e_layers = 2               # number of encoder layer
        self.d_layers = 1               # number of decoder layer
        self.c_out = 1                  # number of prediction (SOC) - NOT CHANGEABLE
        
        # delta tau parameters
        self.p_hidden_dims = [16, 16]   # delta tau d_model
        self.p_hidden_layers = 2        # number of layers in delta tau projector
        
        # inception parameters
        self.kernel_size = 3            # kernel size in inception
        self.causal_conv = False        # whether use causal_conv rather than conv1d in inception module
        self.inception_layers = 2       # number of inception layer
        self.device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')