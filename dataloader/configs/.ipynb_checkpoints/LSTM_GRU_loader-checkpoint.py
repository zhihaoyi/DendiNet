import torch

class Configs():
    def __init__(self):
        self.window_size = 33
        self.stride = 1
        self.enc_seq_len = 32
        self.trg_seq_len = 1
        self.dec_seq_len = 1
        self.transformer = False