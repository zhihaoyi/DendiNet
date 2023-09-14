import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, configs):
        super(LSTM, self).__init__()
        self.num_features = configs.num_features
        self.sequence_length = configs.sequence_length
        self.hidden_unit = configs.hidden_unit
        
        # layers
        self.batchNorm = nn.BatchNorm1d(self.num_features)
        self.lstm = nn.LSTM(self.num_features, self.hidden_unit, batch_first=True, num_layers=1, bidirectional=False)
        self.output = nn.Linear(self.sequence_length * self.hidden_unit, 1)

    def forward(self, input):
        '''input [B, L, C]'''
        input = input.transpose(1, 2).contiguous() # [B, C, L]        
        norm_input = self.batchNorm(input)       
        norm_input = norm_input.transpose(1, 2).contiguous() # [B, L, C]             
        output, _ = self.lstm(norm_input) # [B, L, d]
        output = output.reshape(-1, self.sequence_length*self.hidden_unit) # [B, L*d]
        output = self.output(output) # [B, 1]
        
        return output.squeeze(-1)