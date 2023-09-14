# from sklearn.preprocessing import StandardScaler
import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, random_split


class dataset_pipeline(Dataset):
    def __init__(self, path, test_temp, drive_cycle, configs, convert2soc=True):
        super(dataset_pipeline, self).__init__()
        self.configs = configs
        assert configs.window_size == configs.enc_seq_len + configs.trg_seq_len, 'window_size does not match to enc_seq_len + trg_seq_len'
        
        self.window_size = configs.window_size
        self.stride = configs.stride
        self.enc_seq_len = configs.enc_seq_len
        self.trg_seq_len = configs.trg_seq_len
        self.dec_seq_len = configs.dec_seq_len
        self.transformer = configs.transformer
        
        if configs.transformer is False:
            self.sequence = torch.zeros(0)
            self.target = torch.zeros(0)
        else:
            self.src = torch.zeros(0)
            self.trg = torch.zeros(0)
            self.trg_y = torch.zeros(0)
        
        # collect and combine drive_cycle datasets
        for temp in test_temp:
            file_list = os.listdir(path+temp)
            for drive in drive_cycle:
                for file in file_list:
                    if drive in file:
                        data_frame = pd.read_excel(path+temp+file)
                        num_samples = len(data_frame)
                        # convert Ah to soc in %
                        if convert2soc is True:
                            data_frame['Ah'] = 1.0 + data_frame['Ah']/2.9
                        print(f'{temp[:-1]}: {drive} is loaded, containing {num_samples} samples.')
                    
                        if self.transformer is False:
                            df_sequence, df_target = self.sliding_window(data_frame, self.window_size, self.stride,
                                                                         self.enc_seq_len, self.trg_seq_len,
                                                                         self.dec_seq_len, transformer=self.transformer)
                            self.sequence = torch.cat([self.sequence, df_sequence])
                            self.target = torch.cat([self.target, df_target])
                        else:
                            df_src, df_trg, df_trg_y = self.sliding_window(data_frame, self.window_size, self.stride,
                                                                           self.enc_seq_len, self.trg_seq_len,
                                                                           self.dec_seq_len, transformer=self.transformer)
                            self.src = torch.cat([self.src, df_src])
                            self.trg = torch.cat([self.trg, df_trg])
                            self.trg_y = torch.cat([self.trg_y, df_trg_y])
            print(f'---------- all {temp[:-1]} datasets are uploaded ----------')
        
        if self.transformer is False:
            self._len = len(self.sequence)
        else:
            self._len = len(self.src)
        
        
    def sliding_window(self, data_frame, window_size, stride, enc_seq_len, trg_seq_len, dec_seq_len, noise_prob=0.3, noise_ratio=5e-3, transformer=False):
        num_window = (len(data_frame)  - window_size) // stride + 1
        
        if transformer is False:
            sequence, target = [], []
            for i in range(num_window):
                df_sequence = data_frame.iloc[i*stride:i*stride+enc_seq_len][['Voltage', 'Current', 'Ah']].values.copy()
                df_target = data_frame.iloc[i*stride+enc_seq_len:i*stride+window_size][['Voltage', 'Current', 'Ah']].values.copy()
                
                # noise_prob(%) to add noise
                # noise_sequence = self.Noise(df_sequence, enc_seq_len, noise_prob, noise_ratio)
                # noise_target = self.Noise(df_target, window_size-enc_seq_len, noise_prob, noise_ratio)
                
                sequence.append(df_sequence)# + noise_sequence)
                target.append(df_target)# + noise_target)
            
            sequence = np.array(sequence)  # convert list of arrays to a single NumPy array
            target = np.array(target)  # convert list to NumPy array
            return torch.tensor(sequence).float(), torch.tensor(target).float() 
        else:
            src, trg, trg_y = [], [], []
            for i in range(num_window):
                df_src = data_frame.iloc[i*stride:i*stride+enc_seq_len][['Voltage', 'Current', 'Ah']].values.copy()
                df_trg = data_frame.iloc[i*stride+enc_seq_len-1:i*stride+enc_seq_len+dec_seq_len-1][['Voltage', 'Current', 'Ah']].values.copy()
                df_trg_y = data_frame.iloc[i*stride+enc_seq_len:i*stride+window_size][['Voltage', 'Current', 'Ah']].values.copy()
                
                # noise_prob(%) to add noise
                # noise_src = Noise(df_src, enc_seq_len, noise_prob, noise_ratio)
                # noise_trg = Noise(df_trg, dec_seq_len, noise_prob, noise_ratio)
                # noise_trg_y = Noise(df_trg_y, window_size-enc_seq_len, noise_prob, noise_ratio)
                
                src.append(df_src)#+noise_src)
                trg.append(df_trg)#+noise_trg)
                trg_y.append(df_trg_y)#+noise_trg_y)
                
    
            src = np.array(src)
            trg = np.array(trg)
            trg_y = np.array(trg_y)

            return torch.tensor(src).float(), torch.tensor(trg).float(), torch.tensor(trg_y).float()  
        
    
    def Noise(self, dataframe, window_size, noise_prob, noise_ratio):
        if np.random.rand() < noise_prob:
            avg = np.mean(dataframe, axis=0)
            noise_voltage = np.random.normal(0, noise_ratio*np.abs(avg[0]), (window_size, 3))
            noise_current = np.random.normal(0, noise_ratio*np.abs(avg[1]), (window_size, 3))
            noise_Ah = np.random.normal(0, noise_ratio*np.abs(avg[2]), (window_size, 3))
            
            return np.concatenate((noise_voltage, noise_current, noise_Ah), axis=1)
        else:
            return np.zeros((window_size, 3))
    
        
        
    def __getitem__(self, index):
        if self.transformer is False:
            return self.sequence[index], self.target[index]
        else:
            return self.src[index], self.trg[index], self.trg_y[index]
        
    def __len__(self):
        return self._len