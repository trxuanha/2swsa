# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by czifan (czifan@pku.edu.cn)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import h5py
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset

class SurvivalDataset(Dataset):
    ''' The dataset class performs loading data from .h5 file. '''
    def __init__(self, file_path, is_train, target, event, format_file):
        ''' Loading data from .h5 file based on (h5_file, is_train).

        :param h5_file: (String) the path of .h5 file
        :param is_train: (bool) which kind of data to be loaded?
                is_train=True: loading train data
                is_train=False: loading test data
        '''
        
        if(format_file == 'csv'):
                    # loads data        
            self.X, self.e, self.y, self.covariates = self._read_csv_file(file_path, target, event)
        if(format_file == 'h5'):
            self.X, self.e, self.y, self.covariates = self._read_h5_file(file_path, is_train)
        
        # normalizes data

        #self._normalize()

        print('=> load {} samples'.format(self.X.shape[0]))

    def _read_h5_file(self, h5_file, is_train):
        ''' The function to parsing data from .h5 file.

        :return X: (np.array) (n, m)
            m is features dimension.
        :return e: (np.array) (n, 1)
            whether the event occurs? (1: occurs; 0: others)
        :return y: (np.array) (n, 1)
            the time of event e.
        '''
        split = 'train' if is_train else 'test'
        with h5py.File(h5_file, 'r') as f:
            X = f[split]['x'][()]
            e = f[split]['e'][()].reshape(-1, 1)
            y = f[split]['t'][()].reshape(-1, 1)
        return X, e, y, None


    def _read_csv_file(self, csv_file, target, event):
    
    
        data_in = pd.read_csv(csv_file, encoding = "ISO-8859-1", engine='python')        
        covariates = list(set(data_in.columns) - {target, event})
        
        X = data_in[covariates].to_numpy().astype(np.float32)
        e = data_in[event].to_numpy().reshape(-1, 1)
        e = e.astype(np.float32)
        
        y = data_in[target].to_numpy().reshape(-1, 1)
        y = y.astype(np.float32)
        
        
        
        return X, e, y, covariates
                
    
    def _get_X(self):
        return self.X
    
    def _normalize(self):
        ''' Performs normalizing X data. '''
        self.X = (self.X-self.X.min(axis=0)) / (self.X.max(axis=0)-self.X.min(axis=0))

    def __getitem__(self, item):
        ''' Performs constructing torch.Tensor object'''
        # gets data with index of item
        X_item = self.X[item] # (m)
        e_item = self.e[item] # (1)
        y_item = self.y[item] # (1)
        # constructs torch.Tensor object
        X_tensor = torch.from_numpy(X_item)
        e_tensor = torch.from_numpy(e_item)
        y_tensor = torch.from_numpy(y_item)
        return X_tensor, y_tensor, e_tensor

    def __len__(self):
        return self.X.shape[0]
        
    def __X_dim__(self):
        return self.X.shape[1]  

    def __get_covariates__(self):
        return self.covariates         