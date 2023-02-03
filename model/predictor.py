import torch.nn as nn
import numpy as np
import torch


class Predictor(nn.Module):

    def __init__(self, input_dim, predict_dims, model_init):
        nn.Module.__init__(self)
        
        nn.Module.__init__(self)        
        self.model = self._build_latent_layer(input_dim, predict_dims)
        self.reset_parameters(model_init)
        

    def _getNonlinear(self):
        return nn.LeakyReLU()
        
        
    def _build_latent_layer(self, input_dim, predict_dims):
    
        layers = []
        epsilon = 1e-5
        alpha = 0.9
        dims = predict_dims[:]
        dims.insert(0, input_dim)  
        dims.append(1)
        
        for i in range(len(dims)-1):
            
            layers.append(nn.Linear(dims[i], dims[i+1]))
            #layers.append(nn.Dropout(0.1))
            if(i < (len(dims)-2)):
                #layers.append(nn.BatchNorm1d(dims[i+1], eps=epsilon, momentum=alpha))
                layers.append(self._getNonlinear())
                
            
                
        return nn.Sequential(*layers)
        
        
            
    def reset_parameters(self, model_init):
        for param in self.parameters():
            model_init(param)

    def forward(self, encoded_X):
        return self.model(encoded_X)
