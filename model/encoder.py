import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, input_dim, encode_dims, model_init):
        nn.Module.__init__(self)        
        self.model = self._build_latent_layer(input_dim, encode_dims)
        self.reset_parameters(model_init)
        
    def _getNonlinear(self):
        return nn.LeakyReLU()
        
        
    def _build_latent_layer(self, input_dim, encode_dims):
    

        
        layers = []
        epsilon = 1e-5
        alpha = 0.9
        dims = encode_dims[:]
        
        dims.insert(0, input_dim)  
        
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.BatchNorm1d(dims[i+1], eps=epsilon, momentum=alpha))
            layers.append(self._getNonlinear())
                
        return nn.Sequential(*layers)
            
    def reset_parameters(self, model_init):
        for param in self.parameters():
            model_init(param)

    def forward(self, X):
                             
        return self.model(X)