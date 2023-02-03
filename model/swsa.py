import torch.nn as nn
import numpy as np
from .encoder import Encoder
from .predictor import Predictor
from utils import helpers
import torch
import pandas as pd
from torch import linalg as LA



class SWSA(nn.Module):
    def __init__(self, input_dim, sampleNbr, config, model_init):
        nn.Module.__init__(self)

        
        self.weight_decay  = 1e-4
        encode_dims = config['network']['encode_dims']
        
        self.encoder = Encoder(input_dim, encode_dims, model_init)
        
        predict_dims = config['network']['predict_dims']
        self.predictor = Predictor(encode_dims[-1], predict_dims, model_init)
        
        self.con_W = nn.Parameter(torch.ones([sampleNbr,1]))
        self.cen_W = nn.Parameter(torch.ones([sampleNbr,1]))
        
        
    def reset_parameters(self, model_init):
        for param in self.parameters():
            model_init(param)
            
        self.con_W = nn.Parameter(torch.ones([self.con_W.shape[0],1]))
        self.cen_W = nn.Parameter(torch.ones([self.cen_W.shape[0],1]))
            
    def forward(self, X):
        
        encoding_var = self.encoder(X) 
        risk_pred = self.predictor(encoding_var)
        #return risk_pred, self.con_weight, self.cen_weight, encoding_var
        
        return risk_pred, encoding_var
                    
        
    def do_train(self, train_loader, optimizer, epoch=0):
        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        # enable train model for turning on dropout, batch norm etc
        self.train()
        
        epoch_loss = 0

        ## Just one loop
        for X, y, e in train_loader:
        
            optimizer.zero_grad()              
            # loss values
            cox_loss, con_weight_loss, cen_weight_loss, encoding_var, conW, cenW = self._compute_loss(X, y, e)
   
            anpha = 0.1       
            beta = 1
        
            loss = cox_loss + anpha*con_weight_loss + beta*cen_weight_loss
            
            loss.backward()  # compute gradients
            optimizer.step()  # update model paramters 
            epoch_loss += loss.item() # sum loss value
            
        size = len(train_loader)
        
        
        return cox_loss.item(), con_weight_loss.item(), cen_weight_loss.item(), encoding_var, conW, cenW, e, X

                   

    def _get_con_weight(self):
        
        return self._nonline_transform_for_weight(self.con_W) 
        
        
    def _get_cen_weight(self):
        
        return self._nonline_transform_for_weight(self.cen_W) 

        
        
    def _get_all_weight(self):
        return self._nonline_transform_for_weight(self.cen_W*self.con_W) 

        
    def _nonline_transform_for_weight(self, x ):
    
        return torch.sigmoid(x)
        
    
    def _get_con_weight_loss(self, inder_X):
    
        der_X = nn.functional.normalize(inder_X, dim=0)
        X_mean = torch.mean(der_X, 0, True)
        
        ## binarise encoded X  
        X_bin = torch.ones(der_X.shape[0], der_X.shape[1])
        
        X_bin[(der_X - X_mean) <= 0] = 0

        
        n,p = der_X.shape
        W = self._get_con_weight()
        
        loss_balancing = 0
        
        for j in range(p):
        
            mask = torch.ones(der_X.shape[0], der_X.shape[1])            
            mask[:, j] = 0
            X_j = der_X*mask #  X_j[:, j] = 0
            
            
            I = X_bin[:,j:j+1]
            
            balancing_j =torch.divide(torch.matmul(X_j.T,W*I),torch.maximum(torch.sum(W*I),torch.tensor(0.05))) - torch.divide(torch.matmul(X_j.T,W*(1-I)),torch.maximum(torch.sum(W*(1-I)),torch.tensor(0.05)))
            loss_balancing += LA.norm(balancing_j,ord=2)
            
            
        loss_weight_sum = (torch.sum(W)-n)**2
        loss_weight_l2 = torch.sum((W)**2)
        total_loss = loss_balancing + 0* loss_weight_sum + 0.* loss_weight_l2
        
        return total_loss
    

    def _get_cen_weight_loss(self, der_X, e):
    
        X = nn.functional.normalize(der_X, dim=0)
        
        W = self._get_cen_weight()*self._get_con_weight()
        
        
        balancing = torch.divide(torch.matmul(X.T,W*e),torch.maximum(torch.sum(W*e),torch.tensor(0.1))) - torch.divide(torch.matmul(X.T,W*(1-e)),torch.maximum(torch.sum(W*(1-e)),torch.tensor(0.1)))
            
        loss_weight_sum = (torch.sum(W)-X.shape[0])**2
        loss_weight_l2 = torch.sum((W)**2)
        total_loss = LA.norm(balancing,ord=2) + 0* loss_weight_sum + 0.* loss_weight_l2
        
        return total_loss



    def _get_cox_loss(self, risk_pred, y, e):
    

        weight = self._get_con_weight()*self._get_cen_weight()
        #weight_loss = self._get_all_weight()
        
        
        #weight_loss =  1
        
        #print(weight_loss)
        
        
        
        sort_idx = torch.argsort(y[:, 0], descending = True)
        y = y[sort_idx]
        e = e[sort_idx]        
        risk_pred = risk_pred[sort_idx] ## harzard function
        hazard_ratio = torch.exp(risk_pred) ## risk score = hazard ration
        log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0))
        uncensored_likelihood = risk_pred - log_risk
        censored_likelihood = weight*uncensored_likelihood * e        
        num_observed_events = torch.sum(e)
        neg_log_loss = -torch.sum(censored_likelihood) #/ num_observed_events
        
           
        
        
        
        '''
        mask = torch.ones(y.shape[0], y.shape[0])
        mask[(y.T - y) > 0] = 0
        log_loss = torch.exp(risk_pred) * mask
        log_loss = torch.sum(log_loss, dim=0) / torch.sum(mask, dim=0)
        log_loss = torch.log(log_loss).reshape(-1, 1)
        neg_log_loss = -torch.sum(weight_loss*(risk_pred-log_loss) * e) / torch.sum(e)
        
        '''
        
        
        reg_loss = 0
        
        
        for name, w in self.named_parameters():
        
        
            if ('weight' in name ):
                reg_loss = reg_loss + torch.norm(w, p=1)
        reg_loss = self.weight_decay * reg_loss
        
        
        #print(reg_loss)
        
        return neg_log_loss + reg_loss

        
    def _compute_loss(self, X, y, e):
        
                
        risk_pred, encoding_var = self.forward(X=X)
        cox_loss = self._get_cox_loss(risk_pred, y, e)
        
        con_weight_loss = self._get_con_weight_loss(encoding_var)
        #cen_weight_loss = self._get_cen_weight_loss(X, e)
        cen_weight_loss = self._get_cen_weight_loss(encoding_var, e)
        
        
        conW = self._get_con_weight()
        
        cenW = self._get_cen_weight()
        
        return cox_loss, con_weight_loss, cen_weight_loss, encoding_var, conW, cenW
    