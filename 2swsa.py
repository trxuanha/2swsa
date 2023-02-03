import os
import sys
import os.path
from os import path
import argparse
import numpy as np
import pandas as pd
import torch.optim as optim
import random
import torch
import time
import logging
from multiprocessing import Pool
import numpy.ma as ma

from model.swsa import SWSA
from utils import helpers, custom_batch, datasets


BASE_DIR = os.getcwd()
     
    
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
    
def wrapperPara(fold, ini_file):

    ini_file_path = os.path.join(BASE_DIR, 'config', ini_file + '.ini')
    config = helpers.read_config(ini_file_path)
    method = '2SWSA'   
    separation = '_vv_'

    
    train_file = os.path.join(BASE_DIR, 'input', config['dataset']['inputname'], config['dataset']['inputname'] +'_train' +'.csv')
        
    log_file = 'logs/args.{}_fold_{}.log'.format(config['dataset']['inputname'], fold)
    logging.basicConfig(filename=log_file, filemode='w', level=logging.DEBUG)
    logging.debug(args) 
    outdir = os.path.join(BASE_DIR, 'output', method, config['dataset']['inputname'])
    outdirmodel = os.path.join(BASE_DIR, 'output', 'SavedModels')
    
    if not os.path.exists(outdir):    
        os.mkdir(outdir)  
                  
    model_init = helpers.uniform_initializer(0.01)
    
    target = config['dataset']['target']
    exVars = config['dataset']['excl_vars']
    event = config['dataset']['event']

    
    train_dataset = datasets.SurvivalDataset(train_file, True, target, event, config['dataset']['format_file'])
      
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_dataset.__len__()) #train_dataset.__len__()

   
   
    model = SWSA(input_dim=train_dataset.__X_dim__(), sampleNbr=train_dataset.__len__(), config=config, model_init=model_init).to(device)
   
    optimizer = eval('optim.{}'.format(config['train']['optimizer']))(
        model.parameters(), lr=config['train']['learning_rate'])
        
    logging.debug('epoch loop')
    
    icount = 0
    
    ######
    covariates = train_dataset.__get_covariates__()
    numbr_test = config['dataset']['test_envir']
    list_of_test_loaders = []
    for i in range (1, numbr_test + 1):
    
        test_file = os.path.join(BASE_DIR, 'input', config['dataset']['inputname'], config['dataset']['inputname'] +'_test' + str(i) +'.csv')
        test_dataset = datasets.SurvivalDataset(test_file, False, target, event, config['dataset']['format_file'])
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=test_dataset.__len__())
            
        list_of_test_loaders.append(test_loader)
            
        
    for epoch in range(1, config['train']['epochs']+1):
    
        lr = helpers.adjust_learning_rate(optimizer, epoch, 
                                  config['train']['learning_rate'],
                                  config['train']['lr_decay_rate'])
                      
        start_time = time.time()
            
        cox_loss, con_weight_loss, cen_weight_loss, encoding_var, con_weight, cen_weight, out_e,_ = model.do_train(train_loader=train_loader, optimizer=optimizer, epoch=epoch)
                                                                            
        end_time = time.time()
        epoch_mins, epoch_sec = epoch_time(start_time=start_time, end_time=end_time)
        time_str = 'end epoch train: {}, Time: {}m {}s'.format(epoch + 1, epoch_mins, epoch_sec)
            
            
        print_train = '\rEpoch: {} Time: {}m {}s \tLoss: Cox({:.8f}) ConW({:.8f}) CenW({:.8f}) \tlr: {:g}'.format(
            epoch, epoch_mins, epoch_sec, cox_loss, con_weight_loss, cen_weight_loss, lr)
            
        if(np.isnan(cox_loss)):
            break
        print(print_train, end='')
                   
    
    print('\r')
    
    ## Do testing
    model.eval()
    test_c = 0
    allcolumns = covariates + [event, target, 'h_x']    
    output = pd.DataFrame(columns = allcolumns)

    results_list = []
    
    icount = 0
    
    allFactors = config['dataset']['compare_factor']
    df_train = pd.read_csv(train_file, encoding = "ISO-8859-1", engine='python')
    
    
    for test_loader in list_of_test_loaders:

        X, y, e  = next(iter(test_loader))

        with torch.no_grad():
            risk_pred, encoding_var = model(X)
            test_c = helpers.c_index(-risk_pred, y, e)
            
            icount += 1
            output = pd.DataFrame(np.hstack((X,e,y,risk_pred)), columns = allcolumns)
            outFilePath = os.path.join(outdir, method + '_' + config['dataset']['inputname']+'_' +  str(icount) + '.csv')   
            output.to_csv(outFilePath, index=False) 
        
        
foldNbr = [1,2,3,4,5] 

foldNbr = [1]  
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='SWPM')
    parser.add_argument('--config', type=str, default= 'None')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for i in range(0, len(foldNbr)):
        wrapperPara(foldNbr[i], args.config)
