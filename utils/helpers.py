import torch.nn as nn
import torch
import numpy as np
from scipy import stats
import random
import configparser
import os
import os.path
from os import path
import pandas as pd

from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index

    
def estimateUplift(estimatedImprovements, outcomeName, sortbyabs=False):
    
    if(sortbyabs):
        estimatedImprovements['ABS_Improvement'] = estimatedImprovements['LIFT_SCORE'].abs()
        estimatedImprovements.sort_values(by=['ABS_Improvement'], ascending = [False], inplace=True, axis=0)
    else:
        estimatedImprovements.sort_values(by=['LIFT_SCORE'], ascending = [False], inplace=True, axis=0)
    estimatedImprovements = estimatedImprovements.reset_index(drop=True) 
    
    Sum_Y_Follow_Rec    = np.array([])
    Sum_Nbr_Follow_Rec    = np.array([])
    Sum_Y_Not_Follow_Rec    = np.array([])
    Sum_Nbr_Not_Follow_Rec    = np.array([])
    Improvement    = np.array([])
    total_Y_Follow_Rec  = 0
    total_Nbr_Follow_Rec = 0
    total_Y_Not_Follow_Rec  = 0
    total_Nbr_Not_Follow_Rec = 0    
    for index, individual in estimatedImprovements.iterrows():
        improvementTemp = 0
        if(individual['FOLLOW_REC'] == 1):
            total_Nbr_Follow_Rec = total_Nbr_Follow_Rec + 1
            total_Y_Follow_Rec = total_Y_Follow_Rec + individual[outcomeName]
        else:
            total_Nbr_Not_Follow_Rec = total_Nbr_Not_Follow_Rec + 1
            total_Y_Not_Follow_Rec = total_Y_Not_Follow_Rec + individual[outcomeName]   
        Sum_Nbr_Follow_Rec = np.append (Sum_Nbr_Follow_Rec, total_Nbr_Follow_Rec)
        Sum_Y_Follow_Rec = np.append (Sum_Y_Follow_Rec, total_Y_Follow_Rec)
        Sum_Nbr_Not_Follow_Rec = np.append (Sum_Nbr_Not_Follow_Rec, total_Nbr_Not_Follow_Rec)
        Sum_Y_Not_Follow_Rec = np.append (Sum_Y_Not_Follow_Rec, total_Y_Not_Follow_Rec)
        if(total_Nbr_Follow_Rec == 0 or total_Nbr_Not_Follow_Rec == 0 ):
            if(total_Nbr_Follow_Rec > 0):
                improvementTemp = (total_Y_Follow_Rec/total_Nbr_Follow_Rec)
            else:
                improvementTemp = 0
        else:
            improvementTemp = (total_Y_Follow_Rec/total_Nbr_Follow_Rec) - (total_Y_Not_Follow_Rec/total_Nbr_Not_Follow_Rec)   
        Improvement = np.append (Improvement, improvementTemp)
    ser = pd.Series(Sum_Nbr_Follow_Rec)
    estimatedImprovements['N_TREATED'] = ser
    ser = pd.Series(Sum_Y_Follow_Rec)
    estimatedImprovements['Y_TREATED'] = ser
    ser = pd.Series(Sum_Nbr_Not_Follow_Rec)
    estimatedImprovements['N_UNTREATED'] = ser
    ser = pd.Series(Sum_Y_Not_Follow_Rec)
    estimatedImprovements['Y_UNTREATED'] = ser  
    ser = pd.Series(Improvement)
    estimatedImprovements['UPLIFT'] = ser
    return estimatedImprovements



def areaUnderCurve(models, modelNames):
    modelAreas = []
    for modelName in modelNames:   
        area = 0
        tempModel = models[models['model'] == modelName].copy()
        tempModel.reset_index(drop=True, inplace=True)       
        for i in range(1, len(tempModel)):  # df['A'].iloc[2]
            delta = tempModel['n'].iloc[i] - tempModel['n'].iloc[i-1]
            y = (tempModel['uplift'].iloc[i] + tempModel['uplift'].iloc[i-1]  )/2
            area += y*delta
        modelAreas.append(area)  
    return modelAreas

def estimateQiniCurve(estimatedImprovements, outcomeName, modelName):

    ranked = pd.DataFrame({})
    ranked['uplift_score'] = estimatedImprovements['Improvement']
    ranked['NUPLIFT'] = estimatedImprovements['UPLIFT']
    ranked['FollowRec'] = estimatedImprovements['FollowRec']
    ranked[outcomeName] = estimatedImprovements[outcomeName]
    ranked['countnbr'] = 1
    ranked['n'] = ranked['countnbr'].cumsum() / ranked.shape[0]
    uplift_model, random_model = ranked.copy(), ranked.copy()
    C, T = sum(ranked['FollowRec'] == 0), sum(ranked['FollowRec'] == 1)
    ranked['CR'] = 0
    ranked['TR'] = 0
    ranked.loc[(ranked['FollowRec'] == 0)
                            &(ranked[outcomeName]  == 1),'CR'] = ranked[outcomeName]
    ranked.loc[(ranked['FollowRec'] == 1)
                            &(ranked[outcomeName]  == 1),'TR'] = ranked[outcomeName]
    ranked['NotFollowRec'] = 1
    ranked['NotFollowRec'] = ranked['NotFollowRec']  - ranked['FollowRec'] 
    ranked['NotFollowRecCum'] = ranked['NotFollowRec'].cumsum() 
    ranked['FollowRecCum'] = ranked['FollowRec'].cumsum() 
    ranked['CR/C'] = ranked['CR'].cumsum() / ranked['NotFollowRec'].cumsum()    
    ranked['TR/T'] = ranked['TR'].cumsum() / ranked['FollowRec'] .cumsum()
    # Calculate and put the uplift into dataframe
    uplift_model['uplift'] = round((ranked['TR/T'] - ranked['CR/C'])*ranked['n'] ,5)
    uplift_model['uplift'] = round((ranked['NUPLIFT'])*ranked['n'] ,5)
    uplift_model['grUplift'] =ranked['NUPLIFT']
    random_model['uplift'] = round(ranked['n'] * uplift_model['uplift'].iloc[-1],5)
    ranked['uplift']  = ranked['TR/T'] - ranked['CR/C']
    # Add q0
    q0 = pd.DataFrame({'n':0, 'uplift':0}, index =[0])
    uplift_model = pd.concat([q0, uplift_model]).reset_index(drop = True)
    random_model = pd.concat([q0, random_model]).reset_index(drop = True)  
    # Add model name & concat
    uplift_model['model'] = modelName
    random_model['model'] = 'Random model'
    return uplift_model

    
def genQiniDataML(folderName, fold, prefileName, postfileName, modelName, outcomeName, event):
    improvementMtreeModels = []
    for fileCount in range (1, fold + 1):
        fPath = os.path.join(folderName, prefileName + str(fileCount) + postfileName + '.csv') 
        Sdataset0 = pd.read_csv(fPath,  encoding = "ISO-8859-1", engine='python')
        Sdataset0 = Sdataset0[Sdataset0[event] == 1]
        if (not('Improvement' in Sdataset0.columns)):
            Sdataset0 ['Improvement'] =  Sdataset0 ['LIFT_SCORE'] 
        if (not('FollowRec' in Sdataset0.columns)):
            Sdataset0 ['FollowRec'] = Sdataset0 ['FOLLOW_REC']           
        newImprovement = estimateQiniCurve(Sdataset0, outcomeName, modelName)
        improvementMtreeModels.append(newImprovement)   
    improvementMtreeCurves = pd.DataFrame({})
    improvementMtreeCurves['n'] = improvementMtreeModels[0]['n']
    improvementMtreeCurves['model'] = improvementMtreeModels[0]['model']   
    icount = 1
    modelNames = []
    groupModelNames = []   
    for eachM in improvementMtreeModels:
        improvementMtreeCurves['uplift' + str(icount)] = eachM['uplift']
        modelNames.append('uplift' + str(icount))
        improvementMtreeCurves['grUplift' + str(icount)] = eachM['grUplift']
        groupModelNames.append('grUplift' + str(icount))
        icount = icount + 1   
    improvementMtreeCurves['uplift'] = improvementMtreeCurves[modelNames].mean(axis=1)
    improvementMtreeCurves['grUplift'] = improvementMtreeCurves[groupModelNames].mean(axis=1)
    return improvementMtreeCurves
    
    
def getAUUCTopGroup(FolderLocation, fold, prefileName, postfileName, outcomeName, event):
    improvementMtreeModels = []
    for fileCount in range (1, fold + 1):
        improveFilePath = os.path.join(FolderLocation, prefileName + str(fileCount) + postfileName + '.csv')
        
        if(not (path.exists(improveFilePath) )):
            print('Not exist!!!!')
            print(improveFilePath)
            continue
        results = pd.read_csv(improveFilePath,  encoding = "ISO-8859-1", engine='c')
        results = results.dropna()
        results = results[results[event] == 1]
        if (not('Improvement' in results.columns)):
            results ['Improvement'] =  results ['LIFT_SCORE'] 
        if (not('FollowRec' in results.columns)):
            results ['FollowRec'] = results ['FOLLOW_REC']
                        
        newImprovement = estimateQiniCurve(results, outcomeName, 'Tree')
        improvementMtreeModels.append(newImprovement)   
    improvementMtreeCurves = pd.DataFrame({})
    improvementMtreeCurves['n'] = improvementMtreeModels[0]['n']
    improvementMtreeCurves['model'] = improvementMtreeModels[0]['model']
    icount = 1
    modelNames = []
    groupModelNames = []
    for eachM in improvementMtreeModels:
        improvementMtreeCurves['uplift' + str(icount)] = eachM['uplift']
        modelNames.append('uplift' + str(icount))
        improvementMtreeCurves['grUplift' + str(icount)] = eachM['grUplift']
        groupModelNames.append('grUplift' + str(icount))
        icount = icount + 1  
    improvementMtreeCurves['uplift'] = improvementMtreeCurves[modelNames].mean(axis=1)    
    improvementMtreeCurves['grUplift'] = improvementMtreeCurves[groupModelNames].mean(axis=1)
    improvementModels = pd.DataFrame({})
    improvementModels = improvementModels.append(improvementMtreeCurves)
    ## convert to percent
    improvementModels['uplift'] = improvementModels['uplift']* 100
    improvementModels['grUplift'] = improvementModels['grUplift']* 100
    curveNames = ['Tree']   
    improvementModels['uplift'] = improvementModels['uplift'].fillna(0)
    estimateAres = areaUnderCurve(improvementModels, curveNames)
    return  estimateAres[0]/100, improvementModels

def getAUUCTopGroupSpec(FolderLocation, fileCount, prefileName, postfileName, outcomeName, event):
    improvementMtreeModels = []
    
    improveFilePath = os.path.join(FolderLocation, prefileName + str(fileCount) + postfileName + '.csv')
    if(not (path.exists(improveFilePath) )):
        print('Not exist!!!!')
        print(improveFilePath)
        return
    results = pd.read_csv(improveFilePath,  encoding = "ISO-8859-1", engine='python')
    
    results = results[results[event] == 1]
    if (not('Improvement' in results.columns)):
        results ['Improvement'] =  results ['LIFT_SCORE'] 
    if (not('FollowRec' in results.columns)):
        results ['FollowRec'] = results ['FOLLOW_REC']
                    
    newImprovement = estimateQiniCurve(results, outcomeName, 'Tree')
    improvementMtreeModels.append(newImprovement)   
    improvementMtreeCurves = pd.DataFrame({})
    improvementMtreeCurves['n'] = improvementMtreeModels[0]['n']
    improvementMtreeCurves['model'] = improvementMtreeModels[0]['model']
    icount = 1
    modelNames = []
    groupModelNames = []
    for eachM in improvementMtreeModels:
        improvementMtreeCurves['uplift' + str(icount)] = eachM['uplift']
        modelNames.append('uplift' + str(icount))
        improvementMtreeCurves['grUplift' + str(icount)] = eachM['grUplift']
        groupModelNames.append('grUplift' + str(icount))
        icount = icount + 1  
    improvementMtreeCurves['uplift'] = improvementMtreeCurves[modelNames].mean(axis=1)    
    improvementMtreeCurves['grUplift'] = improvementMtreeCurves[groupModelNames].mean(axis=1)
    improvementModels = pd.DataFrame({})
    improvementModels = improvementModels.append(improvementMtreeCurves)
    ## convert to percent
    improvementModels['uplift'] = improvementModels['uplift']* 100
    improvementModels['grUplift'] = improvementModels['grUplift']* 100
    curveNames = ['Tree']   
    improvementModels['uplift'] = improvementModels['uplift'].fillna(0)
    estimateAres = areaUnderCurve(improvementModels, curveNames)
    return  estimateAres[0]/100
    
def findRec(currentRow):
    separation = '_vv_'
    bestFactor = currentRow['TREATMENT_NAME']    
    res = bestFactor.split(separation)    
    facName = res[0]
    facVal = res[1]    
    return 1 if int(float(currentRow[facName])) == int(float(facVal)) else 0       
    
def c_index_np(risk_pred, y, e):
    ''' Performs calculating c-index

    :param risk_pred: (np.ndarray or torch.Tensor) model prediction
    :param y: (np.ndarray or torch.Tensor) the times of event e
    :param e: (np.ndarray or torch.Tensor) flag that records whether the event occurs
    :return c_index: the c_index is calculated by (risk_pred, y, e)
    '''
    
    if not isinstance(y, np.ndarray):
        y = y.numpy()
    if not isinstance(risk_pred, np.ndarray):
        risk_pred = risk_pred.numpy()
    if not isinstance(e, np.ndarray):
        e = e.numpy()
    return concordance_index(y, risk_pred, e)
    
    

def c_index(risk_pred, y, e):
    ''' Performs calculating c-index

    :param risk_pred: (np.ndarray or torch.Tensor) model prediction
    :param y: (np.ndarray or torch.Tensor) the times of event e
    :param e: (np.ndarray or torch.Tensor) flag that records whether the event occurs
    :return c_index: the c_index is calculated by (risk_pred, y, e)
    '''
    if not isinstance(y, np.ndarray):
        y = y.detach().cpu().numpy()
    if not isinstance(risk_pred, np.ndarray):
        risk_pred = risk_pred.detach().cpu().numpy()
    if not isinstance(e, np.ndarray):
        e = e.detach().cpu().numpy()
    return concordance_index(y, risk_pred, e)
    
    
def adjust_learning_rate(optimizer, epoch, lr, lr_decay_rate):
    ''' Adjusts learning rate according to (epoch, lr and lr_decay_rate)

    :param optimizer: (torch.optim object)
    :param epoch: (int)
    :param lr: (float) the initial learning rate
    :param lr_decay_rate: (float) learning rate decay rate
    :return lr_: (float) updated learning rate
    '''
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr / (1+epoch*lr_decay_rate)
    return optimizer.param_groups[0]['lr']
    
def read_config(ini_file):
    ''' Performs read config file and parses it.

    :param ini_file: (String) the path of a .ini file.
    :return config: (dict) the dictionary of information in ini_file.
    '''
    def _build_dict(items):
        return {item[0]: eval(item[1]) for item in items}
    # create configparser object
    cf = configparser.ConfigParser()
    # read .ini file
    cf.read(ini_file)
    config = {sec: _build_dict(cf.items(sec)) for sec in cf.sections()}
    return config

    
### calculate the number of trainable parameters in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class uniform_initializer(object):
    def __init__(self, stdv):
        self.stdv = stdv

    def __call__(self, tensor):
        nn.init.uniform_(tensor, -self.stdv, self.stdv)  



def gData(data):
    tensor = data
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    if use_cuda:
        tensor = tensor.cuda()
    return tensor


def gVar(data):
    return gData(data)
    


        
