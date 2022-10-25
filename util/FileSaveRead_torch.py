# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 14:30:56 2021

@author: new
"""
import sys
import os
sys.path.append('./../')    
sys.path.append('./../../')   
sys.path.append('./../../../')   
sys.path.append('./../../../../')  

import h5py
import torch
class FileSaveRead():
    def __init__():   
        pass
            


    @staticmethod
    def save_torch_model(epoch, model, optimizer,trian_loss,test_loss, dataset, save_file):
         checkpoint = { 'epoch': epoch, 
                       'model_state_dict': model.state_dict(), 
                      'optimizer_state_dict': optimizer.state_dict(),
                      'train_loss': trian_loss,
                      'test_loss': test_loss,
                      'dataset': dataset,
                      } 
    
         torch.save(checkpoint, save_file )       
         
         
    @staticmethod         
    def load_torch_model(model, model_file):
        checkpoint = torch.load(model_file) 
        
        model.load_state_dict(checkpoint['model_state_dict'])  
        trian_loss = checkpoint['train_loss']
        test_loss = checkpoint['test_loss']
        dataset = checkpoint['dataset']
        return model,   trian_loss,test_loss, dataset



 

        
        
        
        