# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 14:40:48 2021

@author: new
"""
import numpy as np
import sys
import copy
sys.path.append('./../')    
sys.path.append('./../../')   
sys.path.append('./../../../')   
sys.path.append('./../../../../')  
import math

class DataNormalized(object):
    
    def __init__():   
        pass
    
    
    @staticmethod
    def normal_1d(vec):
        import math
        a=0.
        for i in range(len(vec)):
            a += vec[i]
        a = a/ len(vec)
        std = 0.
        for i in range(len(vec)):
            std += pow((vec[i]-a),2.)
            vec[i] = vec[i]/a
        std = math.sqrt(std/len(vec))
        
        return vec,a,std    


    @staticmethod
    def multiplicate_list(vec1,vec2):
        res =[]
        for i in range(len(vec1)):
            res.append( vec1[i]*vec2[i])
        return res

    @staticmethod
    def cal_deviation( powcase, powrec):
        import math
        n= len(powcase)
        sumdevsquare = 0
        sumdevavg = 0
        sumscaldevsquare = 0
        powcasett,avgpowcase,stdpowcase = DataNormalized.normal_1d(powcase) 
        powrectt,avgpowrec,stdpowrec = DataNormalized.normal_1d(powrec) 
        sumcc = 0
        sumrmae = 0
        sumavgm = 0
        meandev = []
        ncout = 0
        for i in range(n):
            if powrec[i] > 0: 
                ncout += 1
                sumrmae += abs(powcase[i]/powrec[i]-1. )
                sumdevsquare+= pow(powcase[i]-powrec[i],2)
                sumdevavg += abs(powcase[i]-powrec[i] )
                sumscaldevsquare += pow(powrec[i]/powcase[i]-1.,2)
                sumcc += (powcase[i]-avgpowcase)*(powrec[i]-avgpowrec)
                meandev.append( abs(powrec[i]/powcase[i]-1) )
                sumavgm += pow((avgpowrec+powrec[i] ),2)
    
        rmae = sumrmae/ncout*100
        rrmse = math.sqrt(sumscaldevsquare/ncout)*100
        ef = 1-(sumdevsquare/ sumavgm)                 
        
        return rmae, rrmse, ef
    
    @staticmethod    
    def normalized_list_1d(list_1d):
        kk = len(list_1d)
        list_np = np.array(list_1d)
        list_np = list_np / np.sum(list_np)*kk
        return list(list_np )

    @staticmethod    
    def normalized_list_2d(list_2d):
        list_2d_np = np.array( list_2d)
        list_2d_np = list_2d_np / np.mean(list_2d_np)
        return list_2d_np.tolist()
  
    @staticmethod    
    def normalized_train_in_out(train_in, train_out):
        train_in_np = np.array(train_in)
        train_out_np = np.array(train_out)
        
        train_in_max = np.max( train_in_np,  axis=0)
        train_in_min = np.min( train_in_np,  axis=0)
        
        train_out_max = np.max( train_out_np,  axis=0)
        train_out_min = np.min( train_out_np,  axis=0)     
        
        m, n = np.shape( train_in_np)
        for i in range(n):
            if train_in_max[i]-train_in_min[i] > 0:
                train_in_np[:,i] = (train_in_np[:,i]- train_in_min[i])/(train_in_max[i]-train_in_min[i])
            else:
                train_in_np[:,i] = 1.0
            
            
        m, n = np.shape( train_out)
        for i in range(n):
            if (train_out_max[i]-train_out_min[i] > 0):   
                train_out_np[:,i] = (train_out_np[:,i]- train_out_min[i])/(train_out_max[i]-train_out_min[i])        
            else:
                train_out_np[:,i]  = 1.0

        return train_in_np.tolist(), train_out_np.tolist(), train_in_max.tolist(), train_in_min.tolist(), train_out_max.tolist(), train_out_min.tolist()
        
    @staticmethod
    def list2d_to_1d(list2d):
        list1d = []
        for i in range(len(list2d)):
            for j in range(len(list2d[i])):
                list1d.append(list2d[i][j])
        return list1d
    
    @staticmethod    
    def normalized_train_lstm(train_in, train_out):
        train_in2d = DataNormalized.list2d_to_1d(train_in)
        train_out2d = DataNormalized.list2d_to_1d(train_out)

        train_in_np = np.array(train_in2d)
        train_out_np = np.array(train_out2d)
        
        print(np.shape(train_in_np), np.shape(train_out_np))
        
        train_in_max = np.max( train_in_np,  axis=0)
        train_in_min = np.min( train_in_np,  axis=0)
        
        train_out_max = np.max( train_out_np,  axis=0)
        train_out_min = np.min( train_out_np,  axis=0)     
        
        for i in range(len(train_in)):
            for j in range(len(train_in[0])):
                for k in range(len(train_in[0][0])):
                    if train_in_max[k]-train_in_min[k] > 0:
                        train_in[i][j][k] = (train_in[i][j][k]- train_in_min[k])/(train_in_max[k]-train_in_min[k])
                    else:
                        train_in[i][j][k] = 1.0
        
        for i in range(len(train_out)):
            for j in range(len(train_out[0])):
                for k in range(len(train_out[0][0])):
                    if (train_out_max[k]-train_out_min[k] > 0):   
                        train_out[i][j][k] = (train_out[i][j][k]- train_out_min[k])/(train_out_max[k]-train_out_min[k])        
                    else:
                        train_out[i][j][k]  = 1.0
                        
                        
                        
                        

        return train_in, train_out, train_in_max.tolist(), train_in_min.tolist(), train_out_max.tolist(), train_out_min.tolist()
                
    
    
    def add_uncertianty(vector, err):
        vector_out = copy.deepcopy(vector)
        for i in range(len(vector)):
            rde = np.random.normal(0, err, 1 )[0]
            vector_out[i] = vector_out[i]*(1+rde)   # ex     
        return vector_out 
    
    def add_delete(vector, percent):
        vector_out = copy.deepcopy(vector)
        n_del = int(len(vector)*percent)
        import random
        nnx = random.sample(range(len(vector)), n_del)
        for i in nnx:
            vector_out[i] = 0   # ex     
        return vector_out     
    
    def get_avg_err(output, gt):
    
        dim1,dim2,dim3,dim4 = np.shape(gt)
        
        error_v = []
        for i in range(dim1): #//sample
            count = 0
            error = 0
            for j in range(dim3):
                for k in range(dim4):
                    if gt[i,1,j,k] > 0:
                        error += pow( gt[i,1,j,k] -output[i,1,j,k],2)             
                        count += 1
            error_v.append( math.sqrt(error/count)  )
    
        return sum(error_v)/len(error_v)    
    
    
    
    
        
if __name__ == "__main__":


    '''
    train_in = [[-1, 2], [-2, 1],[-1, 2], [-2, 1] ]
    train_out = [ [-3, 2,1], [3,-3,-1], [-3, 2,1], [3,-3,-1] ]
    
    
    a,b,c,d,e,f = DataNormalized.normalized_train_in_out(train_in, train_out)    
    print(a,b)
    print(c,d)
    print(e,f)
    '''