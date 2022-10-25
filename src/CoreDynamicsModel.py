# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 09:30:42 2021

@author: new
"""

import sys
sys.path.append('./../')
import math
import numpy as np
import torch
import torch.autograd
from torch.autograd import Variable
import pickle
import util.FileSaveRead as fsr
import util.FileSaveRead_torch as fsr_t
import util.Network as nw
import src.PointCore_critical as pcc
from numpy.random import multivariate_normal
import random
import streamlit as st
# 

is_cuda = torch.cuda.is_available()
is_cuda = False

def outer_product_sum(A, B=None):
    if B is None:
        B = A

    outer = np.einsum('ij,ik->ijk', A, B)
    return np.sum(outer, axis=0)


def scale_to_one(state_action, inmax, inmin):
    dim = len(inmax)
    for i in range(dim):
        if (inmax[i]-inmin[i])>0:
            state_action[:,i] = (state_action[:,i]-inmin[i])/(inmax[i]-inmin[i])   
        else:
            state_action[:,i] =0  
    return state_action



def rescale_to_normal(state, outmax, outmin):
    for i in range(len(outmax)):
        state[:,i] = state[:,i]*(outmax[i]-outmin[i]) +outmin[i]    
    return state





class MixNNReactorModel(torch.nn.Module): #
    def __init__(self, 
                 models_list, dataset  ):   
        super( MixNNReactorModel, self ).__init__() 
        self.model_vector = models_list
        
        self.n_model = len(self.model_vector)
        self.n_dim = 13
        self.X = [] # 
        self.y = [] # 
        
        
        # 
        self.NNmodel= nw.DNNnet(input_size=self.n_dim*self.n_model, output_size=self.n_dim,num_layers=2, hidden_size= 200)
        self.learner = "weight"
        self.weight = [1.0/len(self.model_vector)]*len(self.model_vector)
        
        if torch.cuda.is_available():
            self.NNmodel = self.NNmodel.cuda()
        
        
        
        data = fsr.FileSaveRead.read_pkl(dataset)
        self.inmin = data["min0"]
        self.inmax = data["max0"]     

        self.P = None
        self._mean = None
        self.Q = None
        

    def update_data(self, Xone, yone):
        if len(self.X) <=30:
            self.X.append( Xone)
            self.y.append(yone)
            
        else:
            self.X.pop(0)
            self.y.pop(0)            
            self.X.append( Xone)
            self.y.append(yone)   
            
        return


    def update_weight(self):
        def evaluation(p0):            
            dev = 0
            for i in range(len(self.X)):
                tmp = 0
                for j in range(self.n_model):
                    tmp += self.X[i][j][0]*p0[j]
                dev += math.pow(self.y[i][0]-tmp,  2)
                
            return dev
        def eqcons(x):
            return sum(x)-1            

        from scipy.optimize import fmin_slsqp
        p0 = [0.5]*self.n_model  
        bouand = [(0,1)]*  self.n_model   
        res = fmin_slsqp(evaluation, p0, eqcons=[eqcons, ], bounds=bouand)
        self.weight = list(res)     
        

        
        return self.weight
    



    

    def update_dnn_model(self):
        train_in_2d = [] 
        for i in range(len(self.X)):
            dd1 = []
            for j in range((self.n_model)):
                for k in range(self.n_dim):
                    value = (self.X[i][j][k]- self.inmin[k])/(self.inmax[k]- self.inmin[k])
                    dd1.append(value)
            train_in_2d.append(dd1)
            
        train_out_2d = []            
        for i in range(len(self.X)):
            dd1 = []
            for k in range(self.n_dim):
                value = (self.y[i][k]- self.inmin[k])/(self.inmax[k]- self.inmin[k])  
                dd1.append(value)
            train_out_2d.append(dd1)            


        if self.n_model <= 1:
            out = np.array(train_in_2d, dtype='float32' )
        else:
            dm = int(len(self.X)*0.9)
            
            train_in_2d_np = np.array(train_in_2d, dtype='float32' )
            train_out_2d_np = np.array(train_out_2d, dtype='float32' ) 
    
            nw.TrainingNetwork.train_net(self.NNmodel, train_in_2d_np[:dm,:], train_out_2d_np[:dm,:],train_in_2d_np[dm:,:], train_out_2d_np[dm:,:] , LR =0.0001)
            
            data_train = nw.TrainingNetwork.data_loader(train_in_2d_np, train_out_2d_np, len(train_in_2d) )
            _, out = nw.TrainingNetwork.get_fit(self.NNmodel, data_train)
        
        


        return
 
    def update_second_learner(self,learner):
        if self.n_model <= 1:
            return
        
        self.learner = learner
        if learner == "weight":
            self.update_weight()

        elif learner == "dnn":
            self.update_dnn_model()
        else:
            raise Exception("error type of seconder learner")



    def predict_step_no_sampling (self, state, action):
        
        if state.dim() == 1 or action.dim() == 1: #必须是两维
            state = state.view(1, -1)
            action = action.view(1, -1)          
        


        if self.learner == "dnn":
            state_new_v = torch.zeros((1,13*self.n_model), dtype=torch.float32)
            
            i=0
            for model in self.model_vector :
                state_new = model.predict_step_no_sampling( state, action) 
                state_new.view((-1,13))
                state_new_v[0,i*13:i*13+13] = state_new[0,:]
                i+=1                 
            
            if self.n_model > 1:
                if torch.cuda.is_available()  :
                    state_new_v = Variable(state_new_v.cuda() )
             
                state_new = self.NNmodel.forward(state_new_v)
                state_new = state_new.cpu() 
            else:
                state_new=state_new_v
                
        elif self.learner == "weight":
            state_new_v = torch.zeros((1,13), dtype=torch.float32)
            
            i=0
            for model in self.model_vector :
                state_new = model.predict_step_no_sampling( state, action) 
                state_new.view((-1,13))
                state_new_v += state_new* self.weight[i]
                i+=1                 
                
                
            if self.n_model > 1:
                state_new = state_new_v / torch.sum(torch.tensor(self.weight, dtype=torch.float32))   
            else:
                state_new = state_new_v            



        return state_new            


    def predict_step_sampling(self ,state, action, P):

        N = 20
        if state.dim() == 1 or action.dim() == 1: #必须是两维
            state = state.view(1, -1)
            action = action.view(1, -1)  
        

        sigmas = multivariate_normal(mean=state.detach().numpy()[0], 
                                     cov=P, size= N)

        for i, s in enumerate(sigmas):
            s = torch.tensor(s, dtype=torch.float32 )
            s_n = self.predict_step_no_sampling(s, action)
            sigmas[i] = s_n.detach().numpy()[0]

        e = multivariate_normal(self._mean, self.Q, N)
        sigmas += e
        
        x = np.mean(sigmas, axis=0)
        P = outer_product_sum(sigmas - x) / (N - 1)

        state_new = torch.tensor([x] , dtype=torch.float32 )

        return state_new
                
    
    def forward(self, state, action ):

        if self.P is not None:
            state_new = self.predict_step_sampling( state, action, self.P)
        else:
            state_new = self.predict_step_no_sampling ( state, action)
            
        return state_new         
        
    
    def normalized_status(self, status):
        for i in range(13):
            if (self.inmax[i]-self.inmin[i])> 0:
                status[i] = (status[i]-self.inmin[i])/(self.inmax[i]-self.inmin[i])
            else:
                status[i]= 0
        return status
        
    def reversed_normalized_status(self, status):
        for i in range(13):
            status[i] = status[i]*(self.inmax[i]-self.inmin[i])+ self.inmin[i]    
        return status    
    
    
    

        
    
    
    
    
    
    
class RFRReactorDynamics(torch.nn.Module):
    def __init__(self, 
                 modelfile = "",  dataset = "" ):
        super( RFRReactorDynamics, self ).__init__() 

        

        with open(modelfile,'rb') as f:
            data_dict = pickle.loads(f.read()) 
            self.rfr_vector =  data_dict['rf_vector']
            self.dataset =  data_dict['dataset']
        data = fsr.FileSaveRead.read_pkl(dataset )            
        self.inmin = data["min0"]
        self.inmax = data["max0"]  
        self.outmax = data["max0"][:13]
        self.outmin = data["min0"][:13]  
        self.dimstate=13


    def predict_step_no_sampling(self, state, action):
        if state.dim() == 1 or action.dim() == 1: 
            state = state.view(1, -1)
            action = action.view(1, -1)     
        
        state_action = torch.cat((state, action), dim=1)
        state_action = scale_to_one(state_action, self.inmax, self.inmin)
        
        new_state = []
        for i in range(self.dimstate):
            rfr = self.rfr_vector[i]
            fit_r = rfr.predict ( state_action.detach().numpy()  )
            new_state.append(fit_r)
        state_new = np.array(new_state,dtype="float32")
        state_new = torch.from_numpy(state_new).view((-1,self.dimstate))

        state_new =  rescale_to_normal(state_new, self.outmax, self.outmin)
        return state_new  
        


class SVRReactorDynamics(torch.nn.Module):
    def __init__(self,  modelfile = "",  dataset = ""):
        super( SVRReactorDynamics, self ).__init__() 
        
        with open(modelfile,'rb') as f:     
            data_dict = pickle.loads(f.read()) 
            
            self.svr_vector =  data_dict['svr_vector']
            self.dataset =  dataset           
            
        data = fsr.FileSaveRead.read_pkl(dataset)
        self.inmin = data["min0"]
        self.inmax = data["max0"]  
        self.outmax = data["max0"][:13]
        self.outmin = data["min0"][:13] 
        self.dimstate=13               
        
        
    def predict_step_no_sampling(self, state, action):
        if state.dim() == 1 or action.dim() == 1:
            state = state.view(1, -1)
            action = action.view(1, -1)     
        
        state_action = torch.cat((state, action), dim=1)
        state_action = scale_to_one(state_action, self.inmax, self.inmin)
        
        new_state = []
        for i in range(self.dimstate):
            svr = self.svr_vector[i]
            fit_r = svr.predict (state_action.detach().numpy())
            new_state.append(fit_r)
        state_new = np.array(new_state,dtype="float32")
        state_new = torch.from_numpy(state_new).view((-1,self.dimstate))

        state_new =  rescale_to_normal(state_new, self.outmax, self.outmin)
            
        return state_new


class DNNReactorDynamics( torch.nn.Module):
    def __init__(self, 
                 modelfile = "", 
                 nx = 13, nu=2, num_layers = 2, 
                 hidden_size = 200,  dataset = ""  ):   
        
        super( DNNReactorDynamics, self ).__init__() 
        
  
        self.model_DNN= nw.DNNnet(input_size=nx+nu, output_size=nx, num_layers=num_layers, hidden_size= hidden_size)
        self.model_DNN,_,_,self.dataset = fsr_t.FileSaveRead.load_torch_model(self.model_DNN, modelfile)
        if is_cuda:
            self.model_DNN = self.model_DNN.cuda()   
            
        data = fsr.FileSaveRead.read_pkl(dataset )
        self.inmin = data["min0"]
        self.inmax = data["max0"]  
        self.outmax = data["max0"][:13]
        self.outmin = data["min0"][:13] 
        self.dimstate=13    
        
    def predict_step_no_sampling(self, state, action):
        if state.dim() == 1 or action.dim() == 1:
            state = state.view(1, -1)
            action = action.view(1, -1)               
        
        state_action = torch.cat((state, action), dim=1)
        state_action = scale_to_one(state_action, self.inmax, self.inmin)
        state_new = self.model_DNN(state_action)
        state_new.view((-1,self.dimstate))
        
        state_new =  rescale_to_normal(state_new, self.outmax, self.outmin)  
        
        return state_new


class LSTMReactorDynamics( torch.nn.Module):
    def __init__(self,                  
                 modelfile = "", 
                 nx = 13, nu=2,num_layers = 2, hidden_size = 200,  dataset = ""  ):    
        super( LSTMReactorDynamics, self ).__init__() 
        
   
        self.model_lstm= nw.LSTMnet(input_size=nx+nu, output_size=nx,num_layers=num_layers, hidden_size= hidden_size)
        self.model_lstm,_,_,self.dataset = fsr_t.FileSaveRead.load_torch_model(self.model_lstm, modelfile)
        if is_cuda:
            self.model_lstm = self.model_lstm.cuda()  
            
        data = fsr.FileSaveRead.read_pkl(dataset)
        self.inmin = data["min0"]
        self.inmax = data["max0"]  
        self.outmax = data["max0"][:13]
        self.outmin = data["min0"][:13]  
        self.dimstate = 13                 
            
    def predict_step_no_sampling(self, state, action):
        if state.dim() == 1 or action.dim() == 1:
            state = state.view(1, -1)
            action = action.view(1, -1)               
        
        state_action = torch.cat((state, action), dim=1)
        state_action = scale_to_one(state_action, self.inmax, self.inmin)
        
        state_action= state_action.view(-1, 1, 15)
        state_new = self.model_lstm(state_action)
        state_new.view((-1,self.dimstate))
        
        state_new =  rescale_to_normal(state_new, self.outmax, self.outmin)  
        
        return state_new  



class BiLSTMReactorDynamics(torch.nn.Module):
    def __init__(self,
                 modelfile = "", 
                 nx = 13, nu=2,num_layers = 2, hidden_size = 200,  dataset = ""  ):    
        super( BiLSTMReactorDynamics, self ).__init__() 

        self.model_BILSTM= nw.BiLSTMnet(input_size=nx+nu, output_size=nx,num_layers=num_layers, hidden_size= hidden_size)
        self.model_BILSTM,_,_,self.dataset = fsr_t.FileSaveRead.load_torch_model(self.model_BILSTM, modelfile)     
        if is_cuda:
            self.model_BILSTM = self.model_BILSTM.cuda()    
        data = fsr.FileSaveRead.read_pkl(dataset)
        self.inmin = data["min0"]
        self.inmax = data["max0"]  
        self.outmax = data["max0"][:13]
        self.outmin = data["min0"][:13]  
        self.dimstate=13        

         
    def predict_step_no_sampling(self, state, action):
        if state.dim() == 1 or action.dim() == 1:
            state = state.view(1, -1)
            action = action.view(1, -1)               
        
        state_action = torch.cat((state, action), dim=1)
        state_action = scale_to_one(state_action, self.inmax, self.inmin)
        
        state_action= state_action.view(-1, 1, 15)
        state_new = self.model_BILSTM(state_action)
        state_new.view((-1,13))
        
        state_new =  rescale_to_normal(state_new, self.outmax, self.outmin)
        
        #print(state_new)
        return state_new    

class GRUReactorDynamics( torch.nn.Module):
    def __init__(self, 
                 modelfile = "", 
                 nx = 13, nu=2,num_layers = 2, hidden_size = 200,  dataset = ""  ):   

        super( GRUReactorDynamics, self ).__init__()     

        self.model_GRU= nw.GRUnet(input_size=nx+nu, output_size=nx,num_layers=num_layers, hidden_size= hidden_size)
        self.model_GRU,_,_,self.dataset = fsr_t.FileSaveRead.load_torch_model(self.model_GRU, modelfile) 
        if is_cuda:
            self.model_GRU = self.model_GRU.cuda()           
        
        data = fsr.FileSaveRead.read_pkl(dataset)
        self.inmin = data["min0"]
        self.inmax = data["max0"]  
        self.outmax = data["max0"][:13]
        self.outmin = data["min0"][:13]  
        self.dimstate=13        
        
        
    def predict_step_no_sampling(self, state, action):
        if state.dim() == 1 or action.dim() == 1:
            state = state.view(1, -1)
            action = action.view(1, -1)               
        
        state_action = torch.cat((state, action), dim=1)
        state_action = scale_to_one(state_action, self.inmax, self.inmin)
        
        state_action= state_action.view(-1, 1, len(self.inmax))
        state_new = self.model_GRU(state_action)
        state_new.view((-1,len(self.outmax)))
        
        state_new =  rescale_to_normal(state_new, self.outmax, self.outmin)
        
        #print(state_new)
        return state_new 

     
class BIGRUReactorDynamics( torch.nn.Module ):
    def __init__(self, 
                 modelfile = "", 
                 nx = 13, nu=2,num_layers = 2, hidden_size = 200 ,  dataset = "" ):   

        super( BIGRUReactorDynamics, self ).__init__()     

        self.model_BIGRU= nw.BiGRUnet(input_size=nx+nu, output_size=nx,num_layers=num_layers, hidden_size= hidden_size)
        
        if is_cuda:
            self.model_BIGRU = self.model_BIGRU.cuda()      
        self.model_BIGRU,_,_,self.dataset = fsr_t.FileSaveRead.load_torch_model(self.model_BIGRU, modelfile)             
        data = fsr.FileSaveRead.read_pkl(dataset)
        self.inmin = data["min0"]
        self.inmax = data["max0"]  
        self.outmax = data["max0"][:13]
        self.outmin = data["min0"][:13] 
        self.dimstate=13
        
    def predict_step_no_sampling(self, state, action):
        if state.dim() == 1 or action.dim() == 1:
            state = state.view(1, -1)
            action = action.view(1, -1)               
        
        state_action = torch.cat((state, action), dim=1)
        state_action = scale_to_one(state_action, self.inmax, self.inmin)
        
        state_action= state_action.view(-1, 1, len(self.inmax))
        state_new = self.model_BIGRU(state_action)
        state_new.view((-1,len(self.outmax)))
        
        state_new =  rescale_to_normal(state_new, self.outmax, self.outmin)

        return state_new         
    

    

class ReactorRealDynamics( torch.nn.Module ): #
    def __init__(self,
                 sigma = 0, 
                 param = None,
                 isRandom= False,
                 modelfile = "", dataset = ""  ):    
        super( ReactorRealDynamics, self ).__init__() 

        if modelfile != "":
            with open(modelfile,'rb') as f:
                data_dict = pickle.loads(f.read()) 
                self.dataset =  dataset
                data = fsr.FileSaveRead.read_pkl(self.dataset)
                param = data["param"]
                
        
            self.coreReal= pcc.PointCore_critical(param = param,sigma=sigma)
            
            self.isRandom = isRandom
            self.balance = self.coreReal.define_balance_status(1.0)
        
   
        
        pass
    
    
    def predict_step_no_sampling(self, state, action):
        dt= 0.1
        if state.dim() == 1 or action.dim() == 1:
            state = state.view(1, -1)
            action = action.view(1, -1)            
        
        dim1 = state.shape[0]
        state_pert = []
        for k in range(dim1):
            state_k = state[k]
            action_k = action[k]
            new_state = self.coreReal.predict_state_action(dt,state_k.tolist(), action_k.tolist() )

            if self.isRandom:
                avg = self.balance*0.01
                std = self.balance*0.01
                for i in range(13):
                    new_state[i] = new_state[i]+random.gauss(avg[i],std[i] )
                
                
            state_pert.append(new_state )
        np_state = np.array(state_pert,dtype="float32")
        kk = torch.from_numpy(np_state)

        return kk   
    
    


    
    
def get_method_list( model_dick, param=None,  dataset = "" ):
    models ={}
    

    for key in model_dick:
        if key =="BiGRU":
            reactor_Bigru = BIGRUReactorDynamics(
                 modelfile = model_dick[key] , dataset =dataset       )      
            models[key]=reactor_Bigru
            
            
        elif key =="BiLSTM":
            reactor_bilstm = BiLSTMReactorDynamics(
                 modelfile = model_dick[key] , dataset =dataset         )  
            
            models[key]=reactor_bilstm           
    
    
        elif key =="MLP":
            reactor_dnn = DNNReactorDynamics(
                 modelfile = model_dick[key], dataset =dataset            ) 
            models[key]=reactor_dnn      
    
        elif key =="GRU":
            reactor_gru = GRUReactorDynamics(
                 modelfile = model_dick[key]  , dataset =dataset         ) 
            models[key]=reactor_gru     
            
            
        elif key =="LSTM":
            reactor_lstm = LSTMReactorDynamics(
                 modelfile = model_dick[key]  , dataset =dataset       ) 
            models[key]=reactor_lstm                
            
        elif key =="SVR":
            #import streamlit as st
            #st.write(model_dick[key])
            reactor_svr = SVRReactorDynamics(
                 modelfile = model_dick[key] , dataset =dataset     ) 
            
            models[key]=reactor_svr        
    
        elif key =="RFR":
            
            reactor_rf = RFRReactorDynamics(
                 modelfile = model_dick[key]  , dataset =dataset     ) 
            
            models[key]=reactor_rf   
    
        elif key =="PERT":
            reactor_pert = ReactorRealDynamics( param = param,
                                               sigma = 0.1, 
                                               modelfile = model_dick[key], dataset =dataset 
                                               )  
            models[key]=reactor_pert       
            
        elif key =="REAL":
            reactor_real = ReactorRealDynamics( param = param,
                                               sigma = 0,
                                               modelfile = model_dick[key], dataset =dataset  )  
            models[key]=reactor_real      
            
        elif key =="NOISE":
            reactor_rand = ReactorRealDynamics( param = param, 
                                               sigma = 0,
                                               isRandom = True,
                                               modelfile = model_dick[key], dataset =dataset )  
            models[key]=reactor_rand      
    
        else:
            raise Exception("error type of model name:", key)
    return models
    
    

        
        
def comparison_models(my_bar, models):     #累计误差的应用    
 
    T = 100
    rest = {}
    import random
    import streamlit as st

    n_model = len(list(models.keys()))
    
    coreReal= models["REAL"]
    state_start = coreReal.coreReal.define_balance_status(1.0)
    
    state_start = torch.tensor( state_start, dtype=torch.float32 )    
    u = torch.tensor((0.0,290), dtype=torch.float32)
    if state_start.dim() == 1 or u.dim() == 1:
        state_start = state_start.view(1, -1)
        u = u.view(1, -1)    
            
    u_list = []
    for i in range(T):
        at = (0.5-random.random()) 
        uadd = torch.tensor((-0.005, 0), dtype=torch.float32)
        u = uadd
        u_list.append(u)
            
        
    state_begin = state_start    
    v_REAL_c = [float(state_begin[0,0] )]
    for i in range(T):
        u= u_list[i]
        state_next = models["REAL"].predict_step_no_sampling( state_begin,u )    
        v_REAL_c.append(float(state_next[0,0] )) #############
        state_begin = state_next
        
    
        
    tt = 0
    for key,value in models.items():
        state_begin = state_start
        v_REAL = [float(state_begin[0,0] )]
        for i in range(T):
            u= u_list[i]
            state_next = models[key].predict_step_no_sampling( state_begin,u )    
            v_REAL.append(float(state_next[0,0] )) #############
            state_begin = state_next
            tt += 1
            my_bar.progress(int((tt+1)/n_model/T) *100)
        err = np.mean(pow( np.array(v_REAL) - v_REAL_c, 2.))  
        label = key + " {:7.2e}".format(err)
        rest[label] = v_REAL
        
    return rest
    

def comparison_models_valid(valid_in,valid_out, models):     #累计误差的应用    
 
    import random
    import streamlit as st

    n_sample = min(np.shape(valid_in)[0],200)

    
    valid_out_f = valid_out[:n_sample, :]
   
    rest1 = {}
    rest2 = {}        
    for key,value in models.items():
        v_REAL = []
        v_Real_all = []
        for i in range(n_sample):
            state_start = list(valid_in[ i, 0, :13])

            state_start = torch.tensor( state_start, dtype=torch.float32 )
            u = torch.tensor((valid_in[i, 0,-2],valid_in[i, 0,-1]), dtype=torch.float32)   
            
            if state_start.dim() == 1 or u.dim() == 1:
                state_start = state_start.view(1, -1)
                u = u.view(1, -1)            
                
            state_next = models[key].predict_step_no_sampling( state_start,u )  
            #st.write(list(state_next[0,:]), valid_out[i,:])
            state_next_np =  state_next.detach().numpy()
            v_REAL.append(float(state_next[0,0]) )
            v_Real_all.append(state_next_np[0,:])
        
        err_test1 = np.mean(pow( np.array(v_REAL) - valid_out_f[:,0], 2.))  
        err_test2 = np.mean(pow( np.array(v_Real_all) / valid_out_f-1., 2.)) 
        
        rest1[key] = err_test1
        rest2[key] = err_test2
        
    return rest1,rest2


    
    
    
    
    
    
    


