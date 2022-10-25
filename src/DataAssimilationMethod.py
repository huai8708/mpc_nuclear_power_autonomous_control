# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 15:00:42 2021

@author: new
"""
import torch
from numpy.random import multivariate_normal
import numpy as np
import copy
import streamlit as st


def outer_product_sum(A, B=None):
    if B is None:
        B = A

    outer = np.einsum('ij,ik->ijk', A, B)
    return np.sum(outer, axis=0)

class EnsembleKalmanFilter(torch.nn.Module):
    def __init__(self, 
                 model_prediction = None, 
                 model_observation = None, 
                 
                 dim_x = 13,
                 dim_z = 2,
                 N = 1000,
                 Q = None,
                 R = None,
                 
                 ):
        super( EnsembleKalmanFilter, self ).__init__() 
        
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.N =N
        
        self.K = np.zeros((self.dim_x, self.dim_z)) #卡尔曼增益
        self.z = np.array([[None] * self.dim_z]).T  # 观测值
        self.S = np.zeros((self.dim_z, self.dim_z))   # 系统不确定性
        self.SI = np.zeros((self.dim_z, self.dim_z))  # inverse system uncertainty

        self.Q = Q
        self.R = np.eye(self.dim_z)       # state uncertainty
        self.inv = np.linalg.inv

        # used to create error terms centered at 0 mean for
        # state and measurement
        self._mean = np.zeros(self.dim_x) #状态的均值

        self.P = np.eye(self.dim_x)*1.0e-10  #卡尔马更新后的状态方差
        self.x = None #状态值
        self.sigmas = None
        
        self.x_reference= None

        
        self.model_prediction = model_prediction; #
        self.model_observation = model_observation;



    def initialize_x_P(self, x, P):
        self.x = x
        self.P = P
        
        self.sigmas = multivariate_normal(mean=x, cov=P, size=self.N)

    def set_x_reference(self, x_reference):
        self.x_reference = x_reference


    def initialize_mean_Q(self, mean, Q):

        if self.x_reference is None:
            self._mean = mean
            self.Q = Q
        else:
            self._mean = list(np.array(mean)/np.array(self.x_reference))
            self.Q = np.zeros((13,13))
            for i in range(13):
                self.Q[i,i] = Q[i,i] / self.x_reference[i]/  self.x_reference[i]
        
        
    def update(self, z, _mean_z, R=None):  #一个新的测量来之后 更新
        if z is None: #
            self.z = np.array([[None]*self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            return

        if R is None: #测量的观测误差
            R = self.R
            
        if np.isscalar(R): #是个标量
            R = np.eye(self.dim_z) * R

        N = self.N 
        dim_z = len(z)
        sigmas_h = np.zeros((N, dim_z))
        
        # transform sigma points into measurement space
        for i in range(N):
            sigmas_h[i] = self.model_observation(self.sigmas[i])

        z_mean = np.mean(sigmas_h, axis=0) 
        

        P_zz =( outer_product_sum(sigmas_h - z_mean) / (N-1)) + R
        P_xz = outer_product_sum( self.sigmas - self.x, sigmas_h - z_mean) / (N - 1)

        self.S  = P_zz
        self.SI = self.inv(self.S)
        self.K = np.dot(P_xz, self.SI)

        e_r = multivariate_normal(_mean_z, R, N)
        
        for i in range(N):
            self.sigmas[i] += np.dot(self.K, z + e_r[i] - sigmas_h[i])

        self.x = np.mean(self.sigmas, axis=0)
        P1 = self.P - np.dot(np.dot(self.K, self.S), self.K.T)

        self.P = P1
        # save measurement and posterior state
        self.z = copy.deepcopy(z)  
        return self.x, self.P
        
        
    def predict(self,u):
        """ Predict next position. """

        self.sigmas = multivariate_normal(self.x, self.P, size=self.N)
        N = self.N
        for i, s in enumerate(self.sigmas):
            if self.x_reference is None:
                pass
            else:
                for ix in range(13):
                    s[ix] = s[ix]* self.x_reference[ix]
                
                
            s = torch.tensor(s, dtype=torch.float32 )
            s_n = self.model_prediction(s, u)    
          
            
            self.sigmas[i] =  s_n.detach().numpy()[0]
            if self.x_reference is None:
                pass
            else:
                for ix in range(13):
                    self.sigmas[i][ix] = self.sigmas[i][ix]/ self.x_reference[ix]  
                    

        e = multivariate_normal(self._mean, self.Q, N)
        self.sigmas += e

        self.x = np.mean(self.sigmas, axis=0)
        self.P = outer_product_sum(self.sigmas - self.x) / (N - 1)

        return self.x, self.P


    def generate_model_prediction_err(self,core_real, system_dev,actions=None):
        import random
        random.seed(100)
        x_rep = []
        y_rep = []
        
        if actions is None:
        
            x_models = [[] for i in range(len(self.model_prediction.model_vector))]
            for irep in range(5):
                
                #初始功率台阶
                nr_init=(0.9*random.random()+0.1  )
                state_begin = core_real.define_balance_status(nr_init)
                for ic in range(20):    #在每个功率台阶上做扰动
        
                    rd1 = random.random()-0.5
                    rd2 = random.random()-0.5
                    action = [0.1*rd1, 290] #
                        
                    new_state = core_real.predict_state_action(0.1,state_begin, action) 
                        
                    while new_state[0] > 3. or new_state[0] < 0.01:
                        rd1 = random.random()-0.5
                        rd2 = random.random()-0.5
                        action = [0.1*rd1, 290] #
                        new_state = core_real.predict_state_action(0.1,state_begin, action)                           
                    
                    
    
                    act_tmp = torch.from_numpy(np.array([action], dtype='float32' ) ) #初始动作
                    state_begin_tmp = torch.from_numpy(np.array([state_begin], dtype='float32' ) ) #初始动作                
    
                    
                    new_state_model = self.model_prediction.predict_step_no_sampling(state_begin_tmp, act_tmp)
                    new_state_model_np = new_state_model.detach().numpy()
                    new_state_model_np=np.squeeze(new_state_model_np)
                    
                    for im, model in enumerate(self.model_prediction.model_vector) :
                        state_models = model.predict_step_no_sampling( state_begin_tmp, act_tmp) 
                        state_models_np = state_models.detach().numpy()
                        state_models_np= np.squeeze(state_models_np)
                        x_models[im].append(list(state_models_np))
    
    
                    y_rep.append(list(new_state) )
                    x_rep.append(list(new_state_model_np) )
                    
                    state_begin = new_state  
        else:
            x_models = [[] for i in range(len(self.model_prediction.model_vector))]
            state_begin = core_real.define_balance_status(1)
            
            for ic in range(len(actions)):    #在每个功率台阶上做扰动
                action = actions[ic] #
                    
                new_state = core_real.predict_state_action(0.1,state_begin, action) 

                act_tmp = torch.from_numpy(np.array([action], dtype='float32' ) ) #初始动作
                state_begin_tmp = torch.from_numpy(np.array([state_begin], dtype='float32' ) ) #初始动作                

                
                new_state_model = self.model_prediction.predict_step_no_sampling(state_begin_tmp, act_tmp)
                new_state_model_np = new_state_model.detach().numpy()
                new_state_model_np=np.squeeze(new_state_model_np)
                
                for im, model in enumerate(self.model_prediction.model_vector) :
                    state_models = model.predict_step_no_sampling( state_begin_tmp, act_tmp) 
                    state_models_np = state_models.detach().numpy()
                    state_models_np= np.squeeze(state_models_np)
                    x_models[im].append(list(state_models_np))


                y_rep.append(list(new_state) )
                x_rep.append(list(new_state_model_np) )
                
                state_begin = new_state  

        n_sample = len(x_rep)
        if self.x_reference is None:
            
            Xdev = np.array(y_rep) - np.array(x_rep)
        else:
            Xdev = np.array(y_rep)/np.array(self.x_reference) - np.array(x_rep)/np.array(self.x_reference)
        
        self._mean = np.mean(Xdev, axis=0)
        
        self.Q = np.zeros((13,13))
        for i in range(13):
            self.Q[i,i] = np.std(Xdev[:,i])*np.std(Xdev[:,i])
            if self.Q[i,i] <= 0:
                self.Q[i,i] = 1.0e-30
        
        
        st.write("整体的误差为" )
        r1,r2 = st.columns((1.5,10))
        r1.write(self._mean)
        r2.write(self.Q)
        

        for i in range(len(self.model_prediction.model_vector)):
            Xdev = np.array(y_rep)/np.array(self.x_reference) - np.array(x_models[i])/np.array(self.x_reference)
            
            _mean = np.mean(Xdev, axis=0)
            
            Q = np.zeros((13,13))
            for j in range(13):
                Q[j,j] = np.std(Xdev[:,j])*np.std(Xdev[:,j])            
                if Q[j,j] <= 0:
                    Q[j,j] = 1.0e-30            
            st.write("第",i,"个模型的误差为：")
            r1,r2 = st.columns((1.5,10))
            r1.write(_mean)
            r2.write(Q)


        if system_dev=="yes":
            self._mean = np.zeros((13))










