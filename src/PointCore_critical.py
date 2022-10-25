# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 11:16:49 2021

@author: new
"""
import sys
import os
sys.path.append('./../')    
sys.path.append('./../../')   
sys.path.append('./../../../')   
sys.path.append('./../../../../')  

import numpy as np
import math

class PointCore_critical(object):  
    def __init__(self, 
                 pho_for_cv = 0., 
                 tin_for_cv= 290.,
                 t_inlet = 290,     
                 nr_init = 1.0,
                 sigma= 0,
                 param= None  ):
        
        self.param = param
        self.beta_list = param[0:6]
        self.lambda_list= param[6:12 ]         
        
        self.Lambda_time = param[12] #

        self.Gr = param[13] # dk/k
        self.p0 = param[14] # MW
        
        self.ff = param[15]
        self.v_neutron = param[16]
        self.lamda_I = param[17]
        self.lamda_Xe = param[18]
        self.lamda_Pm = param[19]
        
        self.sigma_Xe = param[20]
        self.sigma_Sm = param[21]
        
        self.gamma_I = param[22]
        self.gamma_Xe = param[23]
        self.gamma_Pm = param[24]        
        
        self.n0 = param[25]
              
        self.sigma_fission = param[26]

        self.mu_c = param[27]
        self.mu_f = param[28]
        self.omega = param[29] #MWs/c
        self.M = param[30] # MW/C	
        self.alpha_f =param[31]  # dk/k/c
        self.alpha_c=param[32]   # dk/k/c  


        self.betasum = sum(self.beta_list)
        
        self.pho_for_cv  = pho_for_cv
        self.tin_for_cv  = tin_for_cv
        
        self.nr_init = nr_init
        self.t_inlet =t_inlet
        self.sigma = sigma
        self.add_noisy(sigma= sigma)
        
        self.blance_state_FP = self.define_balance_status(1.0)
        self.blance_state = self.define_balance_status(self.nr_init)
    
  

     
                  
        
    def add_noisy(self, sigma= 0):
        import numpy as np
        np.random.seed(1000)
        for i in range(len(self.beta_list)):
            a = np.random.standard_normal()*sigma
            self.beta_list[i] = (1+a)*self.beta_list[i]
            b = np.random.standard_normal()*sigma
            self.lambda_list[i] = (1+b)*self.lambda_list[i]

        self.mu_c = self.mu_c* (1+np.random.standard_normal()*sigma)
        self.mu_f = self.mu_f* (1+np.random.standard_normal()*sigma)
        self.omega = self.omega* (1+np.random.standard_normal()*sigma)
        self.M = self.M* (1+np.random.standard_normal()*sigma)
        self.alpha_f =self.alpha_f * (1+np.random.standard_normal()*sigma)
        self.alpha_c=self.alpha_c* (1+np.random.standard_normal()*sigma)        
        
        self.betasum = sum(self.beta_list)  
        self.blance_state_FP = self.define_balance_status(1.0) 
        self.blance_state = self.define_balance_status(self.nr_init)
        
        
    def define_balance_status(self, nr_init):
        # when have pingheng
        nr = nr_init
        cr = [0.]*6
        for i in range(6):
            cr[i] = self.beta_list[i]/self.lambda_list[i]/self.Lambda_time*nr

        phi = self.n0* nr * self.v_neutron
        
        density_I = self.gamma_I*self.sigma_fission*phi / self.lamda_I
        density_Xe = ( self.gamma_Xe*self.sigma_fission*phi + self.lamda_I * density_I ) / (self.sigma_Xe*phi + self.lamda_Xe )
        density_Pm = self.gamma_Pm*self.sigma_fission*phi / self.lamda_Pm
        density_Sm = self.gamma_Pm*self.sigma_fission / self.sigma_Sm  

        omega = self.omega
        M = self.M

        t_coolant = (self.p0*nr+2*M*self.t_inlet)/(2*M )
        t_fuel =  t_coolant + (self.ff * self.p0 * nr ) /omega
        
        
        y = []
        y.append(nr)
        y.extend(cr)        
           
        y.extend( [t_fuel, t_coolant] )                     
        y.extend( [ density_I, density_Xe, density_Pm, density_Sm] )         
          
        return np.array(y) 
            



    def update_pho_feedback(self, current_status,  action):    
        pho_ex, t_inlet = action[0], action[1]
          
        t_fuel = current_status[7]
        t_cool = current_status[8]
    
        t_fuel_ref = self.blance_state_FP[7]
        t_cool_ref = self.blance_state_FP[8] 
    

        pho_t = self.alpha_f  * (t_fuel-t_fuel_ref) + self.alpha_c  * (t_cool-t_cool_ref )
        xe = current_status[10]
        sm = current_status[12]
    
        xe_r = self.blance_state_FP[10]
        sm_r = self.blance_state_FP[12]   
        
        pho_xe = -self.sigma_Xe*(xe-xe_r ) / self.sigma_fission
        pho_sm = -self.sigma_Sm*(sm -sm_r )/self.sigma_fission         
        
        #pho_tot =  pho_ex+ pho_t pho_xe + pho_sm
        #pho_tot =  pho_ex+ pho_t+ pho_xe + pho_sm
        pho_tot =  pho_ex

        return pho_tot
        
    def predict_state_action(self, dt, state, action):
        pho_tot = self.update_pho_feedback(  (state), (action) )
    
        t_inlet = float(action[1])
        y0 = state

        y1 = PointCore_critical.f_cv(y0, dt, pho_tot, t_inlet, self.sigma, self.param)
        return np.array(y1)
    
    @staticmethod
    def f_13(t, y, data):
        import numpy as np
        n_n = 13
        f = np.zeros(shape=(n_n,), dtype=float)
        
        nr = y[0]
        cr = y[1:7]
        
        t_fuel = y[7]
        t_coolant = y[8] 
    
        density_I = y[9] 
        density_Xe = y[10]  
        density_Pm = y[11]  
        density_Sm = y[12] 
    
        pho_1 = data.pho_for_cv
      
        crtot = 0
        for i in range(6):
            crtot += data.lambda_list[i]* cr[i] 
        
        dnr = (pho_1-data.betasum)/ data.Lambda_time * nr+ crtot
    
        dcr = []
        for i in range(6):
            dcr.append( data.beta_list[i]/ data.Lambda_time*nr - data.lambda_list[i]*cr[i]   )
       
        omega = data.omega
        
        dTf = (data.ff* data.p0 * nr -omega*t_fuel +  t_coolant*omega) / data.mu_f 
       
        M = data.M
        dtc =( (1.-data.ff)*data.p0*nr + omega*t_fuel -(2*M + omega)*t_coolant  + 2*M*data.tin_for_cv )/ data.mu_c 
        
        
        phi = data.n0* nr * data.v_neutron
        dIo = -data.lamda_I * density_I + data.gamma_I*data.sigma_fission* phi
        dxe = (data.gamma_Xe*data.sigma_fission*phi + data.lamda_I * density_I) - (data.sigma_Xe*phi + data.lamda_Xe )*density_Xe
    
    
        dpm = data.gamma_Pm* data.sigma_fission* phi - data.lamda_Pm*density_Pm
        dsm = data.lamda_Pm* density_Pm - data.sigma_Sm* density_Sm * phi
    
        f[0] = dnr
        f[1:7] = np.array(dcr)
        
        f[7] =  dTf
        f[8] =  dtc 
        
        f[9] = dIo
        f[10] = dxe
        f[11] = dpm
        f[12] =  dsm
            
        
        return f

    @staticmethod
    def f_cv(y0, dt, pho, t_inlet, sigma, param):
        
        from scipy.integrate import ode
    
        n = ode(PointCore_critical.f_13).set_integrator('dopri5', nsteps=1000)         
        
        data = PointCore_critical(pho_for_cv = pho, tin_for_cv=t_inlet, sigma=sigma,param=param )
        #print(data.pho_for_cv, data.tin_for_cv)
        n.set_initial_value(y0, 0)
        n.set_f_params(data)
    
        n.integrate(dt)
    
        return n.y   


    
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    