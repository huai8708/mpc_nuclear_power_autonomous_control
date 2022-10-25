import sys
sys.path.append('./../')    
sys.path.append('./../../')   
sys.path.append('./../../../')   
sys.path.append('./../../../../')  


import src.PointCore_critical as pcc
import util.FileSaveRead as fsr
import util.DataNormalized as dn
import random
import numpy as np
import torch
import streamlit as st

####################################################################################################
class ReactorDynamics(): #
    def __init__(self, 
                 nr_init = 0.8,
                 sigma = 0,
                 t_inlet = 290,
                 param = None):      
        
        self.nr_init = nr_init
        self.sigma = sigma
        self.t_inlet = t_inlet
        self.param = param
        
        
        self.core = pcc.PointCore_critical( 
                    t_inlet = t_inlet,
                         nr_init=nr_init,
                         sigma = self.sigma,
                         param = param) 
        
        

        
        pass
    

    
    def forward(self, state, action):
        dt = 0.1
        if state.dim() == 1 or action.dim() == 1:
            state = state.view(1, -1)
            action = action.view(1, -1)            
        
        
        dim1 = state.shape[0]
        state_pert = []
        for k in range(dim1):
            state_k = state[k]
            action_k = action[k]
            
            new_state = self.core.predict_state_action(dt, state_k.tolist(), action_k.tolist())
            state_pert.append(new_state )
            
        np_state = np.array(state_pert,dtype="float32")
        kk = torch.from_numpy(np_state)
        
        return kk



    def generate_different(self):
        #random.seed(101)
        dt = 0.1
        
        at = pcc.PointCore_critical( 
                    t_inlet = self.t_inlet,
                         nr_init=self.nr_init,
                         param = self.param) 
        

        
        pert1 = pcc.PointCore_critical( 
                    t_inlet = self.t_inlet,
                         nr_init=self.nr_init ,
                         param = self.param) 
        
        
        pert2 = pcc.PointCore_critical( 
                    t_inlet = self.t_inlet,
                         nr_init=self.nr_init ,
                         param = self.param) 
        
        
        pert3 = pcc.PointCore_critical( 
                    t_inlet = self.t_inlet,
                         nr_init=self.nr_init, 
                         param = self.param) 
        

        
        at.add_noisy(sigma=0)
        pert1.add_noisy(sigma=0.005)
        pert2.add_noisy(sigma=0.01)        
        pert3.add_noisy(sigma=0.02)
        
     
        power_real = []
        power_pert1 = []
        power_pert2 = []
        power_pert3 = []
        
        
        state =at.blance_state.copy()
        state_pert1 = pert1.blance_state.copy()
        state_pert2 = pert2.blance_state.copy()
        state_pert3 = pert3.blance_state.copy()        

        action = [0, self.t_inlet]
        my_bar = st.progress(0)
        
        T = 100
        for j in range(T):
            
            rd = random.random()-0.5
            action_addition = [0.0005*rd, 0] #变化量                 

            action[-2] += action_addition[0]
            action[-1] += action_addition[1]    
                       

            state_c = at.predict_state_action(dt, state, action )                
            state_pert1_c = pert1.predict_state_action(dt, state_pert1, action )
            state_pert2_c = pert2.predict_state_action(dt, state_pert2, action )
            state_pert3_c = pert3.predict_state_action(dt, state_pert3, action )
    
                    
            if state_c[0] > 2. or state_c[0] < 0.1: #后撤
                action[-2] -= action_addition[0]
                action[-1] -= action_addition[1]   
            else:
                power_real.append(state_c[0] )
                power_pert1.append(state_pert1_c[0] )
                power_pert2.append(state_pert2_c[0] )
                power_pert3.append(state_pert3_c[0] )   
                
                print( state_pert1_c[0]- state_c[0])
                
                state = state_c.copy()
                state_pert1 = state_pert1_c.copy()
                state_pert2 = state_pert2_c.copy()
                state_pert3 = state_pert3_c.copy()
                
                
            
            my_bar.progress(int((j+1)/T*100))
                

    
        return power_real,power_pert1,power_pert2,power_pert3


    ###  根据一个点对模型，进行尝试性的控制，并搜索处一堆初始的训练数据    
    @staticmethod
    def data_normalized(npower, nstep, dataTrainIn, dataTrainOut):
        alldata = []
        for i in range((npower)):
            for j in range(nstep ):
                kk1 = dataTrainIn[i][j].copy()
                kk2 = dataTrainOut[i][j].copy()
                kk2.extend([0,290])
                alldata.append(kk1)
                alldata.append(kk2)
                
        max0 = []
        min0 = []
        for i in range(len(alldata[0])):
            max0.append( np.max(np.array(alldata)[:, i]) )
            min0.append( np.min(np.array(alldata)[:, i]) )   

        for i in range(len(dataTrainIn)):
            for j in range(len(dataTrainIn[i])):
                for k in range(len(dataTrainIn[i][j])):
                    if (max0[k]-min0[k]) > 0:
                        dataTrainIn[i][j][k] = (dataTrainIn[i][j][k] -min0[k])/(max0[k]-min0[k])    
                    else:
                        dataTrainIn[i][j][k] = 0
                
        for i in range(len(dataTrainOut)):
            for j in range(len(dataTrainOut[i])):
                for k in range(len(dataTrainOut[i][j])):
                    if (max0[k]-min0[k]) > 0:                
                        dataTrainOut[i][j][k] = (dataTrainOut[i][j][k] -min0[k])/(max0[k]-min0[k])  
                    else:
                        dataTrainOut[i][j][k] = 0
        return max0, min0, dataTrainIn, dataTrainOut



    def generate_data_pkl_critical(self,my_bar, save_file_dir,npower=500, nstep=1000):
        random.seed(101)
        
        dt = 0.1
    
        dataTrainIn = []
        dataTrainOut = []

        tttt = 0
        for i in range(npower):
            nr_init=(0.9*random.random()+0.1  )
            state_begin = self.core.define_balance_status(nr_init)

            tmp_in = []
            tmp_out = []
            done =True
            while done:
                rd = random.random()-0.5
                action = [0.1*rd, 290]
                
                new_state = self.core.predict_state_action(dt,state_begin, action) 
                
                if new_state[0] < state_begin[0]*0.95 or new_state[0] > state_begin[0]*1.05: #搜索临界棒位
                    print(i, "trying...", "{:12.5e} {:12.5e} {:12.5e}".format(state_begin[0],action[0],new_state[0]) )
                    done = True
                else:
                    cout = 0
                    while cout < nstep:  #再临界棒位的基础熵，进行扰动
                    
                        
                        aa = 0.001*(random.random()-0.5)
                        action[0] += aa
                        new_state = self.core.predict_state_action(dt,state_begin, action) 
                        
                        if new_state[0] > 2. or new_state[0] < 0.1:
                            print(i, "##########step",cout+1)
                            action[0] -= aa
                        else:
                            tmp = state_begin.tolist().copy()
                            tmp.extend(action)
                            tmp_in.append(tmp)
                            tmp_out.append(new_state.tolist())
                            state_begin = np.copy(new_state)
                            cout += 1
                        tttt += 1
                            
                    done= False
                    
            if i < 10:                        
                import matplotlib.pyplot as plt        
                fig, ax = plt.subplots()
                ax.plot(range(len(tmp_in)), np.array(tmp_in)[:,0])
                st.write(fig)
                
            dataTrainIn.append(tmp_in)    
            dataTrainOut.append(tmp_out)   
        
            if i%100 == 0:
                print("** powerlevel: ", i)    
        
    
        max0, min0, dataTrainIn, dataTrainOut = ReactorDynamics.data_normalized(npower, nstep, dataTrainIn, dataTrainOut)     
        

    
        return max0, min0, dataTrainIn, dataTrainOut










    def generate_data_pkl_random(self,my_bar, save_file_dir,npower=1000, nstep=20):
        random.seed(101)
        
        dt = 0.1
        import streamlit as st
        
        dataTrainIn = []
        dataTrainOut = []
        tttt = 0
        for j in range(npower):
            nr_init=(0.9*random.random()+0.1  )
                 
            state_begin = self.core.define_balance_status(nr_init)
            
            tmp_in = []
            tmp_out = []
            for i in range(nstep):
                rd1 = random.random()-0.5
                rd2 = random.random()-0.5
                action = [0.1*rd1, 290] #
                
                new_state = self.core.predict_state_action(dt,state_begin, action) 
                
                while new_state[0] > 3. or new_state[0] < 0.01:
                    rd1 = random.random()-0.5
                    rd2 = random.random()-0.5
                    action = [0.1*rd1, 290] #
                    print("trying ...")
                    new_state = self.core.predict_state_action(dt,state_begin, action)                           
                             
                            
                tmp = state_begin.tolist().copy()
                tmp.extend(action)
                tmp_in.append(tmp)
                tmp_out.append(new_state.tolist())
                state_begin = np.copy(new_state)
                
                tttt += 1
                my_bar.progress(int(tttt/npower/nstep*100))
            
            if j < -1:
                import matplotlib.pyplot as plt        
                fig, ax = plt.subplots()
                ax.plot(range(len(tmp_in)), np.array(tmp_in)[:,0])
                st.write(fig)
                

            dataTrainIn.append(tmp_in)    
            dataTrainOut.append(tmp_out)   
        

        
    
        max0, min0, dataTrainIn, dataTrainOut = ReactorDynamics.data_normalized(npower, nstep, dataTrainIn, dataTrainOut)     
        

        
    
        return max0, min0, dataTrainIn, dataTrainOut



    @staticmethod
    def getTraindata(save_file_dir, timeStep =3):
                    
        data = fsr.FileSaveRead.read_pkl(save_file_dir)
        
        d_in = data["input"] 
        d_out = data["output"]  

        data_in_3d = []
        data_out_2d = []
        tot_sample = len(d_in)
        #tot_sample =200
        if len(d_in[0]) <= 10 or timeStep >= 10:
            raise Exception("number of training time is less than 10 step or timestep is more than 10")
                
        for i in range(tot_sample ):
            for j in range(10, len(d_in[i])):
                if j > timeStep-1:
                    data_in_3d.append( d_in[i][j-timeStep:j])
                    data_out_2d.append( d_out[i][j-1])
        
        randomset = False
        if randomset:
            random.seed(10)
            stat = len(data_in_3d)
            listindex = list(range(stat))
            random.shuffle (listindex )
            
            train_in_3d = []
            train_out_2d = []
            
            test_in_3d = []
            test_out_2d = []        
            for i in range( len(listindex)):
                ix = listindex[i]
                if i < 0.7*len(listindex):
                
                    train_in_3d.append( data_in_3d[ix])
                    train_out_2d.append( data_out_2d[ix])
                else:
                    test_in_3d.append( data_in_3d[ix])
                    test_out_2d.append( data_out_2d[ix])                
        else:
            trainsample = int(len(data_in_3d)*0.7)
            
            train_in_3d = data_in_3d[:trainsample]
            train_out_2d = data_out_2d[:trainsample]
            
            test_in_3d = data_in_3d[trainsample:]
            test_out_2d = data_out_2d[trainsample:]    

        
        train_in_3d  =  np.array(train_in_3d, dtype='float32')
        train_out_2d  =  np.array(train_out_2d, dtype='float32')    
        test_in_3d  =  np.array(test_in_3d, dtype='float32')    
        test_out_2d  =  np.array(test_out_2d, dtype='float32')    
    
    
        return train_in_3d,  train_out_2d, test_in_3d, test_out_2d



if __name__ == "__main__":
    #ReactorPertDynamics.generate_different()
    filesave= r"E:\DigitalTwinsReactorCore\db\LearningDataPointCoreCritical\point_core_data"
    ReactorDynamics.generate_data_pkl_random(filesave,npower=500, nstep=300)
    #train_in_3d,  train_out_2d, test_in_3d, test_out_2d=ReactorPertDynamics.getTraindata(filesave, timeStep =3)
    
    #print(np.shape(np.array(test_in_3d)))
    #print(np.shape(np.array(test_out_2d)))






 






        
        
        
        
        
        
        
        