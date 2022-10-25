import sys

sys.path.append('./../')    
sys.path.append('./../../')   
sys.path.append('./../../../')   
sys.path.append('./../../../../')  

import torch
from torch import nn
import torch.nn.functional as F
from numpy.linalg import norm, lstsq
from random import sample
from sklearn.cluster  import KMeans
from torch.autograd import Variable
import util.FileSaveRead as fsr
import util.FileSaveRead_torch as fsr_t
import streamlit as st


def weiboll_loss(ab_pred,y_true ):
    y_ = y_true[:, 0]
    u_ = y_true[:, 1]
    a_ = ab_pred[:, 0]
    b_ = ab_pred[:, 1]

    hazard0 = torch.pow((y_ + 1e-35) / a_, b_)
    hazard1 = torch.pow((y_ + 1) / a_, b_)

    return -1 * torch.mean(u_ * torch.log(torch.exp(hazard1 - hazard0) - 1.0) - hazard1)    




class LSTM_WTT(nn.Module):
    def __init__(self, input_size=24, output_size=2,num_layers=1, hidden_size= 10 ):
        super(LSTM_WTT, self).__init__()

        self.rnn = nn.LSTM(  
            input_size=input_size,
            hidden_size=hidden_size,     # rnn hidden unit
            num_layers=num_layers,       #  RNN layers
            batch_first=True,   # input & output e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(hidden_size, output_size)

        
    def activate(self, ab):
        a = torch.exp(ab[:, 0])
        b = F.sigmoid(ab[:, 1])
    
        a = torch.reshape(a, (a.shape[0], 1))
        b = torch.reshape(b, (b.shape[0], 1))
    
        return torch.cat((a, b), axis=1)


    def output_lambda(self,ab, init_alpha=1.0, max_beta_value=5.0, max_alpha_value=None):
        """Elementwise (Lambda) computation of alpha and regularized beta.
    
            Alpha: 
            (activation) 
            Exponential units seems to give faster training than 
            the original papers softplus units. Makes sense due to logarithmic
            effect of change in alpha. 
            (initialization) 
            To get faster training and fewer exploding gradients,
            initialize alpha to be around its scale when beta is around 1.0,
            approx the expected value/mean of training tte. 
            Because we're lazy we want the correct scale of output built
            into the model so initialize implicitly; 
            multiply assumed exp(0)=1 by scale factor `init_alpha`.
    
            Beta: 
            (activation) 
            We want slow changes when beta-> 0 so Softplus made sense in the original 
            paper but we get similar effect with sigmoid. It also has nice features.
            (regularization) Use max_beta_value to implicitly regularize the model
            (initialization) Fixed to begin moving slowly around 1.0
    
            Assumes tensorflow backend.
    
            Args:
                x: tensor with last dimension having length 2
                    with x[...,0] = alpha, x[...,1] = beta
    
            Usage:
                model.add(Dense(2))
                model.add(Lambda(output_lambda, arguments={"init_alpha":100., "max_beta_value":2.0}))
            Returns:
                A positive `Tensor` of same shape as input
        """
        a = ab[:, 0]
        b = ab[:, 1]
    
        # Implicitly initialize alpha:
        if max_alpha_value is None:
            a = init_alpha * torch.exp(a)
        else:
            a = init_alpha * torch.clip(x=a, min_value=torch(1e-30),
                                    max_value=torch(max_alpha_value))
    
        m = max_beta_value
        if m > 1.05:  # some value >>1.0
            # shift to start around 1.0
            # assuming input is around 0.0
            _shift = np.log(m - 1.0)
    
            b = F.sigmoid(b - _shift)
        else:
            b = F.sigmoid(b)
    
        # Clipped sigmoid : has zero gradient at 0,1
        # Reduces the small tendency of instability after long training
        # by zeroing gradient.
        b = m * torch.clip(x=b, min_value=torch(1e-30), max_value=torch(1. - 1e-30))
    
       
    
        return torch.cat((a, b), axis=1)








    def forward(self, x):  #  hidden state 
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, output_size)
        r_out,  (h_n, h_c)  = self.rnn(x, None)   # h_state  RNN 

        #outs = []    # 
        #for time_step in range(r_out.size(1)):    #  output
        #    outs.append(self.out(r_out[:, time_step, :]))
        out = self.out(r_out[:, -1, :])
        out = self.activate(out)
        return out




# 
class LSTMnet(nn.Module):
    def __init__(self, input_size=3, output_size=100,num_layers=30, hidden_size= 1000 ):
        super(LSTMnet, self).__init__()

        self.rnn = nn.LSTM(  #
            input_size=input_size,
            hidden_size=hidden_size,     # rnn hidden unit
            num_layers=num_layers,       # 
            batch_first=True,   # input & output  e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):  #
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, output_size)
        r_out,  (h_n, h_c)  = self.rnn(x, None) 

        #outs = []   
        #for time_step in range(r_out.size(1)):   
        #    outs.append(self.out(r_out[:, time_step, :]))
        out = self.out(r_out[:, -1, :])
        return out

# 
class BiLSTMnet(nn.Module):
    def __init__(self, input_size=3, output_size=1,num_layers=3, hidden_size= 10 ):
        super(BiLSTMnet, self).__init__()

        self.rnn = nn.LSTM(  # 
            input_size=input_size,
            hidden_size=hidden_size,     # rnn hidden unit
            num_layers=num_layers,       # 
            batch_first=True,   # input & output  e.g. (batch, time_step, input_size)
            bidirectional=True
        )
        self.out = nn.Linear(hidden_size*2, output_size)

    def forward(self, x):  #
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, output_size)
        r_out,  (h_n, h_c)  = self.rnn(x, None)   # h_state 

        #outs = []    
        #for time_step in range(r_out.size(1)):   
        #    outs.append(self.out(r_out[:, time_step, :]))
        out = self.out(r_out[:, -1, :])
        return out


class GRUnet(nn.Module):     
    def __init__(self, input_size=3, output_size=1,num_layers=3, hidden_size= 10 ):
        super(GRUnet, self).__init__()        
        self.rnn = nn.GRU( 
                input_size=input_size, 
                hidden_size=hidden_size,  
                num_layers=num_layers, 
                batch_first=True,           
                bidirectional=False  )
        
        self.out = nn.Sequential(           
                nn.Linear(hidden_size, output_size)        )     
        
    def forward(self, x):        
        r_out, _ = self.rnn(x, None)  
        out = self.out(r_out[:, -1, :])         
        return out


class BiGRUnet(nn.Module):     
    def __init__(self, input_size=3, output_size=1,num_layers=3, hidden_size= 10 ):
        super(BiGRUnet, self).__init__()        
        self.rnn = nn.GRU( 
                input_size=input_size, 
                hidden_size=hidden_size,  
                num_layers=num_layers, 
                batch_first=True,           
                bidirectional=True  )
        
        self.out = nn.Sequential(           
                nn.Linear(hidden_size*2, output_size)        )     
        
    def forward(self, x):        
        r_out, _ = self.rnn(x, None)  
        out = self.out(r_out[:, -1, :])         
        return out


class DNNnet(nn.Module):
    def __init__(self, input_size=3, output_size=100,num_layers=30, hidden_size= 10, dropout = 0.5, types="ReLU"):
        super(DNNnet, self).__init__()
        self.layers = torch.nn.Sequential()
    
            
        self.layers.add_module("linear_0", nn.Linear(input_size, hidden_size) )

        if types == "ReLU":
            self.layers.add_module("relu_0", nn.ReLU() )
        elif types == "Sigmoid":
            self.layers.add_module("Sigmoid_0", nn.Sigmoid() )
        elif types == "Tanh":
            self.layers.add_module("Tanh_0", nn.Tanh() )
  
            
            
        for i in range(num_layers-1):
            labs1 = "linear_"+str(i+1)
            labs2 = "jihuo_" + str(i+1)
            labs3 = "dropout_" + str(i+1)
            self.layers.add_module(labs1, nn.Linear(hidden_size, hidden_size) )            
            #self.layers.add_module( labs3, nn.Dropout(dropout) )
  
            if types == "ReLU":
                self.layers.add_module(labs2, nn.ReLU() )
            elif types == "Sigmoid":
                self.layers.add_module(labs2, nn.Sigmoid() )
            elif types == "Tanh":
                self.layers.add_module(labs2, nn.Tanh() )                

        # last layer
        self.layers.add_module("linear_f", nn.Linear(hidden_size, output_size, bias=False) )     

            
    def forward(self, x):
        x = self.layers.forward(x)
            
        return x


###########################################################################################################
        







######################################################################################
class TrainingNetwork(object):
    def __init__():   
        pass    
    
    # 
    @staticmethod
    def data_loader(train_in, 
                    train_out, 
                    BATCH_SIZE,  
                    shuffle=False):
        
        this_type_str = type(train_in)
        if this_type_str is not np.ndarray:
            train_in = np.array(train_in)
            train_out = np.array(train_out)

        
        import torch.utils.data as Data    
        
        train_in_torch = torch.from_numpy(train_in)
        train_out_torch = torch.from_numpy(train_out)
        

        torch_dataset = Data.TensorDataset(train_in_torch, train_out_torch )
        

        data_train = Data.DataLoader(
            dataset=torch_dataset,      # torch TensorDataset format
            batch_size=BATCH_SIZE,      # mini batch size
            shuffle=shuffle,               # 
            num_workers=0,              # 
        )    
        return data_train

    

    @staticmethod   
    def train_net( model, 
                  train_in, 
                  train_out, 
                  test_in,
                  test_out, 
                  LR = 0.001, 
                  EPOCH =10000,
                  EndPoch = 600, 
                  save_internal = 500,
                  save_file = "",
                  dataset = "",
                  criterion_fun = "",
                  if_log = True,
                  ):
        import streamlit as st
        if torch.cuda.is_available():
            model = model.cuda()
        

        ############################################################        
        dim1 = train_in.shape[0]
        BATCH_SIZE = dim1
        data_train = TrainingNetwork.data_loader(train_in, train_out, BATCH_SIZE )
        

        ##############################################################################        

        dim1 = test_in.shape[0]
        ############################################################
        BATCH_SIZE = dim1
        data_test = TrainingNetwork.data_loader(test_in, test_out, BATCH_SIZE )
        ##############################################################################        
        
        EPOCH =EPOCH      
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)   # optimize all parameters
        
        
        criterion = nn.MSELoss()
            
        is_cuda = torch.cuda.is_available() 


        # training and testing
        datas_train = []
        datas_test = []
        lossd = 0
        cout = 0
        lossfin = float(1.0e8)
        
        
        import math
        my_bar = st.progress(0) 
        my_empty = st.empty()
        trian_loss = []
        test_loss = []
        for epoch in range(EPOCH):
             #st.write(epoch)
             tot = 0
             model.train()

             for step, (d_in, d_out) in enumerate(data_train) :

                if is_cuda:
                    d_in = Variable(d_in.cuda())
                    d_out = Variable(d_out.cuda())
                else:
                    d_in = Variable(d_in)
                    d_out = Variable(d_out)     
    
                optimizer.zero_grad() 
                prediction = model(d_in)
                
                if criterion_fun == "weiboll":
                    loss = weiboll_loss(prediction, d_out)
                elif criterion_fun == "":
                    loss = criterion(prediction, d_out)   # cross entropy loss
                
    
                # clear gradients for this training step
                loss.backward()                 # backpropagation, compute gradients
                optimizer.step()                # apply gradients
                
                if is_cuda:
                    datas_train = prediction.cuda().data.cpu().numpy().tolist()
                else:
                    datas_train = prediction.data.numpy().tolist()
                    
                tot += loss.data
            
             if if_log:
                 trian_loss.append(math.log(float(tot) ) )
             else:
                 trian_loss.append((float(tot) ) )
            
             model.eval() 
             if criterion_fun == "weiboll":
                lossd,_ = TrainingNetwork.get_fit_weiboll(model, data_test)   
             elif criterion_fun == "":   
                lossd,_ = TrainingNetwork.get_fit(model, data_test)
            
             if if_log:
                 test_loss.append(math.log(float(lossd)) )     
             else:
                 test_loss.append((float(lossd)) )             
   
             my_bar.progress(int((epoch+1.0)/EPOCH*100))                
                
             if lossfin > tot:
                 lossfin = tot
                 cout = 0

                 c = {'train_loss':trian_loss,
                      'test_loss':test_loss,
                      }
                 import pandas as pd
                 pd_c = pd.DataFrame(c)
                 my_empty.line_chart(pd_c)                 
                 
                 fsr_t.FileSaveRead.save_torch_model(epoch, model, optimizer,trian_loss,test_loss, dataset, save_file)        
             else:
                 cout += 1
             
             
             if cout > EndPoch:
                 break;             

         
        power_train = []   
        for i in range(len(datas_train)):
            power_train.append(datas_train[i][0])         
             
        power_test = []   
        for i in range(len(datas_test)):
            power_test.append(datas_test[i][0])             
             
        return power_train, power_test
    
    @staticmethod
    def get_fit(model,data_test1 ):
        model.eval()
        err_test1 = 0
        is_cuda = torch.cuda.is_available()    
        out = None
        for test_in_one, test_out_one in data_test1:
            if is_cuda:
                test_in_one = Variable(test_in_one.cuda() )
                test_out_one = Variable(test_out_one.cuda() )
            else:
                test_in_one = Variable(test_in_one)
                test_out_one = Variable(test_out_one)    
    
            out = model(test_in_one)
            err_test1 = np.mean(pow(out.cpu().detach().numpy() - test_out_one.cpu().detach().numpy(), 2.))
            
        return err_test1, out.cpu().detach().numpy()
    
    @staticmethod
    def get_fit_weiboll(model,data_test1 ):
        model.eval()
        err_test1 = 0

        is_cuda = torch.cuda.is_available()    
        out = None
        for test_in_one, test_out_one in data_test1:
            if is_cuda:
                test_in_one = Variable(test_in_one.cuda() )
                test_out_one = Variable(test_out_one.cuda() )
            else:
                test_in_one = Variable(test_in_one)
                test_out_one = Variable(test_out_one)    
    
            out = model(test_in_one)
            loss = weiboll_loss(out, test_out_one)
            err_test1 = loss.data
            
        return err_test1, out.cpu().detach().numpy()
    




  
    
    
    
    
    
    
    
    
    
    
    
    
    
    