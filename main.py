import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import src.CoreDynamicsModel as cdm
import src.CorePhysicalModel as cpm
import util.Network as nw
import torch
import src.reactor_mpc_control as rmc

import util.FileSaveRead as fsr
import os
import pandas as pd
st.set_page_config(layout="wide")

#models_list = ["BiGRU","BiLSTM", "MLP","GRU","LSTM","SVR","RFR","PERT","REAL",  "NOISE"  ]

models_list = ["BiGRU","BiLSTM", "MLP","GRU","LSTM","SVR","PERT","REAL",  "NOISE"  ]


def render_power_diff(power_real,power_pert1,power_pert2,power_pert3):
    
    c1 = {'real':power_real,
         'pert1':power_pert1,
         'pert2':power_pert2,
         'pert3':power_pert3, }
    
    c2 = {
        'pert3':np.array(power_pert3)-np.array(power_real),
        'pert2':np.array(power_pert2)-np.array(power_real),
         'pert1':np.array(power_pert1)-np.array(power_real), }
    
    pd_c1 = pd.DataFrame(c1)
    pd_c2 = pd.DataFrame(c2)
    
    st.line_chart(pd_c1)
    st.area_chart(pd_c2)
    
    
    return
    
def train_ml_models(method, dataset = "",save_file = ""):
    
    train_in_3d,  train_out_2d, test_in_3d, test_out_2d = cpm.ReactorDynamics.getTraindata(dataset, timeStep =1)
    dim1, dim2, dim3 = train_in_3d.shape
    dim_out1, dimout2 = train_out_2d.shape
    
    dim1, dim2, dim3 = train_in_3d.shape
    dim_out1, dimout2 = train_out_2d.shape
    hidden_size = 200
    num_layers = 2
    LR = 0.001        
        
    if method == "BiGRU":

        
        model= nw.BiGRUnet(input_size=dim3, output_size=dimout2,
                           num_layers=num_layers, hidden_size= hidden_size)
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            model = model.cuda()

        nw.TrainingNetwork.train_net(model, 
                                     train_in_3d, train_out_2d,
                                     test_in_3d, test_out_2d , 
                                     LR =LR, save_file = save_file, dataset =dataset )
                                     
    elif method == "BiLSTM":                                     
                                     
        model= nw.BiLSTMnet(input_size=dim3, output_size=dimout2,num_layers=num_layers, hidden_size= hidden_size)
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            model = model.cuda()
        
        nw.TrainingNetwork.train_net(model, train_in_3d, train_out_2d,
                                     test_in_3d, test_out_2d , 
                                     LR =LR, save_file = save_file, dataset =dataset )

    elif method == "MLP":    


        timeStep = 1
        
        train_in_2d = train_in_3d.reshape((-1, dim3*timeStep))
        test_in_2d = test_in_3d.reshape((-1, dim3*timeStep))
        
        model= nw.DNNnet(input_size=dim3*timeStep, output_size=dimout2,num_layers=num_layers, hidden_size= hidden_size)
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            model = model.cuda()
        
        nw.TrainingNetwork.train_net(model, train_in_2d, train_out_2d,
                                     test_in_2d, test_out_2d , LR =LR, save_file = save_file, dataset =dataset )
        
    elif method == "GRU":    



        model= nw.GRUnet(input_size=dim3, output_size=dimout2,
                         num_layers=num_layers, hidden_size= hidden_size)
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            model = model.cuda()
        
        nw.TrainingNetwork.train_net(model, 
                                     train_in_3d, train_out_2d,
                                     test_in_3d, test_out_2d , LR =LR, save_file = save_file, dataset =dataset )
            
    elif method == "LSTM":    



        model= nw.LSTMnet(input_size=dim3, output_size=dimout2,num_layers=num_layers, hidden_size= hidden_size)
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            model = model.cuda()
        
        nw.TrainingNetwork.train_net(model, train_in_3d, train_out_2d,test_in_3d, test_out_2d , LR =LR, save_file = save_file, dataset =dataset )
            
    elif method == "RFR":    
        from sklearn.ensemble import RandomForestRegressor 
        timeStep = 1
        train_in_2d = train_in_3d.reshape((-1, dim3*timeStep))
        test_in_2d = test_in_3d.reshape((-1, dim3*timeStep))
        

        
        rf_vector = []

        for k in range(dimout2):
            random_forest = RandomForestRegressor(100)
            random_forest.fit( train_in_2d, train_out_2d[:, k])
            rf_vector.append(random_forest)
        
        
        
        
        rf_dict = {"rf_vector":rf_vector,
                   "dataset":dataset}
        
        import pickle  
        with open(save_file,'wb') as f:
            f.write( pickle.dumps(rf_dict ))     
            
    elif method == "SVR":    
        timeStep = 1
        train_in_2d = train_in_3d.reshape((-1, dim3*timeStep))
        test_in_2d = test_in_3d.reshape((-1, dim3*timeStep))

        svr_vector = []
        from sklearn import svm
        for k in range(dimout2):
            clf1=svm.SVR(tol =1.0e-6)
            clf1.fit(train_in_2d,train_out_2d[:, k])
            svr_vector.append(clf1) 
    
        svr_dict = {"svr_vector":svr_vector,
                   "dataset":dataset}
        
        import pickle
        with open(save_file,'wb') as f:
            f.write( pickle.dumps(svr_dict ))     
    
    elif method == "REAL" or  method == "PERT" or  method == "NOISE":
        ME_dict = {
                   "dataset":dataset}
        
        import pickle
        with open(save_file,'wb') as f:
            f.write( pickle.dumps(ME_dict ))         
    
    return 

label = ['1st group beta',
         '2th group beta',
         '3th group beta',
         '4th group beta',
         '5th group beta',
         '6th group beta',
         '1st group lambda',
         '2th group lambda',
         '3th group lambda',
         '4th group lambda',
         '5th group lambda',
         '6th group lambda',
         "neutron life ",
         "Gr(dk/k)",
         "core power(MW)",
         "fission fraction",
         "avg neutron speed(cm/s)",
         "decay constant of I(s-1)",
         "decay constant of Xe(s-1)",
         "decay constant of Pm(s-1)",
         "absorption Xsection of Xe(cm2)",
         "absorption Xsection of Sm(cm2)",
         "yeild of I",
         "yeild of Xe",
         "yeild of pm",
         "avg neutron level(n/cm3)",
         "fission Xsection(cm-1)",
         "Coolant heat transfer coefficient(MWs/C)",
         "Thermal conductivity of fuel(MW/C)",
         "Heat transfer coefficient of fuel coolant(MWs/C)",
         "Thermal conductivity of fuel(MW/C)",
         "Fuel temperature reactivity coefficient(dk/k/c)",
         "Coolant temperature reactivity coefficient(dk/k/c)",
         ]    

def render_data_param(max0,min0,dataTrainIn, aa):
    aa = aa+".png"
    #st.write(aa)
    if fsr.FileSaveRead.have_file(aa) :
        from PIL import Image
        image = Image.open(aa)        
        
        st.image(image)
    else:
    
        import matplotlib
        #matplotlib.use("Agg")
        import matplotlib.pyplot as plt      
        plt.rcParams["font.sans-serif"] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        #fig = plt.figure(figsize=(15,5))     
        for i in range(len(dataTrainIn)):
            for k in range(len(dataTrainIn[0])):
                for j in range(len(dataTrainIn[0][0])):
                    dataTrainIn[i][k][j] = dataTrainIn[i][k][j]*(max0[j]-min0[j])+min0[j]
        
        
        
        name_full_state = ["power","cr-1","cr-2","cr-3","cr-4","cr-5","cr-6","T_fuel","T_cool","I","Xe","Pm","Sm","pho","t_in"]
        dim1,dim2,dim3 = np.shape(np.array(dataTrainIn))
        data = pd.DataFrame(np.array(dataTrainIn).reshape(dim1*dim2, dim3), columns=name_full_state)
        #data.hist()
        
        
        from pandas.plotting import scatter_matrix
        axl = scatter_matrix(data,figsize=(10,10),diagonal="kde",grid=True,s=1)  
        #plt.show()    
    
        for i, axs in enumerate(axl):
            for j, ax in enumerate(axs):
                ax.set_xticks([])
                ax.set_yticks([])   
        plt.savefig(aa)
        
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
        
        
    
    
    #plt.close(fig)    
    
    return 



# generate the trainning data-set
def render_data(datatmp):
    
     
    datam = st.selectbox("data source operation", ["","new source","view source","delete source"]) 
    if datam == "new source":
        

        values = [0.00017, 0.00106, 0.00118, 0.00321, 0.00111, 0.00035 , #beta_list 
                 0.01249, 0.03156, 0.11063, 0.32256, 1.34188, 9.00518  ,#lambda_list
                 
                 2e-5,  #Lambda_time 12
                 0.0145,  #Gr 13
                 2500,  #p0  14
                 0.92, #ff  15
                 2.2e5, #v_neutron  16
                 2.9e-5,  #lamda_I 17
                 2.1e-5, #lamda_Xe 18
                 3.6e-6,  #lamda_Pm 19
                 
                 3.5e-18, #sigma_Xe 20
                 40800 *1e-24,  #sigma_Sm 21
                 0.056,  #gamma_I 22
                 0.003,  #gamma_Xe 23
                 1.08e-2,  #gamma_Pm 24
                 5e8,  #n0 25
                 0.3358, #sigma_fission 26
                 160./9.*1.0 + 54.022, #mu_c 27
                 26.3,  #mu_f 28
                 5./3.*1.0 +4.9333,   #omega 29
                 28*1.0 + 74 ,  #M 30
                 -3.24e-5, #alpha_f 31
                 -21.3e-5 ] #alpha_c 32
                           



        form = st.form(key='off_line_in')
        
        form.info("1.1 setup point core parameters")
        txt = form.text_input("data source file name", value=""),  
        
        for i in range(len(values)):
            if i%3 == 0:
                r1,r2,r3 = form.columns(3)
                values[i] =r1.number_input(label[i], value= values[i],format="%e")
                
            elif i%3 == 1:
                values[i] =r2.number_input(label[i], value= values[i],format="%e")
            elif i%3 == 2:
                values[i] =r3.number_input(label[i], value= values[i],format="%e")
    
        form.info("1.2 setup point core initial status")
        r1,r2 = form.columns(2)
        
        p0 = r1.number_input("power level(FP)",value= 1.0,min_value=0., max_value=1.2,format="%f")
        t0 = r2.number_input("inlet temperature (C)",value= 320.,min_value=200., max_value=400.,format="%f")
        t0 = 290.
        
        form.info("1.3 setup data source parameters")    
        r1,r2,r3 = form.columns(3)    
        npower = r1.number_input("number of power levels",value= 2,min_value=2, max_value=3000)
        nstep = r2.number_input("simulation for each power level",value= 11,min_value=11, max_value=3000)
        fangshi = r3.selectbox("generate method", ["Random large range search","Post critical perturbation"])
        
        submit_button = form.form_submit_button(label='submit and create data source file')
        
    

    
        if submit_button:
            coremodel = cpm.ReactorDynamics(nr_init = p0,
                                            t_inlet = t0,
                                            param = values
                                            )            
            
            
            if str(txt[0]) == "":
                st.error("data source file name not input")
                st.stop()
                pass
            else:
                filesave= datatmp +"/"+ str(txt[0])+".s"
                my_bar = st.progress(0)   
                if fangshi == "Random large range search":
                    max0, min0, dataTrainIn, dataTrainOut = coremodel.generate_data_pkl_random(my_bar,filesave,npower=npower, nstep=nstep)        
                elif fangshi == "Post critical perturbation":
                    max0, min0, dataTrainIn, dataTrainOut = coremodel.generate_data_pkl_critical(my_bar,filesave,npower=npower, nstep=nstep)        
                
                data = {}
                data["input"] = dataTrainIn
                data["output"] = dataTrainOut
                data["max0"] = max0
                data["min0"] = min0   
                data["nr_init"] = p0
                data["t_inlet"] = t0                
                data["npower"] = npower
                data["nstep"] = nstep                   
                data["param"] = values   
                data["fangshi"] = fangshi   

                fsr.FileSaveRead.write_pkl(filesave, data)
                render_data_param(max0,min0,dataTrainIn,filesave)   
                
    
    
                power_real,power_pert1,power_pert2,power_pert3 = coremodel.generate_different()
                render_power_diff(power_real,power_pert1,power_pert2,power_pert3)
                

    elif datam == "view source":
            
        
        q = []
        for root, dirs, files in os.walk(datatmp):  
            for name in files:
                if name[-1] == 's':
                    q.append(name)
        q.insert(0,"")           
        simulate_set = st.selectbox('select file', q )   
        if simulate_set != "":
            aa = datatmp+"/"+simulate_set
        
            data =fsr.FileSaveRead.read_pkl(aa)

            p0 = data["nr_init"] 
            t0 = data["t_inlet"]                
            npower = data["npower"] 
            nstep = data["nstep"]         
            values = data["param"]
            fangshi = data["fangshi"]  
            dataTrainIn = data["input"]
            dataTrainOut = data["output"]
            max0 = data["max0"]
            min0 = data["min0"]
            
            
            for i in range(len(values)):
                st.write(label[i],  values[i])
            
            st.write("power level(FP)",p0)
            st.write("inlet temperature(C)",t0)
            st.write("number of power levels",npower)
            st.write("simulation for each power level",nstep)
            st.write("generate method",fangshi)
            
            #render_data_param(max0,min0,dataTrainIn,aa)            
            
         
    elif datam == "delete source":   

        
        q = []
        for root, dirs, files in os.walk(datatmp):  
            for name in files:
                if name[-1] == 's':
                    q.append(name)
        q.insert(0,"")           
        
        form = st.form(key='off_line_delet')                
        simulate_set = form.selectbox('', q )   
        
        submit_button = form.form_submit_button(label='submit and delete source file')
        
        if submit_button:
            if str(simulate_set) == "":  
                st.error("non select file")
            else:
                    path = datatmp +"/"+str(simulate_set)
                    os.remove(path)
                    st.success("delete successful："+str(simulate_set))            
        
        
    else:
        pass
          




    return


# learning model 
def render_model(datatmp):


    form = st.form(key="off_line_two")    

    q = []
    for root, dirs, files in os.walk(datatmp):  
        for name in files:
            if name[-1] == "s":
                q.append(name)    
    
    simulate_set = form.selectbox( 'select data source file', q ) 
    r1c,r2c,r3c= form.columns(3)    
    ml_options = form.multiselect ( 'select ML method', 
                                 models_list,  models_list )  
    
    retrain = r1c.selectbox ( 'overwrite calculation results', 
                                 ("No","Yes") )      
    

    submit_button = form.form_submit_button(label='submit and calculation')
    
    if submit_button:
        dataset_path = ""
        Info = "selected "+str(simulate_set)+" as ML model input"
        st.info(Info)        
        dataset_path = datatmp + "/"+str(simulate_set)
    
        data =fsr.FileSaveRead.read_pkl(dataset_path)

        p0 = data["nr_init"] 
        t0 = data["t_inlet"]                
        npower = data["npower"] 
        nstep = data["nstep"]         
        values = data["param"]        
        
        cc_train_loss = {}
        cc_test_loss = {}
        for i in range(len(ml_options)):
            model_file = datatmp + "/"+str(simulate_set)+"_"+ml_options[i]
            
            havefile = fsr.FileSaveRead.have_file(model_file) 
            
            if havefile:
                if retrain == "Yes":
                    with st.spinner(ml_options[i]+"model file have existence,retraining and overwrite model file ... "):
                        train_ml_models(ml_options[i], dataset =dataset_path, save_file = model_file)
                else:
                    st.success(ml_options[i]+"model file have existence")
                    
                   
             
             
            else:
                with st.spinner(ml_options[i]+"model file no existence, training ... "):
                    train_ml_models(ml_options[i], dataset =dataset_path, save_file = model_file)


            if ml_options[i] in ["BiLSTM","BiGRU","GRU","LSTM","MLP"]:
                checkpoint = torch.load(model_file) 
                
                trian_loss = checkpoint['train_loss']
                test_loss = checkpoint['test_loss']
              
                c = {'train_loss':trian_loss,
                      'test_loss':test_loss,
                      }
                
                cc_train_loss[ml_options[i]] = trian_loss
                cc_test_loss[ml_options[i]] = test_loss
                

                x_ti = list(range(len(trian_loss)))
                x_lists = [x_ti, x_ti]
                y_lists = [trian_loss,test_loss ]
                label_lists = [ml_options[i]+" train dataset", ml_options[i]+" test dataset"]
                
                rmc.xy_plot(x_lists, y_lists, label_lists, "epoch", "log of loss values", [], [], [], [], [], [])


        if cc_train_loss:
            import matplotlib.pyplot as plt        
            fig, ax = plt.subplots()
            
            x_lists = []
            y_lists = []
            label_lists = []
            for value in cc_train_loss:
                ax.plot(range(len(cc_train_loss[value])), cc_train_loss[value])
                
                while len(cc_train_loss[value])< 10000:
                          cc_train_loss[value].append(cc_train_loss[value][-1])
                          cc_test_loss[value].append(cc_test_loss[value][-1])
                
                x_lists.append(list(range(len(cc_train_loss[value]))) )
                y_lists.append(cc_train_loss[value] )
                label_lists.append(value )
                
            #st.write(fig)     
            
            if len(y_lists) >0:
                rmc.xy_plot(x_lists, y_lists, label_lists, "epoch", "log of loss values", [], [], [], [], [], [])

    
                import pandas as pd
                pd_c = pd.DataFrame(cc_train_loss)
                st.line_chart(pd_c)     
        
                pd_c = pd.DataFrame(cc_test_loss)
                st.line_chart(pd_c) 
         
        
        

                 
    
            
        
    return
    



# setup control target and MPC
def render_mpc_no_filter( datatmp):

    datam = st.selectbox("MPC without uncertainty", ["","new target","delete target","select a target and control"],key="cccc") 

        
        
    if datam == "new target":
        r1,r2 = st.columns((2,1))
        name_full_state = ["power","cr-1","cr-2","cr-3","cr-4","cr-5","cr-6","T_fuel","T_cool","I","Xe","Pm","Sm"] #full status
        variables = r1.multiselect ( 'view variables', name_full_state )         
        ntimes = r1.selectbox("time steps of target curve", [2,4,6,8,10])

        q = []
        for root, dirs, files in os.walk(datatmp):  
            for name in files:
                if name[-1] == "s":
                    q.append(name)    
        
        simulate_set = st.selectbox( 'select data source file for initial', q )            
        aa = datatmp+"/"+simulate_set
    
        data =fsr.FileSaveRead.read_pkl(aa)

        p0 = data["nr_init"] 
        t0 = data["t_inlet"]                
        values = data["param"]       
        
        coremodel = cpm.ReactorDynamics(nr_init = p0,
                                        t_inlet = t0,
                                        param = values
                                        )            
                    

        
        form = st.form(key='target_line')
        txt = form.text_input("target file name", value=""),  

        variables_weight = []
        variablee_t_list = []
        variablee_v_list = []
        variables_index = []
        for i in range(len(variables)):
            vindx = name_full_state.index(variables[i])
            variables_index.append(vindx)
            r1,r2 = form.columns((2,1))
            r1.info("setup control target value，for "+variables[i])
            weigt = r2.number_input( "weight for variable",  value=1.0,key=variables[i]+str(1)) 
            variables_weight.append(weigt)
            
            r1,r2,r3,r4 = form.columns((1,2,1,2))
            t1 = r1.number_input( "time(s)",  value=10, key=variables[i]+str(2) ) 
            s1 = r2.number_input( "target",   value=coremodel.core.blance_state[vindx],key=variables[i]+str(3) ) 
            t2 = r3.number_input( "time(s)",  value=20, key=variables[i]+str(4) ) 
            s2 = r4.number_input( "target",   value=coremodel.core.blance_state[vindx], key=variables[i]+str(5) )            
            
            t_list = [0, t1,t2]
            v_list = [coremodel.core.blance_state[vindx], s1,s2]
            if ntimes> 2:
                for j in range(int((ntimes-2)/2)):
                    tmp1 = r1.number_input( "",  value=30+20*j,key="csd1"+str(i)+str(j)) 
                    smp1 = r2.number_input( "",   value=coremodel.core.blance_state[vindx], key="csd2"+str(i)+str(j)) 
                    tmp2 = r3.number_input( "",  value=40+20*j,key="csd3"+str(i)+str(j)) 
                    smp2 = r4.number_input( "",   value=coremodel.core.blance_state[vindx], key="csd4"+str(i)+str(j))   
                    t_list.extend([tmp1, tmp2])
                    v_list.extend([smp1, smp2])                      
            
            variablee_t_list.append(t_list)
            variablee_v_list.append(v_list)
            form.write("---")
            
        r1,r2 = form.columns((3,3))  
        min1,max1 = r1.slider("control boundard of control rod reactivity (ppm)", min_value=-2000., max_value=2000., value=(-1000.,1000.),   key="r1")
        min2,max2 = r2.slider("ontrol boundard of inlet coolant temperature(C)", min_value=200., max_value=400., value=(290., 290.),    )         
                
                
        submit_button = form.form_submit_button(label='submit and generate target file')
        
        if submit_button:
            
            name_full_state = ["power","cr-1","cr-2","cr-3","cr-4","cr-5","cr-6","T_fuel","T_cool","I","Xe","Pm","Sm"] 

            for i in range(len(variablee_v_list)):
                 rmc.xy_plot([variablee_t_list[i]],[variablee_v_list[i]], [name_full_state[variables_index[i]] ], "times", name_full_state[variables_index[i]],
                 [],[],[],[],[],[])
                


            data = {}
            data["variables_weight"] = variables_weight
            data["variablee_t_list"] = variablee_t_list
            data["variablee_v_list"] = variablee_v_list
            data["variables_index"] = variables_index  
            data["variables_label"] = name_full_state
            
            data["min1"] =min1
            data["max1"] =max1
            data["min2"] =min2
            data["max2"] =max2
            

            if str(txt[0]) == "":
                st.error("target file name no input")
                st.stop()
                pass
            else:
                filesave= datatmp +"/"+ str(txt[0])+".t"
                fsr.FileSaveRead.write_pkl(filesave, data)

            st.success("generate target file success")


    if datam == "delete target":

        form = st.form(key='target_line1')
        form.info(" delete a target file")

        q = []
        for root, dirs, files in os.walk(datatmp):  
            for name in files:
                if name[-1] == "t":
                    q.append(name)
                
        target_set = form.selectbox('select file for delete', q )          
        submit_button = form.form_submit_button(label='submit and delete')
        if submit_button:
            path = datatmp+"/"+str(target_set)
            os.remove(path)
            st.success("delete success："+str(target_set))  


    if datam == "select a target and control":
        

        q_s = []
        q_m = []
        q_t = []        
        for root, dirs, files in os.walk(datatmp):  
            for name in files:
                if name[-1] == "s":
                    q_s.append(name)  
                if name[-1] == "m":                    
                    q_m.append(name)                     
                if name[-1] == "t":                    
                    q_t.append(name)                      
           
                    
        form = st.form(key='target_line2')     
            
        simuate_set = form.selectbox( 'select data source for initial', q_s ) 
        dataset_path = datatmp+"/"+  str(simuate_set)   
        data =fsr.FileSaveRead.read_pkl(dataset_path)
        values = data["param" ]      

        r1,r2 = form.columns((2,1))
        ensemble_options = r1.multiselect ( 'ensemble multi-models for MPC without uncertainty',
                                     models_list )   
        
        learner = r2.selectbox( 'second-layer learner', ["weight", 'dnn'] )  


        r1,r2 = form.columns((2,1))
        target_set = r1.selectbox( 'select a target file', q_t ) 
        online_plot = "Yes"

        submit_button = form.form_submit_button(label='submit')
        if submit_button:    

            data = fsr.FileSaveRead.read_pkl(datatmp+"/"+str(target_set))
    
            variables_weight= data["variables_weight"]
            variablee_t_list=data["variablee_t_list"]
            variablee_v_list=data["variablee_v_list"]
            variables_index =data["variables_index"]
            name_full_state = data["variables_label"]
            
            min1= data["min1"]
            max1=data["max1"]
            min2=data["min2"]
            max2 =data["max2"]    
    
            u_lower_uper = [[min1*1.e-5,max1*1.e-5],[min2, max2]]
            
            
            status_list_cv_f = []
            goal_list_cv_f = []
            control_list_cv_f = []
            mix_weight_list_cv_f = []
            
            
            c = {}
            for value in ensemble_options :
                c[value] = datatmp+"/"+ str(simuate_set) +"_"+value      
            
            ax,bx, cx,dx = rmc.run_MPC_nofliter(  
                            c,   
                            variables_index, variables_weight, variablee_t_list, 
                            variablee_v_list, name_full_state,
                            u_lower_uper,
                            learner, online_plot,
                        param = values, dataset = dataset_path)
            

            status_list_cv_f.append(ax)
            goal_list_cv_f.append(bx)
            control_list_cv_f.append(cx)
            mix_weight_list_cv_f.append(dx)
            

            

            if len(ensemble_options) > 1:
                with st.spinner("mpc for each indivial model in the selected model list"):
                    for value in ensemble_options:
                        model_file = datatmp + "/"+str(simuate_set)+"_"+value
                        c_tmp = {}
                        c_tmp[value] = model_file
                        ax,bx, cx,dx = rmc.run_MPC_nofliter(  
                                        c_tmp,   
                                        variables_index, variables_weight, variablee_t_list, 
                                        variablee_v_list, name_full_state,
                                        u_lower_uper,
                                        learner, "No",
                                    param = values,dataset = dataset_path)   
                        
                        status_list_cv_f.append(ax)
                        goal_list_cv_f.append(bx)
                        control_list_cv_f.append(cx)
                        mix_weight_list_cv_f.append(dx)
                        
                    
            rmc.ppppppppp_v(status_list_cv_f,goal_list_cv_f, control_list_cv_f,mix_weight_list_cv_f,
                 u_lower_uper,name_full_state,variables_index,c)






def render_MPC(datatmp):
    #####################################################


    q_s = []
    q_m = []
    q_t = []        
    for root, dirs, files in os.walk(datatmp):  
        for name in files:
            if name[-1] == "s":
                q_s.append(name)  
            if name[-1] == "m":                    
                q_m.append(name)                     
            if name[-1] == "t":                    
                q_t.append(name)      
    
    if len(q_s) == 0:
    	st.stop()

    simulate_set = st.selectbox( 'select data source file for initial', q_s ) 
    dataset_path = datatmp+"/"+  str(simulate_set)   
    data =fsr.FileSaveRead.read_pkl(dataset_path)
    values = data["param"]   
    
    
    
    form = st.form(key='render_MPC')    
    ensemble_options = form.multiselect ( 'ensemble multi-models for MPC with uncertainty',
                                 models_list )   

    r1,r2,r3,r4 = form.columns(4)    
    target_set = r1.selectbox( 'select a target file', q_t )     
    learner = r2.selectbox( 'second-layer learner', ["weight", 'dnn'] ) 
    system_dev = r4.selectbox( 'disable systematic deviation correction', ["yes", 'no'] ) 
    name_full_state = ["power","cr-1","cr-2","cr-3","cr-4","cr-5","cr-6","T_fuel","T_cool","I","Xe","Pm","Sm"] 
    
    r1,r2,r3,r4 = form.columns(4)  
    e_s = r1.number_input("power measure systematic error", value=0.0,  min_value = -0.5, max_value= 0.5)  
    e_sigma = r2.number_input("power measure standard deviation", value=0.01,  min_value = 0.0, max_value= 0.5)      
    partical_number = r3.number_input("Number of filtered particles", value=100)
    comparison = r4.selectbox( 'compare each basic learners', ["yes", 'no'] )  


    submit_button = form.form_submit_button(label='submit')
    if submit_button:    

        data = fsr.FileSaveRead.read_pkl(datatmp+"/"+str(target_set))

        variables_weight= data["variables_weight"]
        variablee_t_list=data["variablee_t_list"]
        variablee_v_list=data["variablee_v_list"]
        variables_index =data["variables_index"]
        name_full_state = data["variables_label"]
        
        min1= data["min1"]
        max1=data["max1"]
        min2=data["min2"]
        max2 =data["max2"]    

        u_lower_uper = [[min1*1.e-5,max1*1.e-5],[min2, max2]]
        
        if len(ensemble_options) <= 0:
            st.error("need select modes")
            st.stop()
        

        nomralized_status = "yes"
            

        c = {}
        for value in ensemble_options : 
                model_file = datatmp + "/"+str(simulate_set)+"_"+value
                c[value] = model_file
            

        status_list_cv,goal_list_cv, control_list_cv,mix_weight_list_cv,measure_cv, status_prd_cv,status_est_cv,status_est_P_cv=rmc.run_MPC(  c,   
                        variables_index, variables_weight, variablee_t_list, variablee_v_list,name_full_state,
                        u_lower_uper,
                        learner, partical_number,nomralized_status,e_s,e_sigma,system_dev,
                    param = values,dataset = dataset_path)
        
        
        rmc.mpc_with_filter_xy_plot(u_lower_uper,name_full_state,variables_index, 
                                    status_list_cv,goal_list_cv, control_list_cv,c,
                            mix_weight_list_cv,measure_cv, status_prd_cv,status_est_cv,status_est_P_cv)        
        
        
        
        if len(ensemble_options) > 1 and comparison == "yes" :
            for value in ensemble_options:
                model_file = datatmp + "/"+str(simulate_set)+"_"+value
                c_tmp = {}
                c_tmp[value] = model_file        
        
                status_list_cv,goal_list_cv, control_list_cv,mix_weight_list_cv,measure_cv, status_prd_cv,status_est_cv,status_est_P_cv=rmc.run_MPC(  c_tmp,   
                                variables_index, variables_weight, variablee_t_list, variablee_v_list,name_full_state,
                                u_lower_uper,
                                learner, partical_number,nomralized_status,e_s,e_sigma,system_dev,
                            param = values,dataset = dataset_path)
                
                
                rmc.mpc_with_filter_xy_plot(u_lower_uper,name_full_state,variables_index, 
                                            status_list_cv,goal_list_cv, control_list_cv,c_tmp,
                                    mix_weight_list_cv,measure_cv, status_prd_cv,status_est_cv,status_est_P_cv)              
        
        
                
    return    
    



    


datatmp = "db"
st.header("Learning and Ensemble based MPC with Differential Dynamic Programming for Nuclear Power Autonomous Control")

fsr.FileSaveRead.creatpath(datatmp)
tab1, tab2, tab3, tab4 = st.tabs(["1 data source", "2 model generation", "3 MPC with ensemble learner and without uncertainty",  "4 MPC with ensemble learner and Enkf"])

with tab1:
    values = render_data(datatmp)

with tab2:        
    dataset_path = render_model(datatmp)
with tab3:        
    render_mpc_no_filter(datatmp)

with tab4:                  
    render_MPC(datatmp)               
        
        
        
