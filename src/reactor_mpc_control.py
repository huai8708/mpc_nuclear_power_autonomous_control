import numpy as np
import torch
import random
import src.CoreDynamicsModel as cdm
import src.CorePhysicalModel as cpm
import src.ModelPredictControl as mpc
import util.FileSaveRead as fsr
import src.PointCore_critical as pcc
import src.DataAssimilationMethod as dam
import pandas as pd


def xy_plot(x_lists, y_lists, label_lists,
            xlabel, ylabel,
            xs_lists, ys_lists, labels_lists,
            x_updw, y_up, y_dw,
            empty = None,
            ):
    import streamlit as st
    import matplotlib.pyplot as plt  

    plt.rcParams["font.sans-serif"] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(15,5)) 
    
    
    if len(x_updw) > 0:
        plt.plot(x_updw, y_dw, c='k',linestyle=':', lw=1, label='2$\sigma$')
        plt.plot(x_updw, y_up, c='k',linestyle=':', lw=1)
        plt.fill_between(x_updw, y_dw, y_up, facecolor='r', alpha=.1)

    linestyle = ["-","--","-.","-","--","-.",":"]
    color = ["b","g","r","m","y","k","b","g","r","c","m","y","k"]
    markers = ["o","v","<",">","1","2","3","4","s","p","*","h",".",",",]
    labelindex = ["measured","predicted","estimated", "real","target"]

    mpc = True
    for i in range(len(y_lists)):
        try:
            idx = labelindex.index(label_lists[i])
        except:
            idx = 2
        if mpc :
            idx = i
        plt.plot(list(x_lists[i]), y_lists[i], c=color[idx],linestyle=linestyle[idx], lw=1, label=label_lists[i])
    
    for i in range(len(ys_lists)):
        try:
            idx = labelindex.index(labels_lists[i])
        except:
            idx = 2            
        if mpc :
            idx = i            
        plt.scatter(list(xs_lists[i]), ys_lists[i], c=color[idx],marker= markers[idx], lw=1, label=labels_lists[i])
        
            
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    
    
    if empty is None:
        st.pyplot(fig)
    else:
        empty.pyplot(fig)
        
    plt.close(fig)    
    return 







def xy_plot1(x_lists, y_lists, label_lists,
            xlabel, ylabel,
            xs_lists, ys_lists, labels_lists,
            x_updw, y_up, y_dw,
            empty = None,
            ):
    import streamlit as st
    import plotly.graph_objs as go    
    fig = go.Figure()
    
    if len(x_updw) > 0:
        fig.add_trace(go.Scatter(
            x = x_updw,
            y = y_dw,
            mode = None,
            opacity=0.99,
            name = '95%-95% dw'
        ) )   
        
        
        fig.add_trace(go.Scatter(
            x = x_updw,
            y = y_up,
            opacity=0.99,
            mode = None,
            fill='tonexty',
            name = '95%-95% up'
        ) )     
        
    

    for i in range(len(y_lists)):
        fig.add_trace(go.Scatter(
                x = list(x_lists[i]),
                y = y_lists[i],
                name = label_lists[i],
                mode  = "lines",
                ) )
        
        
    
    for i in range(len(ys_lists)):
        fig.add_trace(go.Scatter(
                x = list(xs_lists[i]),
                y = ys_lists[i],
                name = labels_lists[i],
                mode  = "markers",
                
                ) )    
    

    fig.update_layout(
        xaxis= { 'title': {'text': xlabel}},
        yaxis= {'title':  {'text': ylabel}   }  )
    
    if empty is None:
        st.plotly_chart(fig, use_container_width=True)
    else:
        empty.plotly_chart(fig, use_container_width=True)
    return 









def xy_stackplot(x, y_lists, labels):
    
    import streamlit as st
    import matplotlib
    
    from matplotlib import pyplot as plt
    
    fig = plt.figure(figsize=(15,5)) 
    
    
    '''
    if len(y_lists) == 2:
        plt.stackplot(x, y_lists[0], y_lists[1],  labels=labels)
    elif len(y_lists) == 3:
        plt.stackplot(x, y_lists[0], y_lists[1], y_lists[2], labels=labels)
    elif len(y_lists) == 4:
        plt.stackplot(x, y_lists[0], y_lists[1], y_lists[2],y_lists[3], labels=labels)
    elif len(y_lists) == 5:
        plt.stackplot(x, y_lists[0], y_lists[1], y_lists[2],y_lists[3], y_lists[4],labels=labels)                
    '''
    plt.stackplot(x, [0.3]*len(x),  [0.5]*len(x),  labels=["lab1","lab2"])


    plt.xlabel("step")
    plt.ylabel("weights")
    plt.legend()

    
    st.plotly_chart(fig, use_container_width=True)
    
    return 




def get_Target(t, powerTarget):
    
    for i in range(powerTarget.shape[0]-1):
        if t >= powerTarget[i,0] and t < powerTarget[i+1,0]:
            slope = (powerTarget[i+1,1] - powerTarget[i,1])/ (powerTarget[i+1,0] - powerTarget[i,0])
            target = slope* (t - powerTarget[i,0]) + powerTarget[i,1]
            break
        else:
            target=powerTarget[-1,1]
            
    return target    



import streamlit as st


def cost_function_update(TIMESTEPS,N_BATCH, nx, nu, timeT, 
                  v_weight, v_index, v_times, v_values ):
    ############################################## begin
    Q_vec = []
    p_vec = []


    for j in range(TIMESTEPS):
        goal_weight_np = np.zeros((nx),dtype='float32')
        #goal_weight_np[0] = 1.0*math.pow(0.5, j)

        for i in range(len(v_weight)):       
            goal_weight_np[v_index[i]] = v_weight[i] 


        # swingup goal (observe theta and theta_dt)
        goal_weights = torch.from_numpy(goal_weight_np) # nx
        #goal_state = torch.tensor((np.pi, 1.))
        #ctrl_penalty = 1.0e-30*math.pow(0.8, j)

        ctrl_penalty = 1
        q = torch.cat((
            goal_weights,
            #ctrl_penalty * torch.ones(nu,dtype=torch.float32)
            torch.tensor( [1.0e-30, 1.0e-30], dtype=torch.float32)
        ))  # nx + nu                    
        Q_tmp = torch.diag(q).repeat(N_BATCH, 1, 1)  # T x B x nx+nu x nx+nu
        
        Q_vec.append(Q_tmp)
        
        
        
        goal_state_np = np.zeros((nx),dtype='float32')
        for i in range(len(v_weight)):
            target = np.array([v_times[i],v_values[i]], dtype='float32').T
            goal_tmp = get_Target(timeT+j, target)
            
            goal_state_np[v_index[i]] = goal_tmp


        goal_state = torch.from_numpy(goal_state_np) # nx
        px = -torch.sqrt(goal_weights) * goal_state
        
        p_tmp = torch.cat((px, torch.from_numpy(np.array([0, 0], dtype='float32' ) )     ))
        p_tmp = p_tmp.repeat( N_BATCH, 1)

        p_vec.append(p_tmp)
    

    Q = torch.cat(Q_vec, dim=0).reshape(TIMESTEPS,N_BATCH, nx+nu, nx+nu)
    p = torch.cat(p_vec, dim=0).reshape(TIMESTEPS,N_BATCH, nx+nu)
    #print(Q.shape,p.shape )
    
    cost = mpc.QuadCost(Q, p)  # T x B x nx+nu (linear component of cost)        

    return cost





def rescale_to_normal(state, outmax, outmin):
    for i in range(len(outmax)):
        state[:,i] = state[:,i]*(outmax[i]-outmin[i]) +outmin[i]    
    return state    

# 




def generate_measurement(v_reactivity, v_inlet, e_s, e_sigma, param=None):
    
    core_real = pcc.PointCore_critical(sigma=0,param=param )

    current_state = core_real.define_balance_status(1.0)
    
    v_reactivity.insert(0, [0, 0])
    v_inlet.insert(0, [0, 290])
    v_reactivity = np.array(v_reactivity)
    v_inlet = np.array(v_inlet)
    import random
    mes = []
    real = []
    status = []
    action = []
    last_status = []
    np.random.seed(100)


    for ic in range(int(v_reactivity[-1][0])):
        
        reactivity =  get_Target(ic+1, v_reactivity)
        inlet =  get_Target(ic+1, v_inlet)
        act_tmp = [reactivity, inlet]
        
        state_n = core_real.predict_state_action( 0.1, current_state , act_tmp )
        
        
        obz = state_n[0]+ e_s + random.gauss(0,1)*e_sigma
        
        last_status.append(current_state)
        action.append(act_tmp )
        mes.append(obz)
        real.append(state_n[0])
        status.append(state_n)
    
        current_state = state_n
        
        
    return mes, real, status, action, last_status
        









def initial_mixmodel_data(model_mix,  core_real):
    import streamlit as st
    if model_mix.n_model <= 1:
        return
    


    for irep in range(3):
        
        
        nr_init=(0.9*random.random()+0.1  )
        state_begin = core_real.define_balance_status(nr_init)
        for ic in range(10):    
    
            rd1 = random.random()-0.5
            rd2 = random.random()-0.5
            action = [0.1*rd1, 290] #
                
            new_state = core_real.predict_state_action(0.1,state_begin, action) 
                
            while new_state[0] > 3. or new_state[0] < 0.01:
                rd1 = random.random()-0.5
                rd2 = random.random()-0.5
                action = [0.1*rd1, 290] #
                new_state = core_real.predict_state_action(0.1,state_begin, action)                           
            
            act_tmp = np.array([action], dtype='float32' ) 
            
            #st.write(new_state)
            X_one = []
            for i in range(model_mix.n_model): 
                new_state_model = model_mix.model_vector[i].predict_step_no_sampling( 
                        torch.tensor(state_begin, dtype=torch.float32), 
                        torch.from_numpy(act_tmp) )
                
                new_state_model_np = new_state_model.detach().numpy()
                new_state_model_np=np.squeeze(new_state_model_np)
                X_one.append( new_state_model_np.tolist() )
            
    
            model_mix.update_data(X_one, list(new_state) ) 
            state_begin = new_state

              
            



    return      

def ppppppppp_one(status_list_cv,goal_list_cv, control_list_cv,mix_weight_list_cv,
                 u_lower_uper,v_labels,v_index,model_name_dict, empty_status , empty_act ):

    
    target_rms = []
    x = list(range(len(status_list_cv[0])))
    

    if len(list(model_name_dict.keys()))>1:
           label_model = ["ensembled"]
           label_model.extend(list(model_name_dict.keys() ))
    else:
        label_model = list(model_name_dict.keys() )
         
    
    for i in range(13):
        x_list = []
        y_list = []
        z_list = []
        
        if i in v_index:
            x_list.append(x)
            y_list.append(goal_list_cv[i] )
            z_list.append("target")

        x_list.append(x)
        y_list.append(status_list_cv[i])
            
        if i in v_index:
            rrms = np.sqrt( np.mean(np.power(np.array(status_list_cv[i])/np.array(goal_list_cv[i])-1., 2)  ))
            z_list.append( " RRMS:{:7.2e}".format(rrms))
            
        else:
            z_list.append("model")
 
        if i==0:
            xy_plot(x_list, y_list, z_list, "step", v_labels[i],
                       [], [], [], [], [], [], empty=empty_status[i] )            
            
    rrms = np.sqrt(np.mean(np.array(target_rms)) )    
        
       
        
        
    label_control = ["reactivity","inlet temperture"]
    for i in range(2):
        x_list = []
        y_list = []
        z_list = []
        

        x_list.append(x)
        y_list.append(control_list_cv[i])
        z_list.append("model")
        
        x_list.extend([x,x])
        y_list.extend( [[u_lower_uper[i][0]]*len(x), [u_lower_uper[i][1]]*len(x)])
        z_list.extend(["down boundary", "up boundary"] )
        
        
        #xy_plot(x_list, y_list, z_list, "step", label_control[i],
        #           [], [], [], [], [], [], empty=empty_act[i] )                      
    
    
  
    #xy_stackplot(x, mix_weight_list_cv[0], list(model_name_dict.keys() ) )  
    return


def ppppppppp_v(status_list_cv,goal_list_cv, control_list_cv,mix_weight_list_cv,
                 u_lower_uper,v_labels,v_index,model_name_dict):

    
    target_rms = []
    x = list(range(len(status_list_cv[0][0])))
    
    
    
    if len(list(model_name_dict.keys()))>1:
           label_model = ["ensembled"]
           label_model.extend(list(model_name_dict.keys() ))
    else:
        label_model = list(model_name_dict.keys() )
        
    xdim = min(len(status_list_cv),len(label_model))    
    
    for i in range(13):
        x_list = []
        y_list = []
        z_list = []
        
        if i in v_index:
            x_list.append(x)
            y_list.append(goal_list_cv[0][i] )
            z_list.append("target")
            
        
        for ic in range(xdim):
            x_list.append(x)
            y_list.append(status_list_cv[ic][i])
            
            if i in v_index:
                rrms = np.sqrt( np.mean(np.power(np.array(status_list_cv[ic][i])/np.array(goal_list_cv[0][i])-1., 2)  ))
                z_list.append(label_model[ic]+ " RRMS:{:7.2e}".format(rrms))
                
            else:
                z_list.append(label_model[ic])
 

        xy_plot(x_list, y_list, z_list, "step", v_labels[i],
                   [], [], [], [], [], [])
    
        
    rrms = np.sqrt(np.mean(np.array(target_rms)) )    
        
       
        
        
    label_control = ["reactivity","inlet temperture"]
    for i in range(2):
        x_list = []
        y_list = []
        z_list = []
        
        for ic in range(len(status_list_cv)):
            x_list.append(x)
            y_list.append(control_list_cv[ic][i])
            z_list.append(label_model[ic])
        
        x_list.extend([x,x])
        y_list.extend( [[u_lower_uper[i][0]]*len(x), [u_lower_uper[i][1]]*len(x)])
        z_list.extend(["down boundary", "up boundary"] )
        
        xy_plot(x_list, y_list, z_list, "step", label_control[i],
                   [], [], [], [], [], [])     
                   
    
    
    #xy_stackplot(x, mix_weight_list_cv[0], list(model_name_dict.keys() ) )    

    return




def fiter_xy_plot(status_prd, status_est, status_est_P,measure,v_labels, model_name_dict):
    x = list(range(len(measure)))
    
    if len(list(model_name_dict.keys()))>1:
           label_model = ["ensembled"]
           label_model.extend(list(model_name_dict.keys() ))
    else:
        label_model = list(model_name_dict.keys() )
        
    xdim = min(len(status_prd),len(label_model))  
        
    for i in range(13):
        x_list = []
        y_list = []
        z_list = []    
    
        xs_list = []
        ys_list = []
        zs_list = []       
        
 
            
        for ic in range(xdim):    
            xs_list.append(x)
            ys_list.append(status_prd[ic][i] )
            zs_list.append(label_model[ic]+"_predicted")         

        for ic in range(xdim):    
            x_list.append(x)
            y_list.append(status_est[ic][i] )
            z_list.append(label_model[ic]+"_estimated")   
            
        if i==0:
            xs_list.append(x)
            ys_list.append(measure )
            zs_list.append("measure")             
            
        
        x_updw = x
        y_up = list(np.array(status_est[0][i])+3.*np.sqrt(np.array(status_est_P[0][i])) )
        y_dw = list(np.array(status_est[0][i])-3.*np.sqrt(np.array(status_est_P[0][i])) )
        


        xy_plot(x_list, y_list, z_list,
                    "step", v_labels[i],
                    xs_list, ys_list, zs_list,
                    x_updw, y_up, y_dw )
 
                    


def fiter_xy_plot_one(status_prd, status_est, status_est_P, measure,v_labels, model_name_dict, empty_status,realtmp):
    
    x = list(range(len(measure)))
    
    x_r = list(range(len(status_prd[0])))
    
    if len(list(model_name_dict.keys()))>1:
           label_model = ["ensembled"]
           label_model.extend(list(model_name_dict.keys() ))
    else:
        label_model = list(model_name_dict.keys() )
        
    
        
    for i in range(13):
        x_list = []
        y_list = []
        z_list = []    
    
        xs_list = []
        ys_list = []
        zs_list = []       
        
          
            

        xs_list.append(x_r)
        ys_list.append(status_prd[i] )
        zs_list.append("predicted")         

  
        x_list.append(x_r)
        y_list.append(status_est[i] )
        z_list.append("estimated")   
        
        if i==0:
            xs_list.append(x)
            ys_list.append(measure )
            zs_list.append("measured")  
            
            x_list.append(x)
            y_list.append(realtmp )
            z_list.append("real")           
        
        
        
        x_updw = x_r
        y_up = list(np.array(status_est[i])+2.*np.sqrt(np.array(status_est_P[i])) )
        y_dw = list(np.array(status_est[i])-2.*np.sqrt(np.array(status_est_P[i])) )
        


        xy_plot(x_list, y_list, z_list,
                    "step", v_labels[i],
                    xs_list, ys_list, zs_list,
                    x_updw, y_up, y_dw, empty = empty_status[i] )        
       



def mpc_with_filter_xy_plot(u_lower_uper, v_labels, v_index, 
                            status_list_cv, goal_list_cv, control_list_cv, model_name_dict,
                            mix_weight_list_cv, measure_cv, status_prd_cv, status_est_cv, status_est_P_cv, 
                            empty_status = None, empty_act = None):


    x = list(np.array(range(len(measure_cv[0])))+1)
    x1 = list(range(len(measure_cv[0])+1))


    
    #xy_stackplot(x1, mix_weight_list_cv, list(model_name_dict.keys() ) )    

    

    for i in range(13):
        x_list = []
        y_list = []
        z_list = []    
    
        xs_list = []
        ys_list = []
        zs_list = []       
        
        if i==0:
            xs_list.append(x)
            ys_list.append(measure_cv[0] )
            zs_list.append("measure")  
            
        if i in v_index:
            x_list.append(x1)
            y_list.append(goal_list_cv[i] )
            z_list.append("target")

            
            
        xs_list.append(x)
        ys_list.append(status_prd_cv[i] )
        if len(list(model_name_dict.keys() )) >1 :
            zs_list.append("ensemble predict")
        else:
            zs_list.append(list(model_name_dict.keys() )[0]+" predict")

        #x_list.append(x1)
        #y_list.append(status_list_cv[i])
        #z_list.append("estmated_1")
        

        x_list.append(x)
        y_list.append(status_est_cv[i] )
        if i in v_index:
            rrms = np.sqrt( np.mean(np.power(np.array(status_est_cv[i])/np.array(goal_list_cv[i][1:])-1., 2)  ))
            z_list.append("estimated"+ " RRMS:{:7.2e}".format(rrms))
        else:
            z_list.append("estimated")   
        
        
        
        x_updw = x
        y_up = list(np.array(status_est_cv[i])+3.*np.sqrt(np.array(status_est_P_cv[i])) )
        y_dw = list(np.array(status_est_cv[i])-3.*np.sqrt(np.array(status_est_P_cv[i])) )
        
        
        if empty_status is None:
            xy_plot(x_list, y_list, z_list,
                        "step", v_labels[i],
                        xs_list, ys_list, zs_list,
                        x_updw, y_up, y_dw )
        else:
            xy_plot(x_list, y_list, z_list,
                        "step", v_labels[i],
                        xs_list, ys_list, zs_list,
                        x_updw, y_up, y_dw, empty=empty_status[i]  )            



    label_control = ["reactivity","inlet temperture"]
    for i in range(2):
        x_list = []
        y_list = []
        z_list = []
        

        x_list.append(x1)
        y_list.append(control_list_cv[i])
        z_list.append("")
        
        x_list.extend([x,x])
        y_list.extend( [[u_lower_uper[i][0]]*len(x), [u_lower_uper[i][1]]*len(x)])
        z_list.extend(["down boundary", "up boundary"] )
        

        if empty_act is None:
            xy_plot(x_list, y_list, z_list, "step", label_control[i],
                       [], [], [], [], [], [])      
        else:
            xy_plot(x_list, y_list, z_list,
                        "step", v_labels[i],
                        xs_list, ys_list, zs_list,
                        x_updw, y_up, y_dw, empty=empty_act[i]  )              
    


def run_MPC_nofliter(  
                model_name_dict,   
                v_index, v_weight, v_times, v_values,v_labels,
                u_lower_uper,
                learner, online_plot,
            param = None, 
            
            TIMESTEPS = 3,# T 
            N_BATCH = 1 ,# 
            LQR_ITER = 20 ,# 
        
            nx = 13, #
            nu = 2 ,#
            dt = 0.1, #               
            grad_method = "FINITE" ,dataset = ""   ):
    
    import numpy as np
    import streamlit as st
  

    act = np.array([[0, 290]], dtype='float32' )
    u_init_0 = torch.from_numpy(act) #
    u_init_1 = torch.from_numpy(act) #

    core_real = pcc.PointCore_critical(sigma=0,param= param)
    current_state = core_real.define_balance_status(1.0)


                                      

    u_lower = -torch.rand(TIMESTEPS, 1, nu)
    u_upper = -torch.rand(TIMESTEPS, 1, nu)
    for iu in range(TIMESTEPS):
        u_lower[iu, 0, 0] = u_lower_uper[0][0]
        u_upper[iu, 0, 0] = u_lower_uper[0][1]

        u_lower[iu, 0, 1] = u_lower_uper[1][0]
        u_upper[iu, 0, 1] = u_lower_uper[1][1]
    


    
    model_dict = cdm.get_method_list( model_name_dict, dataset=dataset )
    model_list = list(model_dict.values() )
    model_mix = cdm.MixNNReactorModel(  model_list, dataset=dataset )
    

    if model_mix.n_model > 1:
        with st.spinner("weight evaluate between different models..."):
            initial_mixmodel_data(model_mix,core_real)
           
            model_mix.update_second_learner(learner)  

    Time = 0
    for i in range(len(v_times)):
        if max(v_times[i]) > Time:
            Time = max(v_times[i])
    
    #Time = 5
    status_list = [current_state]
    control_list = [  [0, 290] ]
                    
    mix_weight_list = [model_mix.weight]
    goal_list = [current_state]
    
    
    
    st.write(model_name_dict.keys())
    
    
    myempt = st.progress(0)
    
    
    empty_status = []
    for i in range(13):
        empty_status.append(st.empty())
        
    empty_act = []
    for i in range(2):
        empty_act.append(st.empty())        
    
    for i in range(Time):

        myempt.progress(int(((i+1)/Time)*100))
        goal_save = [0]*13
        for k in range(len(v_weight)):
            target = np.array([v_times[k],v_values[k]], dtype='float32').T
            goal_tmp = get_Target(i+1, target)

            goal_save[v_index[k]] = goal_tmp


        cost = cost_function_update(TIMESTEPS, N_BATCH, nx, nu, i, v_weight, v_index, v_times, v_values  )

        ctrl_pert = mpc.MPC(nx, nu, TIMESTEPS, u_lower=u_lower,
                            u_upper=u_upper, lqr_iter=LQR_ITER,
                       exit_unconverged=False, 
                       n_batch=N_BATCH, backprop=False, verbose=0,
                       u_init=u_init_0,
                       grad_method=mpc.GradMethods.FINITE_DIFF)   #FINITE_DIFF #AUTO_DIFF #ANALYTIC      
        
        current_state_torch = torch.tensor(current_state, dtype=torch.float32).view(1, -1) 
        nominal_states, nominal_actions, nominal_objs = ctrl_pert(current_state_torch, cost, model_mix )  
        
        
        
        if mpc.WARNING_INDEX >= 3:
            print("WARNING_INDEX", mpc.WARNING_INDEX)
            mpc.WARNING_INDEX = 0  
            
        else:
            action = nominal_actions[0] 
            u_init_0 = torch.cat(( nominal_actions[1:], u_init_1.view(1, 1, nu) ), dim=0)

    
        # step2 
        new_state = core_real.predict_state_action(dt, current_state, action[0] ) 

        state_new_np=np.squeeze(new_state)
        core_new_state = state_new_np.tolist()

         
        X_one = []
        for k in range(model_mix.n_model):
            new_state1 = model_mix.model_vector[k].predict_step_no_sampling(  
                        torch.tensor(current_state, dtype=torch.float32), 
                        action[0] )
            
            
            new_state_np = new_state1.detach().numpy()
            state_new_np=np.squeeze(new_state_np)
            X_one.append( state_new_np.tolist() )
            
         
        model_mix.update_data(X_one,core_new_state )       
        model_mix.update_second_learner(learner) 
        
        status_list.append(core_new_state)
        control_list.append([float(action[0][0]),float(action[0][1])] )   
        mix_weight_list.append(model_mix.weight)
        goal_list.append(list(goal_save))
        

        current_state = new_state
    
        status_list_cv_f = [ [status_list[j][i] for j in range(len(status_list))] for i in range(len(status_list[0])) ]     
        control_list_cv_f = [ [control_list[j][i] for j in range(len(control_list))] for i in range(len(control_list[0])) ]
        mix_weight_list_cv_f =  [ [mix_weight_list[j][i] for j in range(len(mix_weight_list))] for i in range(len(mix_weight_list[0])) ]
        goal_list_cv_f =  [ [goal_list[j][i] for j in range(len(goal_list))] for i in range(len(goal_list[0])) ]
            
        if online_plot== "Yes":
            ppppppppp_one(status_list_cv_f,goal_list_cv_f, control_list_cv_f,mix_weight_list_cv_f,
                 u_lower_uper,v_labels,v_index,model_name_dict,empty_status = empty_status, empty_act = empty_act)

    
    status_list_cv = [ [status_list[j][i] for j in range(len(status_list))] for i in range(len(status_list[0])) ]     
    control_list_cv = [ [control_list[j][i] for j in range(len(control_list))] for i in range(len(control_list[0])) ]
    mix_weight_list_cv =  [ [mix_weight_list[j][i] for j in range(len(mix_weight_list))] for i in range(len(mix_weight_list[0])) ]
    goal_list_cv =  [ [goal_list[j][i] for j in range(len(goal_list))] for i in range(len(goal_list[0])) ]
        


    return  status_list_cv,goal_list_cv, control_list_cv,mix_weight_list_cv

class Model_observation(torch.nn.Module):
    def __init__(self ):   
        super( Model_observation, self ).__init__()   
        pass
    
    def forward(self, state):   
        return state[0]

def run_EnKF(model_name_dict, learner,variables,
            measure, status, 
            e_s,e_sigma,
            action, last_status,partical_number,system_dev,real,
            param=None,
            nomralized_status= "no", dataset = ""):
    
    import numpy as np
    import streamlit as st


    core_real = pcc.PointCore_critical(sigma=0,param=param)

    current_state = core_real.define_balance_status(1.0)   
    #st.write(current_state)


    model_dict = cdm.get_method_list( model_name_dict ,dataset=dataset)
    model_list = list(model_dict.values() )
    model_mix = cdm.MixNNReactorModel(  model_list ,dataset=dataset)
    model_list = list(model_dict.values() )
    

    initial_mixmodel_data(model_mix,core_real)
    

    model_mix.update_second_learner(learner)  
     

    model_obz = Model_observation()    
    
    
    enkf = dam.EnsembleKalmanFilter(  
            N = partical_number,               
            model_prediction = model_mix, 
            model_observation = model_obz)   
    
    if nomralized_status=="yes":
        x = np.array(  [1.0]*13  )
        P = np.eye(13)*x*x*0.001
        enkf.initialize_x_P( x, P )
        enkf.set_x_reference(core_real.define_balance_status(1.0) )
        enkf.generate_model_prediction_err(core_real, system_dev,actions=action) 
    

    my_bar = st.progress(0)

    T = len(measure)
    
    status_prd = []
    status_est = []
    status_est_P = []
    
    empty_status = []
    for i in range(13):
        empty_status.append(st.empty())    
    
    mestemp = []
    realtmp = []
    for i in range(T):


        act = action[i]
        
        u_a = torch.from_numpy(np.array([act], dtype='float32' ) ) 


        x_prd,P_prd = enkf.predict(u_a[0]) 
        

        R=np.array([[e_sigma**2]])
        z = [measure[i]]
        x,P = enkf.update(z,[e_s], R=R) 
        
        my_bar.progress(int((i + 1)/T*100))
        
        mestemp.append(measure[i])
        realtmp.append(real[i])
        status_prd.append([x_prd[i]*enkf.x_reference[i] for i in range(13)])
        status_est.append([x[i]*enkf.x_reference[i] for i in range(13)])
        status_est_P.append([P[i,i]*enkf.x_reference[i]*enkf.x_reference[i] for i in range(13)])
        
        status_prd_cv = [ [status_prd[j][i] for j in range(len(status_prd))] for i in range(len(status_prd[0])) ]     
        status_est_cv = [ [status_est[j][i] for j in range(len(status_est))] for i in range(len(status_est[0])) ]
        status_est_P_cv =  [ [status_est_P[j][i] for j in range(len(status_est_P))] for i in range(len(status_est_P[0])) ]
            
        
        fiter_xy_plot_one(status_prd_cv, status_est_cv, status_est_P_cv, mestemp,variables, model_name_dict,empty_status,realtmp)
  
        
        
    status_prd_cv = [ [status_prd[j][i] for j in range(len(status_prd))] for i in range(len(status_prd[0])) ]     
    status_est_cv = [ [status_est[j][i] for j in range(len(status_est))] for i in range(len(status_est[0])) ]
    status_est_P_cv =  [ [status_est_P[j][i] for j in range(len(status_est_P))] for i in range(len(status_est_P[0])) ]

                                     
    
    return status_prd_cv, status_est_cv, status_est_P_cv





def run_MPC(  
                model_name_dict,   
                v_index, v_weight, v_times, v_values,v_labels,
                u_lower_uper,
                learner, partical_number,nomralized_status,e_s,e_sigma,system_dev,
            param = None, 
            
            TIMESTEPS = 3,# T 
            N_BATCH = 1 ,# 
            LQR_ITER = 20 ,# 
        
            nx = 13, #
            nu = 2 ,#
            dt = 0.1, #            
            grad_method = "FINITE" ,dataset = ""   ):
    
    import numpy as np
    import streamlit as st
  

    act = np.array([[0, 290]], dtype='float32' )
    u_init_0 = torch.from_numpy(act) #
    u_init_1 = torch.from_numpy(act) #

    core_real = pcc.PointCore_critical(sigma=0,param= param)
    current_state = core_real.define_balance_status(1.0)

    u_lower = -torch.rand(TIMESTEPS, 1, nu)
    u_upper = -torch.rand(TIMESTEPS, 1, nu)
    for iu in range(TIMESTEPS):
        u_lower[iu, 0, 0] = u_lower_uper[0][0]
        u_upper[iu, 0, 0] = u_lower_uper[0][1]

        u_lower[iu, 0, 1] = u_lower_uper[1][0]
        u_upper[iu, 0, 1] = u_lower_uper[1][1]
    


   
    st.info("weight between different models")
    
    model_dict = cdm.get_method_list( model_name_dict ,dataset =dataset)
    model_list = list(model_dict.values() )
    model_mix = cdm.MixNNReactorModel(  model_list,dataset =dataset )

    initial_mixmodel_data(model_mix,core_real)
    model_mix.update_second_learner(learner)  
    

    model_obz = Model_observation()    
    
    
    enkf = dam.EnsembleKalmanFilter(  
            N = partical_number,               
            model_prediction = model_mix, 
            model_observation = model_obz)   
    
    if nomralized_status=="yes":
        x = np.array(  [1.0]*13  )
        P = np.eye(13)*x*x*0.001
        enkf.initialize_x_P( x, P )
        enkf.set_x_reference(core_real.define_balance_status(1.0) )
        enkf.generate_model_prediction_err(core_real,system_dev) 
     




    Time = 0
    for i in range(len(v_times)):
        if max(v_times[i]) > Time:
            Time = max(v_times[i])
    
    #Time = 5

    myempt1 = st.progress(0)                    
    empty_status = []
    for i in range(13):
        empty_status.append(st.empty())
        

    empty_act = []
    for i in range(2):
        empty_act.append(st.empty())    
        
    
    for i in range(100):

        new_state = core_real.predict_state_action(dt, current_state, u_init_0[0] ) 



        R=np.array([[e_sigma**2]])
        z = np.array( [new_state[0]+e_s + random.gauss(0,e_sigma ) ]  )
        

        x_prd,P_prd = enkf.predict( u_init_0[0])

        x,P = enkf.update(z,[e_s], R=R)               
    
        current_state= new_state
    
        myempt1.progress(int(((i+1)/100)*100))
    
    
    
    mix_weight_list = [model_mix.weight]
    goal_list = [current_state]
    
    measure = []
    status_prd = []
    status_est = []
    status_est_P = []    
    
    status_list = [current_state]
    control_list = [  [0, 290] ]
    myempt = st.progress(0)
    for i in range(Time):

        myempt.progress(int(((i+1)/Time)*100))
        goal_save = [0]*13
        for k in range(len(v_weight)):
            target = np.array([v_times[k],v_values[k]], dtype='float32').T
            goal_tmp = get_Target(i+1, target)

            goal_save[v_index[k]] = goal_tmp


        cost = cost_function_update(TIMESTEPS, N_BATCH, nx, nu, i, v_weight, v_index, v_times, v_values  )

        ctrl_pert = mpc.MPC(nx, nu, TIMESTEPS, u_lower=u_lower,
                            u_upper=u_upper, lqr_iter=LQR_ITER,
                       exit_unconverged=False, 
                       n_batch=N_BATCH, backprop=False, verbose=0,
                       u_init=u_init_0,
                       grad_method=mpc.GradMethods.FINITE_DIFF)   #FINITE_DIFF #AUTO_DIFF #ANALYTIC      
        
        current_state_torch = torch.tensor(current_state, dtype=torch.float32).view(1, -1) #
        nominal_states, nominal_actions, nominal_objs = ctrl_pert(current_state_torch, cost, model_mix )  
        




        
        if mpc.WARNING_INDEX >= 3:
            print("WARNING_INDEX", mpc.WARNING_INDEX)
            mpc.WARNING_INDEX = 0  
            
        else:
            action = nominal_actions[0] 
            u_init_0 = torch.cat(( nominal_actions[1:], u_init_1.view(1, 1, nu) ), dim=0)

    
        # step2 
        new_state = core_real.predict_state_action(dt, current_state, action[0] ) 


        # step3, 
        R=np.array([[e_sigma**2]])
        z = np.array( [new_state[0]+e_s + random.gauss(0,e_sigma ) ]  )
        

        x_prd,P_prd = enkf.predict( action[0])
        x,P = enkf.update(z,[e_s], R=R)       
       
        measure.append(list(z))
        status_prd.append([x_prd[i]*enkf.x_reference[i] for i in range(13)])
        status_est.append([x[i]*enkf.x_reference[i] for i in range(13)])
        status_est_P.append([P[i,i]*enkf.x_reference[i] for i in range(13)])

        new_state = x*np.array(enkf.x_reference)

        state_new_np=np.squeeze(new_state)
        core_new_state = state_new_np.tolist()

         
        X_one = []
        for k in range(model_mix.n_model):
            new_state1 = model_mix.model_vector[k].predict_step_no_sampling(  
                        torch.tensor(current_state, dtype=torch.float32), 
                        action[0] )
            
            
            new_state_np = new_state1.detach().numpy()
            state_new_np=np.squeeze(new_state_np)
            X_one.append( state_new_np.tolist() )
            
         
        model_mix.update_data(X_one,core_new_state )       
        model_mix.update_second_learner(learner) 
        
        
        
        status_list.append(core_new_state)
        control_list.append([float(action[0][0]),float(action[0][1])] )   
        mix_weight_list.append(model_mix.weight)
        goal_list.append(list(goal_save))
        
        current_state= new_state
        

    
        status_list_cv = [ [status_list[j][i] for j in range(len(status_list))] for i in range(len(status_list[0])) ]     
        control_list_cv = [ [control_list[j][i] for j in range(len(control_list))] for i in range(len(control_list[0])) ]
        mix_weight_list_cv =  [ [mix_weight_list[j][i] for j in range(len(mix_weight_list))] for i in range(len(mix_weight_list[0])) ]
        goal_list_cv =  [ [goal_list[j][i] for j in range(len(goal_list))] for i in range(len(goal_list[0])) ]
            
        status_prd_cv = [ [status_prd[j][i] for j in range(len(status_prd))] for i in range(len(status_prd[0])) ]     
        status_est_cv = [ [status_est[j][i] for j in range(len(status_est))] for i in range(len(status_est[0])) ]
        status_est_P_cv =  [ [status_est_P[j][i] for j in range(len(status_est_P))] for i in range(len(status_est_P[0])) ]
    
        measure_cv = [ [measure[j][i] for j in range(len(measure))] for i in range(len(measure[0])) ] 
                    



        mpc_with_filter_xy_plot(u_lower_uper,v_labels,v_index, 
                                    status_list_cv,goal_list_cv, control_list_cv,model_name_dict,
                            mix_weight_list_cv,measure_cv, status_prd_cv,status_est_cv,status_est_P_cv,empty_status = empty_status, empty_act = empty_act)   




   

    return  status_list_cv,goal_list_cv, control_list_cv,mix_weight_list_cv,measure_cv,status_prd_cv,status_est_cv,status_est_P_cv














    












