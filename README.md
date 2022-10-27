start command:

streamlit run main.py
 
open a website, which would following below steps:

1.	Given: a mechanism core model, setup parameter values
2.	Add noise to real core model to construct parameter perturbation model  
3.	Generated training data based on given real core simulator 
4.	Offline training different types of first layer (meta learning) model, include BiLSTM,BiGRU,LSTM,GRU,MLP,SVR,RFR .... Hyper-parameters of each learning model are optimized. 
5.	Offline training the second layer (e.g. linear regression) to get weight coefficient and ensemble learning model
6.	Online core monitoring and controlling, repeat for time-step t:
  6.1	update cost function base on target curve and control constraint for t time step.
  6.2	DPP based on the model  local linearization, to get optimal sequence control action. Including following steps:
    For each k-th iteration in LQR_number:
    (a)	Get current trajectory   
    (b)	Linearized dynamics model around the current trajectory, to get first-order and second-order linear function
    (c)	Linearized cost function around the current trajectory to get first-order and second-order linear function
    (d)	Solving the iLQR sub-problem, to get k+1 iteration sub-optimal action
    (e)	Set k=k+1, go to (a), until LQR_number is reach. Final get optimal action  

  6.3	Execute first step action in optimal sequence action , or not
  6.4	Predict core state based on ensemble model, control action, and current estimated status and relative uncertainty, to get next time-step predictive core state  and prior probability  
  6.5	receive real core measurable parameter and the relative uncertainty 
  6.6	Update core state based on measurement, to get next time-step best estimated state and the posterior probability  
  6.7	Predict t+1 time-step core status  based on first layer learner 
  6.8	online training the second layer to update weight coefficient , based on predicted first layer predicted status and the best estimated status,update ensemble learning model  
  6.9	go to 6.1 for next time step






