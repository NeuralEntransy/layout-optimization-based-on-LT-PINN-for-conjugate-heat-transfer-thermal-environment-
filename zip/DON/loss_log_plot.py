import matplotlib.pyplot as plt
import numpy as np
# load the train loss data
loss =  np.loadtxt('loss_log.csv', delimiter=',')

## extract the steps and error from the loss data
## flag -1 L_t; flag -2 Lu_t; flag -3 Lv_t; flag -4 Lw_t
## flag -5 train_u_error; flag -6 train_v_error; flag -7 train_w_error
Steps = loss[:, 0]  
error = loss[:, 1]

## plot the train loss
plt.figure(figsize=(8,5))
plt.plot(Steps, error,c = 'r') # 
plt.xlabel('Steps') # 
plt.ylabel('train loss') # 
plt.show() # 