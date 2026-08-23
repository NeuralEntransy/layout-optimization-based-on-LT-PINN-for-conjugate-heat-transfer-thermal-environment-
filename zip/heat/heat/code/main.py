# import tools
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import csv
import math
import matplotlib.pyplot as plt
import time
import loss
import NeuralNetwork


## set global parameters
device = torch.device("cuda:0")
torch.manual_seed(20231028)
torch.set_default_dtype(torch.float32)


def main():

    # read data
    def load_data(filename):
        data_pd = pd.read_csv(filename, encoding='gbk',header=None)
        data = np.array(data_pd)
        return data[1:,:].copy().astype(float)
    
    data = load_data('../../../data/heat_128x128.csv')

    Lx_max = 0
    Lx_min = 0
    Ly_max = 0
    Ly_min = 0
    

    data_xy = data[:,0:2]
    data_xy[:,0] = data_xy[:,0]+7.5
    data_u = data[:,3:]-273 # normalize to 273K
    # x_temp = data_xy[:,0] # x coordinate
    # y_temp = data_xy[:,1] # y coordinate 
    idx = (x_temp<=1.1) & (x_temp>=-1.1) & (y_temp<=Ly_max+1.1) & (y_temp>=Ly_min-1.1)   
    x_all = data_xy[idx] # input
    y_all = data_u[idx] # output

    
    # test point
    x_temp = x_all[:,0]
    y_temp = x_all[:,1]
    # index of interior points
    idx = (x_temp<=1) & (x_temp>=-1) & (y_temp<=Ly_max+1) & (y_temp>=Ly_min-1)
    x_test = x_all[idx]
    y_test = y_all[idx]
    X_test = torch.tensor(x_test,dtype=torch.float32,device=device)
    Y_test = torch.tensor(y_test,dtype=torch.float32,device=device)
    
    
    # supervise point
    x_temp = x_all[:,0]
    y_temp = x_all[:,1]
    idx = (x_temp>1) | (x_temp<-1) | (y_temp>1) | (y_temp<-1)
    x_sup = x_all[idx]
    y_sup = y_all[idx]
    
#   # add some supervise data
    idx = np.random.choice(len(x_test),int(0.1*len(x_test)),replace=False)
    x_sup_ = x_test[idx]
    y_sup_ = y_test[idx]
    
#   X_sup = torch.tensor(np.concatenate((x_sup,x_sup_),axis=0),dtype=torch.float32,device=device)
#   Y_sup = torch.tensor(np.concatenate((y_sup,y_sup_),axis=0),dtype=torch.float32,device=device)
    X_sup = torch.tensor(x_sup_,dtype=torch.float32,device=device)
    Y_sup = torch.tensor(y_sup_,dtype=torch.float32,device=device)

    
    # internal collocation point
    num_points_unit = 60
    num_points_x = int((Lx_max-Lx_min+2)*num_points_unit)  
    num_points_y = int((Ly_max-Ly_min+2)*num_points_unit) 
    x = np.linspace(-1, 1, num_points_x)
    y = np.linspace(Ly_min-1, Ly_max+1, num_points_y)
    X, Y = np.meshgrid(x, y)
    x_internal = np.stack((X.flatten(), Y.flatten()), axis=1)
    X_internal = torch.tensor(x_internal,dtype=torch.float32,device=device)
    X_internal.requires_grad = True
    
    print('test:',X_test.shape,Y_test.shape,'sup:',X_sup.shape,Y_sup.shape,'Internal:',X_internal.shape)
    
    # boundary point
    # this may not the circle boundar, It may be the Boundary and interior points.
    def get_circle_points(center_x, center_y): 
        theta = torch.linspace(0, 2 * torch.pi, 256, requires_grad=False,device=device)
        all_points = []
        for dr in range(4):
            x = center_x + (0.5-0.1*dr)*torch.cos(theta)
            y = center_y + (0.5-0.1*dr)*torch.sin(theta)
            points = torch.stack((x, y), dim=1)  
            all_points.append(points)
        all_points = torch.cat(all_points, dim=0)
        return all_points

    
    
    # define trainable network 
    numCenter = 1
    nn_fun = NeuralNetwork.MyNet(2,1,256,4,numCenter)
    nn_fun = nn_fun.cuda()
    
    # load ckp
    try:
        load_param = torch.load('../checkpoint/nn_fun_params')
        nn_fun.load_state_dict(load_param)
    except:
        print('no saved network')
        
    train_parameters =  list(nn_fun.parameters())
        
    # define optimizer  
    optimizer = torch.optim.Adam(lr=1e-4,params=train_parameters)
    try:
        load_param = torch.load('../checkpoint/opt_params')
        optimizer.load_state_dict(load_param)
    except:
        print('no saved opt')

    # define loss
    dataLoss = loss.DataLoss(nn_fun)
    pdeLoss = loss.PDELoss(nn_fun,Lx_min,Lx_max,Ly_min,Ly_max)
    bcLoss = loss.BCLoss(nn_fun,Lx_min,Lx_max,Ly_min,Ly_max)
    
    # train 
    wData = 1e4
    wEq = 1.0
    wBc = 1.0
    wTp = 1.0
    loss_sup = 0.0
    loss_eq = 0.0
    loss_test = 0.0
    loss_bc = 0.0
    loss_tp = 0.0
    
    for epoch in range(1,400001):
        
        time0 = time.time()
        
        # calculate cylinder center
        
        X_center = (Lx_max-Lx_min+2)*torch.sigmoid(nn_fun.X_center.weight[0,0])+(Lx_min-1)
        Y_center = (Ly_max-Ly_min+2)*torch.sigmoid(nn_fun.Y_center.weight[0,0])+(Ly_min-1)
        
        # data loss
        loss_sup,_ = dataLoss(X_sup,Y_sup)
        loss_sup =  wData * loss_sup
        loss_sup.backward()
        
        # bc loss
        X_bc = get_circle_points(X_center,Y_center)
        loss_bc = wBc * bcLoss(X_bc)
        
        # equation loss            
        loss_eq = wEq * pdeLoss(X_internal)

        # topology loss
#        loss_tp = wTp * (Y_center)**2
        
        (loss_bc+loss_eq).backward()
        
        # test monitor
        with torch.no_grad():
            loss_test,monitor_test = dataLoss(X_test,Y_test)

        optimizer.step()        
        optimizer.zero_grad()
        loss_total = loss_sup + loss_eq + loss_bc

        time1 = time.time()
        

        
        if epoch % 100 == 0:
            print('epoch:',epoch,'loss:',float(loss_total),'loss_sup:',float(loss_sup),'loss_eq:',float(loss_eq),'loss_bc:',float(loss_bc),'loss_tp:',float(loss_tp),'loss_test:',float(loss_test),'monitor_test:',float(monitor_test),'center',float(X_center),float(Y_center),time1-time0)
'''        
        if epoch % 1000 == 0:
            loss_log = [epoch,float(loss_total),float(loss_sup),float(loss_eq),float(loss_bc),float(loss_tp),float(loss_test),float(monitor_test),float(X_center),float(Y_center)]
            with open('../checkpoint/loss_log.csv', 'a') as file:
                writer = csv.writer(file)
                writer.writerow(loss_log)

        if epoch % 1000 == 0:
            torch.save(nn_fun.state_dict(),'../checkpoint/nn_fun_params')
            torch.save(optimizer.state_dict(),'../checkpoint/opt_params')

        if epoch % 100000 == 0:
            torch.save(nn_fun.state_dict(),'../checkpoint/nn_fun_'+str(epoch))
            torch.save(optimizer.state_dict(),'../checkpoint/opt_params_'+str(epoch))
'''
      
 
         
if __name__ == '__main__':
    
    main()

