--# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 22:29:38 2025

@author: Administrator
"""

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
from scipy.interpolate import griddata

## set global parameters
device = torch.device("cuda:0")
torch.manual_seed(20231028)
torch.set_default_dtype(torch.float32)

# read data
def load_data(filename):
    data_pd = pd.read_csv(filename, encoding='gbk',header=None)
    data = np.array(data_pd)
    return data[1:,:].copy().astype(float)

data = load_data('../../../data/electromag.csv')

Lx_max = -7.5
Lx_min = -7.5
Ly_max = 0
Ly_min = 0

data_xy = data[:,0:2]
data_uvb = data[:,3:6]
x_temp = data_xy[:,0]
y_temp = data_xy[:,1]   
idx = (x_temp<=Lx_max+1.1) & (x_temp>=Lx_min-1.1) & (y_temp<=Ly_max+1.1) & (y_temp>=Ly_min-1.1)   
x_all = data_xy[idx]
y_all = data_uvb[idx]


# test point
x_temp = x_all[:,0]
y_temp = x_all[:,1]
idx = (x_temp<=Lx_max+1) & (x_temp>=Lx_min-1) & (y_temp<=Ly_max+1) & (y_temp>=Ly_min-1)
x_test = x_all[idx]
y_test = y_all[idx]
X_test = torch.tensor(x_test,dtype=torch.float32,device=device)
Y_test = torch.tensor(y_test,dtype=torch.float32,device=device)


# supervise point
x_temp = x_all[:,0]
y_temp = x_all[:,1]
idx = (x_temp>Lx_max+1) | (x_temp<Lx_min-1) | (y_temp>Ly_max+1) | (y_temp<Ly_min-1)
x_sup = x_all[idx]
y_sup = y_all[idx]
X_sup = torch.tensor(x_sup,dtype=torch.float32,device=device)
Y_sup = torch.tensor(y_sup,dtype=torch.float32,device=device)
X_sup.requires_grad = True


# internal collocation point
num_points_unit = 60
num_points_x = int((Lx_max-Lx_min+2)*num_points_unit)  
num_points_y = int((Ly_max-Ly_min+2)*num_points_unit) 
x = np.linspace(Lx_min-1, Lx_max+1, num_points_x)
y = np.linspace(Ly_min-1, Ly_max+1, num_points_y)
X, Y = np.meshgrid(x, y)
x_internal = np.stack((X.flatten(), Y.flatten()), axis=1)
X_internal = torch.tensor(x_internal,dtype=torch.float32,device=device)
X_internal.requires_grad = True

print('test:',X_test.shape,Y_test.shape,'sup:',X_sup.shape,Y_sup.shape,'Internal:',X_internal.shape)

# boundary point

def get_circle_points(center_x, center_y): 
    theta = torch.linspace(0, 2 * torch.pi, 128, requires_grad=False,device=device)
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
nn_fun = NeuralNetwork.MyNet(2,3,64,4,numCenter)
nn_fun = nn_fun.cuda()

# load ckp
try:
    load_param = torch.load('../checkpoint/nn_fun_params')
    nn_fun.load_state_dict(load_param)
except:
    print('no saved network')

Y_pred = nn_fun(X_internal)


extent = [-8.5, -6.5, -1, 1]


umag_pred = torch.sqrt(Y_pred[:,1:2]**2 + Y_pred[:,2:3]**2)
umag_pred_img = umag_pred.reshape([num_points_y,num_points_x]).detach().cpu()

plt.figure(figsize=(10, 8))
contour = plt.imshow(umag_pred_img, extent=extent)
plt.colorbar(label="Velocity Magnitude")  # 添加颜色条
plt.axis('equal') 
plt.show()


umag_test = torch.sqrt(Y_test[:,1:2]**2 + Y_test[:,2:3]**2)
umag_true = griddata(X_test.detach().cpu(), umag_test.detach().cpu(), (X, Y), method='cubic')


center_x, center_y = -7.5, 0
radius = 0.5
distances = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
umag_true[distances < radius] = 0.0




plt.figure(figsize=(10, 8))
contour = plt.imshow(umag_true, extent=extent)
plt.colorbar(label="Velocity Magnitude")  # 添加颜色条
plt.axis('equal') 
plt.show()



X_center = (Lx_max-Lx_min+2)*torch.sigmoid(nn_fun.X_center.weight[0,0])+(Lx_min-1)
Y_center = (Ly_max-Ly_min+2)*torch.sigmoid(nn_fun.Y_center.weight[0,0])+(Ly_min-1) 

X_center = X_center.detach().cpu().numpy()
Y_center = Y_center.detach().cpu().numpy()

def drawCircle(X_center,Y_center):
    # 生成圆的参数方程
    radius = 0.5
    theta = np.linspace(0, 2 * np.pi, 128)  # 生成 100 个点，覆盖 0 到 2π
    # 绘制圆
    plt.figure()
    
    
    center_x = X_center
    center_y = Y_center
    x = center_x + radius * np.cos(theta)  # 圆的 x 坐标
    y = center_y + radius * np.sin(theta)  # 圆的 y 坐标
    plt.plot(x, y,color='red',label='Predicted')
    
    center_x = -7.5
    center_y = 0.0
    x = center_x + radius * np.cos(theta)  # 圆的 x 坐标
    y = center_y + radius * np.sin(theta)  # 圆的 y 坐标
    plt.plot(x, y,color='black',label='Truth')    
    
    plt.xlabel('x')
    plt.ylabel('y')
    
    plt.xlim(Lx_min-1, Lx_max+1)
    plt.ylim(Ly_min-1, Ly_max+1)
    plt.legend()  # 显示图例
    plt.axis('equal')  # 设置 x 轴和 y 轴的刻度比例相等
    plt.show()
    
    
drawCircle(X_center,Y_center)

