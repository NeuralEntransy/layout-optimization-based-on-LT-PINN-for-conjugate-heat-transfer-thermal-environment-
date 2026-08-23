
import os
import torch
import numpy as np


# PDE loss
class PDELoss(torch.nn.Module):
    def __init__(self,nn_fun,Lx_min,Lx_max,Ly_min,Ly_max):
        super().__init__()
        self.fun = nn_fun
        self.Lx_min = Lx_min
        self.Lx_max = Lx_max
        self.Ly_min = Ly_min
        self.Ly_max = Ly_max


    def forward(self, X):
        
        x = X[:,0:1]
        y = X[:,1:2]
        X = torch.cat([x,y],dim=1) # [NxNy,2]
        


        Y_pred = self.fun(X) # [NxNyNt,3] 
    
        u = Y_pred[:,0:1]


         
        u_x = torch.autograd.grad(u,x,grad_outputs=torch.ones_like(u),create_graph=True,retain_graph=True)[0]
        u_y = torch.autograd.grad(u,y,grad_outputs=torch.ones_like(u),create_graph=True,retain_graph=True)[0]
        u_xx = torch.autograd.grad(u_x,x,grad_outputs=torch.ones_like(u_x),create_graph=True,retain_graph=True)[0]
        u_yy = torch.autograd.grad(u_y,y,grad_outputs=torch.ones_like(u_y),create_graph=True,retain_graph=True)[0]


        # eq1 = (u_x + v_y) # Coulomb Gauge
        eq1 = u_xx + u_yy  # laplace equation 
        
        # add cylinder weight mask
        X_center = (self.Lx_max-self.Lx_min+2)*torch.sigmoid(self.fun.X_center.weight)+(self.Lx_min-1)
        Y_center = (self.Ly_max-self.Ly_min+2)*torch.sigmoid(self.fun.Y_center.weight)+(self.Ly_min-1)
        
        dist = torch.sqrt((x -X_center)**2 + (y - Y_center)**2)

        
        radius = 0.5
        k = 100
        mask = 1 / (1 + torch.exp(-k * (dist - radius)))
        
        eq1 = mask * eq1 
     


        output1 = torch.sum(torch.square(eq1))
 


        output = output1
        
        return output


# Boundary loss
class BCLoss(torch.nn.Module):
    def __init__(self,nn_fun,Lx_min,Lx_max,Ly_min,Ly_max):
        super().__init__()
        self.fun = nn_fun
        self.Lx_min = Lx_min
        self.Lx_max = Lx_max
        self.Ly_min = Ly_min
        self.Ly_max = Ly_max

    def forward(self, X):
        
        x = X[:,0:1]
        y = X[:,1:2]
        X = torch.cat([x,y],dim=1) # [NxNy,2]
        
        Y_pred = self.fun(X) # [NxNyNt,3] 
    
        u = Y_pred[:,0:1]
        
        u_x = torch.autograd.grad(u,x,grad_outputs=torch.ones_like(u),create_graph=True,retain_graph=True)[0]
        u_y = torch.autograd.grad(u,y,grad_outputs=torch.ones_like(u),create_graph=True,retain_graph=True)[0]

        
        X_center = (self.Lx_max-self.Lx_min+2)*torch.sigmoid(self.fun.X_center.weight)+(self.Lx_min-1)
        Y_center = (self.Ly_max-self.Ly_min+2)*torch.sigmoid(self.fun.Y_center.weight)+(self.Ly_min-1)
        
        #print('T_x/T_y',u_x[0],u_y[0],(x-X_center)[0],(y-Y_center)[0])


        
        r0 = u_x*(x-X_center) + u_y*(y-Y_center) + 0.5

        output0 = torch.sum(torch.square(r0))
            

        output = output0 / len(u)
        
        return output

# Supervise Data Loss
class DataLoss(torch.nn.Module):

    def __init__(self,nn_fun):
        super().__init__()
        self.fun = nn_fun

    def forward(self, X, Y_true):

        Y_pred = self.fun(X) # [NxNy,4] 
              
        r0 = Y_pred[:,0] - Y_true[:,0]

        r0 = torch.sum(torch.square(r0))
    
        e0 = torch.sqrt(r0/torch.sum(torch.square(Y_true[:,0])))

        output = r0  / len(X)
        
        monitor = e0 

        return output, monitor

