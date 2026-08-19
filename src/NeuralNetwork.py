import os
import torch.nn as nn
import torch
import math
import numpy as np

torch.set_default_dtype(torch.float32)
device = torch.device("cuda:0")
torch.manual_seed(20231028)

################################################################################


# create a neural network

class MyNet(torch.nn.Module):
    def __init__(self,numIN,numOUT,numHidden,numLayer,numCenter):
        super().__init__()
        self.numIN = numIN
        self.numOUT = numOUT
        self.numHidden = numHidden
        self.numLayer = numLayer
        self.numCenter = numCenter
        
        self.X_center = nn.Linear(1,self.numCenter,bias=False,device='cuda:0')
        self.Y_center = nn.Linear(1,self.numCenter,bias=False,device='cuda:0')

        # Input Layer
        self.IN = nn.Linear(self.numIN,self.numHidden,device='cuda:0')
        
        # Hidden Layer        
        self.FC0 = nn.Linear(self.numHidden,self.numHidden,device='cuda:0')
        self.FC1 = nn.Linear(self.numHidden,self.numHidden,device='cuda:0')
        self.FC2 = nn.Linear(self.numHidden,self.numHidden,device='cuda:0')
        self.FC3 = nn.Linear(self.numHidden,self.numHidden,device='cuda:0')
        self.FC4 = nn.Linear(self.numHidden,self.numHidden,device='cuda:0')
        self.FC5 = nn.Linear(self.numHidden,self.numHidden,device='cuda:0')
        self.FC6 = nn.Linear(self.numHidden,self.numHidden,device='cuda:0')
            
        # Output Layer
        self.OUT = nn.Linear(self.numHidden,self.numOUT,device='cuda:0')
        
    def forward(self, inputs):
        
        h = self.IN(inputs)
        h = torch.tanh(h)
        
        h = self.FC0(h)
        h = torch.tanh(h)
        
        h = self.FC1(h)
        h = torch.tanh(h)

        h = self.FC2(h)
        h = torch.tanh(h)

        h = self.FC3(h)
        h = torch.tanh(h)        

        h = self.FC4(h)
        h = torch.tanh(h)   
        
        h = self.FC5(h)
        h = torch.tanh(h)   

        h = self.FC6(h)
        h = torch.tanh(h)   
        
        h = self.OUT(h)

        return h

