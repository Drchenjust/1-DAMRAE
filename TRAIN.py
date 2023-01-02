# import bcolz 
import importlib
import numpy as np
import torch.utils.data
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm

from torch.autograd import Variable
import torch# import bcolz 
import importlib
import numpy as np
import torch.utils.data
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
from torch.autograd import Variable
import torch

import torch.nn as nn
class SEWeightModule(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels,channels//reduction)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(channels//reduction, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        a,b,c, = x.size()
        out = out.view(a, b)
        
        
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out).view(a, b,1)
        return weight

class identity_block(nn.Module):
    def __init__(self,channels_in,channels_out,ra):
        super(identity_block,self).__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.stride = 1
        self.kernel_size = 3
        self.ra=ra
        self.conv1 = nn.Conv1d(in_channels=self.channels_in,out_channels=self.channels_out,kernel_size=1,stride=1, padding=0)
        self.conv2 = nn.Conv1d(in_channels=self.channels_out,out_channels=self.channels_out,kernel_size=3,stride=1, padding=1)
        self.prl = nn.ReLU()
        self.conv11 = nn.Conv1d(in_channels=self.channels_out,out_channels=self.channels_out//4,kernel_size=self.kernel_size,stride=self.stride, padding=1,dilation=1)
        self.conv12 = nn.Conv1d(in_channels=self.channels_out,out_channels=self.channels_out//4,kernel_size=self.kernel_size,stride=self.stride, padding=3,dilation=3)
        self.conv13 = nn.Conv1d(in_channels=self.channels_out,out_channels=self.channels_out//4,kernel_size=self.kernel_size,stride=self.stride, padding=5,dilation=5)
        self.conv14 = nn.Conv1d(in_channels=self.channels_out,out_channels=self.channels_out//4,kernel_size=self.kernel_size,stride=self.stride, padding=7,dilation=7)
        self.se = SEWeightModule(self.channels_out,self.ra)

        self.conv3 = nn.Conv1d(in_channels=self.channels_out,out_channels=self.channels_out,kernel_size=15,stride=1, padding=7)
        self.rl = nn.ReLU()
        self.maxp = nn.MaxPool1d(kernel_size=2, stride=2)

    
    def forward(self, x):
        x=self.conv1(x)
        x1=x
        x=self.prl(self.conv2(x))
        
        x11=self.conv11(x)
        x12=self.conv12(x)
        x13=self.conv13(x)
        x14=self.conv14(x)
        
        output = torch.cat([x11,x12,x13,x14],1)
        Woutput= self.se(output )

        Woutput=Woutput* output +x1

        Woutput=self.rl(Woutput)
        Woutput=self.maxp(Woutput)
        
        return Woutput

class identity_block3(nn.Module):
    def __init__(self,channels_in,channels_out,ra):
        super(identity_block3,self).__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.stride = 1
        self.kernel_size = 3
        self.ra=ra
        
        self.conv1 = nn.ConvTranspose1d(in_channels=self.channels_in,out_channels=self.channels_out,kernel_size=4,stride=2, padding=1)
        
        self.conv2 = nn.ConvTranspose1d(in_channels=self.channels_out,out_channels=self.channels_out,kernel_size=3,stride=1, padding=1)
        self.prl = nn.ReLU()
        
        self.conv11 = nn.ConvTranspose1d(in_channels=self.channels_out,out_channels=self.channels_out//4,kernel_size=self.kernel_size,stride=self.stride, padding=1,dilation=1)
        self.conv12 = nn.ConvTranspose1d(in_channels=self.channels_out,out_channels=self.channels_out//4,kernel_size=self.kernel_size,stride=self.stride, padding=3,dilation=3)
        self.conv13 = nn.ConvTranspose1d(in_channels=self.channels_out,out_channels=self.channels_out//4,kernel_size=self.kernel_size,stride=self.stride, padding=5,dilation=5)
        self.conv14 = nn.ConvTranspose1d(in_channels=self.channels_out,out_channels=self.channels_out//4,kernel_size=self.kernel_size,stride=self.stride, padding=7,dilation=7)
                           
    
        self.se = SEWeightModule(self.channels_out,self.ra)

        
        self.conv3 = nn.Conv1d(in_channels=self.channels_out,out_channels=self.channels_out,kernel_size=1,stride=1, padding=0)
        self.pr2 = nn.ReLU()


    
    def forward(self, x):
        x=self.conv1(x)
        x1=x
        x=self.prl(self.conv2(x))
        
        x11=self.conv11(x)
        x12=self.conv12(x)
        x13=self.conv13(x)
        x14=self.conv14(x)
        
        output = torch.cat([x11,x12,x13,x14],1)
        Woutput= self.se(output )

        Woutput=Woutput* output +x1
  
        
        Woutput=self.pr2(Woutput)
        Woutput=self.conv3(Woutput)
  
        
        return Woutput

class identity_block4(nn.Module):
    def __init__(self,channels_in,channels_out,ra):
        super(identity_block4,self).__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.stride = 1
        self.kernel_size = 3
        self.ra=ra
        
        self.conv1 = nn.ConvTranspose1d(in_channels=self.channels_in,out_channels=self.channels_out,kernel_size=4,stride=2, padding=1)
        
        self.conv2 = nn.ConvTranspose1d(in_channels=self.channels_out,out_channels=self.channels_out,kernel_size=3,stride=1, padding=1)
        self.prl = nn.ReLU()
        
        self.conv11 = nn.ConvTranspose1d(in_channels=self.channels_out,out_channels=self.channels_out,kernel_size=self.kernel_size,stride=self.stride, padding=1,dilation=1)

        
        self.conv3 = nn.Conv1d(in_channels=self.channels_out,out_channels=self.channels_out,kernel_size=1,stride=1, padding=0)
        self.pr2 = nn.ReLU()


    
    def forward(self, x):
        x=self.conv1(x)
        x1=x
        x=self.prl(self.conv2(x))
        
        x11=self.conv11(x)+x1


        Woutput=self.pr2(x11)
        Woutput=self.conv3(Woutput)
  
        
        return Woutput



S_in=8
class DAE(nn.Module):
    def __init__(self):
        super(DAE,self).__init__()
        self.A1 = identity_block(1,16,2)
        self.A2 = identity_block(16,32,4)
        self.A3 = identity_block(32,64,4)
        self.A4 = identity_block(64,128,8)
        
        self.B1 = identity_block3(128,64,4)
        self.B2 = identity_block3(64,32,4)
        self.B3 = identity_block3(32,16,2)
        self.B4 = identity_block4(16,8,2)
        
        self.B5= nn.ConvTranspose1d(8,1,3,1,1)
        self.SG = nn.Sigmoid()
        self.init_weights()
        
        self.lstm = nn.LSTM(input_size=S_in, hidden_size=S_in, num_layers=1, batch_first=True,bidirectional=True)
        self.fc1 = nn.Linear(S_in*8,S_in)
   
    def init_weights(self):
        """
        Initialize weights for convolution layers using Xavier initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_normal(m.weight.data)      
    
    def forward(self, x):
       
        x1=self.A1(x)
        x2=self.A2(x1)
        x3=self.A3(x2)
        x4=self.A4(x3)
        h1,h2,h3=x4.shape
        x44=x4.reshape(h1*h2,16,8)
        x44, h=self.lstm(x44)
        o1,o2=torch.split(x44,S_in,2)
        x44=(o1+o2)/2
        x44=x44.reshape(h1,h2,h3)
        x5=self.B1(x44)
        x5=x5+x3
        x6=self.B2(x5)
        x6=x6+x2
        x7=self.B3(x6)
        x6=x7+x1
        x8=self.B4(x7)
        x9=self.B5(x8)        
        return self.SG(x9)

model = DAE()
model = model.to('cuda')

TRAINA=np.load(r"E:\data\FINT\TRAINA.npy")
TRAINB=np.load(r"E:\data\FINT\TRAINB.npy")
VAL2=np.load(r"E:\data\FINT\VAL2.npy")
VALB2=np.load(r"E:\data\FINT\VALB2.npy")


import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from torch.autograd import Variable
#from utils import get_training_dataloader, get_test_dataloader
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchsummary import summary

import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
from torch.utils.data import Dataset
class MyVAL(Dataset):
    def __init__(self):

        VAL=torch.from_numpy(VAL2).float().view(45000, 1, 2048)
        VAL1=torch.from_numpy(VALB2).float().view(45000, 1, 2048)


        #y_train=y_train.type(torch.LongTensor)
        
        
        self.x = list(zip(VAL,VAL1))
    def __getitem__(self, idx):
        
        assert idx < len(self.x)
        return self.x[idx]
    def __len__(self):
        
        return len(self.x)

datasetVAL = MyVAL()
dataloaderVAL = DataLoader(datasetVAL, batch_size =256, shuffle=True)

import torch
from torch.utils.data import Dataset
class Mydata(Dataset):
    def __init__(self):

        X_train=torch.from_numpy(TRAINA).float().view(450000, 1, 2048)
        X_train1=torch.from_numpy(TRAINB).float().view(450000, 1, 2048)


        #y_train=y_train.type(torch.LongTensor)
        
        
        self.x = list(zip(X_train,X_train1))
    def __getitem__(self, idx):
        
        assert idx < len(self.x)
        return self.x[idx]
    def __len__(self):
        
        return len(self.x)

dataset = Mydata()
dataloader = DataLoader(dataset, batch_size = 256, shuffle=True)

loss_func1 = nn.BCELoss()
loss_func2 = nn.MSELoss()

import os
def train(epoch):

    loss = 0
    loss2=0
    model.train()
    for batch_index, (X_train,X_train1) in enumerate(dataloader):
        if use_cuda:
            X_train,X_train1 = X_train.cuda(),X_train1.cuda()
        optimizer.zero_grad()#
        outputs2_1= model(X_train)#
        loss1 = loss_func1(outputs2_1,X_train1)
        loss_value=loss1

        loss_value.backward()
        optimizer.step()
        loss += loss_value.item()

    if epoch%1==0:
        g_path = os.path.join('H:/Weight/', 'MODEL-{}.pth'.format(epoch + 1))
        torch.save(model.state_dict(), g_path)
        
    model.eval()
    for batch_index, (VAL,VAL1) in enumerate(dataloaderVAL):
        if use_cuda:
            VAL,VAL1 = VAL.cuda(),VAL1.cuda()
        optimizer.zero_grad()
        
        outputs2_1= model(VAL)
        loss22 =  loss_func2(outputs2_1,VAL1)
        
        loss_value2=loss22
        loss2 += loss_value2.item()    
    print(epoch,"train",loss,"VAL",loss2)
    
use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
EPOCH = 150#迭代100次

optimizer = optim.Adam(model.parameters(), lr=0.001)#随机梯度下降优化
scheduler = StepLR(optimizer, step_size=50, gamma=0.1)#动态学习率调整，每10轮迭代，学习率乘0.5
#categorical_crossentropy


for i in range(1,EPOCH+1):
    train(i)
    scheduler.step()