
# coding: utf-8

# In[1]:

from __future__ import print_function
import argparse
import os
import random
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.parallel
from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

import torch
#import torch.nn.functional as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
from tensorflow.examples.tutorials.mnist import input_data

torch.cuda.set_device(7)


# In[2]:

""" ==================== DATALOADER ======================== """

train = dset.MNIST('./data', train=True, download=False,
                   transform=transforms.Compose([
                       transforms.ToTensor()
                   ]))

train_loader = torch.utils.data.DataLoader(train, batch_size = 1,shuffle=False)
test = dset.MNIST('./data', train=False, download=False,
                   transform=transforms.Compose([
                       transforms.ToTensor()
                   ]))

test_loader = torch.utils.data.DataLoader(test, batch_size = 1,shuffle=False)

dataiter = iter(train_loader)
images, labels = dataiter.next()
labels


# In[3]:

mb_size = 32
Z_dim = 16
X_dim = 28
y_dim = 28
h_dim = 128
cnt = 0
lr = 1e-3


# In[4]:

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return Variable(torch.randn(*size) * xavier_stddev, requires_grad=True)


# In[5]:

""" ==================== GENERATOR ======================== """

class _netG(nn.Module):
    def __init__(self):
        super(_netG, self).__init__()
        main = nn.Sequential()
        # input is Z, going into a convolution
        self.relu = nn.ReLU(True)
        self.fc1 = nn.Linear(74, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 7*7*128)        
        self.bn2 = nn.BatchNorm2d(128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.upconv2 = nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False)
        
    def forward(self, x):
        x = x.view(-1,74)
        x = self.fc1(x)
      #  x = x.view(1,1024,1,1)
        x = self.relu(x)
        x = self.bn1(x)
       # x = x.view(-1,1024)
        x = self.fc2(x)
        x = x.view(1,128,7,7)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.upconv1(x)
        x = self.relu(x)
        x = self.bn3(x)
        x = self.upconv2(x)
        return x

       # self.bn2
      #  self.bn2
      #  self.upconv1
      #  self.bn1
      #  self.upconv2
       # main.add_module('reshape',view([1,7,7,128]))
        """main.add_module('relu',nn.ReLU(True))
        main.add_module('bn',nn.BatcFFFhNorm2d(1024))

        main.add_module('fc1',nn.Linear(74, 1024)
        main.add_module('relu',nn.ReLU(True))
        main.add_module('bn',nn.BatchNorm2d(1024))

                        
        main.add_module('initial.{0}.relu'.format(cngf),
                        nn.ReLU(True))

        csize, cndf = 4, cngf
        isize = 256
        while csize < isize//2:
            main.add_module('pyramid.{0}-{1}.convt'.format(cngf, cngf//2),
                            nn.ConvTranspose2d(cngf, cngf//2, 4, 2, 1, bias=False))
            main.add_module('pyramid.{0}.batchnorm'.format(cngf//2),
                            nn.BatchNorm2d(cngf//2))
            main.add_module('pyramid.{0}.relu'.format(cngf//2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        main.add_module('final.{0}-{1}.convt'.format(cngf, nc),
                        nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module('final.{0}.tanh'.format(nc),
                        nn.Tanh())
        self.main = main
    def forward(self, input):
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.ngpu)
        return nn.parallel.data_parallel(self.main, input, gpu_ids)
    """

    
#Wzh = xavier_init(size=[Z_dim + 10, h_dim])
#bzh = Variable(torch.zeros(h_dim), requires_grad=True)

#Whx = xavier_init(size=[h_dim, X_dim])
#bhx = Variable(torch.zeros(X_dim), requires_grad=True)

#x = Variable(torch.zeros(1,74,1,1), requires_grad = True)
#netD.apply(weights_init)
netG = _netG()
print (netG)
#output = netG(x)
#output


# In[6]:

""" ==================== DISCRIMINATOR  ======================== """
class _netD(nn.Module):
    def __init__(self):
        super(_netD, self).__init__()
        main = nn.Sequential()
        # input is Z, going into a convolution
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.cov1 = nn.Conv2d(1, 64, 4, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.cov2 =nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(4608, 1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 1)
        self.fc3 = nn.Linear(1024, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 10)
        
    def forward(self, x, mode):
        x = self.lrelu(x)
        x = self.cov1(x)
        x = self.bn1(x)
        x = self.cov2(x)
        x = self.lrelu(x)
        x = self.bn2(x)
        x = x.view(1,-1)
        x = self.fc(x)
       # x = x.view(1,-1,1,1)
        x = self.lrelu(x)
        x = self.bn3(x)
        if mode == 'D':
            x = self.fc2(x)
        if mode == 'Q':
            x = self.fc3(x)
            x = self.lrelu(x)
            x = self.bn4(x)
            x = self.fc4(x)
        return x

x = Variable(torch.zeros(1,1,28,28), requires_grad = True)
#netD.apply(weights_init)
netD = _netD()
#print (netD)
#output = netD(x,'D')
#output


# In[7]:

def weights_init(m):
    classname = m.__class__.__name__
   # print (classname)
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        print (classname)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        print (classname)

netG.apply(weights_init)
netD.apply(weights_init)


# In[8]:

""" ====================== OPTIMISER ========================== """
optimizerD = optim.Adam(netD.parameters(), lr = 0.0002, betas = (0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr = 0.0002, betas = (0.5, 0.999))


# In[9]:

input = torch.FloatTensor(1, 1, 28, 28)
noise = torch.FloatTensor(1, 74, 1, 1)

fixed_noise = torch.FloatTensor(np.random.multinomial(1, 10*[0.1], size=1))
c = torch.FloatTensor(np.random.multinomial(1, 10*[0.1], size=1).reshape((1,10,1,1)))
z = torch.randn(1, 64,1,1)

label = torch.FloatTensor(1)
label_Q = torch.ones(10)

real_label = 1
fake_label = 0


# In[10]:

#real_label = torch.FloatTensor(1)
#fake_label = torch.FloatTensor(0)
criterion = nn.BCELoss()


# In[11]:

netD.cuda()
netG.cuda()
criterion.cuda()
input, label, label_Q = input.cuda(), label.cuda(), label_Q.cuda()
noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
z, c = z.cuda(), c.cuda()


# In[12]:

input = Variable(input)
noise = Variable(noise)
fixed_noise = Variable(fixed_noise)
c = Variable(c)
z = Variable(z)


# In[13]:

label = Variable(label)
label_Q = Variable(label_Q)


# In[14]:

""" ======================TRAIN========================== """
for p in netD.parameters():
    p.grad.data.zero_()

for p in netG.parameters():
    p.grad.data.zero_()


# In[15]:

def sample_c():
    rand_c = np.random.multinomial(1, 10*[0.1], size=1)
    rand_c = np.resize(rand_c,(1,10,1,1))
    rand_c = torch.from_numpy(rand_c.astype('float32'))
    c.data.copy_(rand_c)
    return c

def generate_z():
    rand_z = torch.randn(1, 64,1,1)
    rand_z.normal_(0, 1)
    z.data.resize_(rand_z.size()).copy_(rand_z)
    return z


# In[ ]:

for epoch in range(100000):
    for i, data in enumerate(train_loader, 0):
        netD.zero_grad()
        image, labels = data
        z.data.normal_(0, 1)
        c = sample_c()
        noise = torch.cat([z, c], 1)

    # update D
        input.data.resize_(image.size()).copy_(image)
        G_sample = netG(noise)
        D_real = netD(input,'D')
        D_fake = netD(G_sample,'D')

        label.data.resize_(1).fill_(real_label)
        D_loss_real = criterion(D_real, label)
        D_loss_real.backward()
        
        label.data.resize_(1).fill_(fake_label)
        D_loss_fake = criterion(D_fake, label)
        D_loss_fake.backward()
        
        D_loss = D_loss_real + D_loss_fake
        optimizerD.step()
        
    # update G  
        netG.zero_grad()
        G_sample = netG(noise)
        D_fake = netD(G_sample,'D')
        
        label.data.resize_(1).fill_(real_label)
        G_loss = criterion(D_fake, label)
        G_loss.backward()
        optimizerD.step()

    # update Q
        netG.zero_grad()
        G_sample = netG(noise)
        Q_c_given_x = netD(G_sample,'Q')
        
        crossent_loss = criterion(Q_c_given_x, label_Q)
        ent_loss = criterion(c, label_Q)
        mi_loss = crossent_loss + ent_loss

        mi_loss.backward()
        optimizerD.step()
        
        if i % 10000 == 0:
            print (i)

    print (epoch)    
    if epoch % 3 == 0:
        noise = torch.cat((z, fixed_noise), 1)
        fake = netG(noise)
        vutils.save_image(fake.data, str(epoch) + '.png' )


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



