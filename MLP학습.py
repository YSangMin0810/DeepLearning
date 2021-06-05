#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np


# # 1.MNIST train,test dataset 가져오기

# In[2]:


mnist_train=dset.MNIST("",train=True,transform=transforms.ToTensor(),
                      target_transform=None, download=True)           #train 용으로 쓰겠다
mnist_test=dset.MNIST("",train=False,transform=transforms.ToTensor(),
                      target_transform=None, download=True)           #test 용으로 쓰겠다


# # 2.대략적인 데이터 형태

# In[3]:


print "mnist_train 길이:",len(mnist_train)
print "mnist_test 길이:",len(mnist_test)

#데이터 하나 형태
image, label = mnist_train.__getitem__(0) #0번째 데이터
print "image data 형태:", image.size()
print "label: ",label

#그리기
img = image.numpy() #image 타입을 numpy 로 변환 (1,28,28)
plt.title("label: %d" %label)
plt.imshow(img[0], cmap='gray')
plt.show()


# # 3.데이터 로드함수
# ## 학습시킬 때 batch_size 단위로 끊어서 로드하기 위함

# In[4]:


# hyper parameters
batch_size = 1024
learning_rate = 0.01 # 0.1, 0.01, 0.001, 0.0001, ...
num_epoch = 400


# In[5]:


train_loader = torch.utils.data.DataLoader(mnist_train,
                                          batch_size=batch_size, # mnist_traind 을 트레인 시키자.
                                          shuffle=True, num_workers=2,
                                          drop_last=True)  # batch_size 만큼 나눌 때 나머지는 버려라
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, 
                                          shuffle=False, num_workers=2,
                                          drop_last=True) 


# ## 데이터 로드함수 이해하기

# In[6]:


n=3 #샘플로 그려볼 데이터 개수
for i, [imgs, labels] in enumerate(test_loader):  # batch_size 만큼
    if i>5:
        break
        
    print "[%d]" %i
    print "한 번에 로드되는 데이터 크기:",len(imgs)
    
    #그리기
    for j in range(n):
        img = imgs[j].numpy() # image 타입을 numpy 로 변환 (1,28,28)
        img = img.reshape((img.shape[1], img.shape[2])) #(1,28,28) -> (28,28)
        #print img.shape
        
        plt.subplot(1,n,j+1) # (1,3) 형태 플랏의 j 번째 자리에 그리겠다
        plt.imshow(img, cmap='gray')
        plt.title("label: %d" %labels[j])
    plt.show()
        


# # 4.모델 선언

# In[7]:


# 모델 선언
# + 퍼셉트론(2 hidden layer) *
model = nn.Sequential(
    nn.Linear(28*28,256),
    nn.Sigmoid(),
    nn.Linear(256,128),
    nn.Linear(128,10),
)
#파라미터 보기
#print(list(model.parameters())) #초기 파라미터 출력


# In[8]:


def ComputeAccr(dloader, imodel):
    correct = 0
    total = 0
    
    for j, [imgs, labels] in enumerate(dloader): # batch_size 만큼
        img = imgs # x
        label = Variable(labels) #y
        #label = Variable(label).cuda()
        #.cuda() : GPU에 로드되기 위함. 만약 CPU로 설정되어 있다면 에러남
        
        #(batch_size,1,28,28)->(batch_size,28,28)
        img = img.reshape((img.shape[0],img.shape[2],img.shape[3]))
        #(batch_size,28,28)->(batch_size,28*28)
        img = img.reshape((img.shape[0],img.shape[1]*img.shape[2]))
        img = Variable(img, requires_grad=False)
        #img = Variable(img, requires_grad=False).cuda()
        
        output = imodel(img) # forward prop.
        _, output_index = torch.max(output,1)
        
        total += label.size(0)
        correct += (output_index == label).sum().float()
    print("Accuracy of Test Data: {}".format(100*correct/total))


# In[9]:


ComputeAccr(test_loader,model)


# # 5.loss, optimizer

# In[10]:


loss_func = nn.CrossEntropyLoss() # logit(# of classes), target(1)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


# In[ ]:


num_epoch = 400
for i in range(num_epoch):
    for j, [imgs, labels] in enumerate(train_loader):
        img = imgs # (batch_size,1,28,28)
        label = Variable(labels) #(batch_size)
        #label = Variable(label).cuda() #(batch_size)
        
        
        #(batch_size,1,28,28)->(batch_size,28,28)
        img = img.reshape((img.shape[0],img.shape[2],img.shape[3]))
        #(batch_size,28,28)->(batch_size,28*28)
        img = img.reshape((img.shape[0],img.shape[1]*img.shape[2]))
        img = Variable(img, requires_grad=False)
        #img = Variable(img, requires_grad=False).cuda()
        
        optimizer.zero_grad()
        output = model(img) # forward prop.
        loss = loss_func(output, label) # logit(# of classes), target(1)
        
        loss.backward() #back prop.
        optimizer.step() # weight 조정
        
    if i%50==0:
        print("%d.."%i)
        ComputeAccr(test_loader, model)
        print loss


# In[ ]:


ComputeAccr(test_loader, model) #98 %(ReLU), 92.48%(ReLU X)


# In[ ]:


netname='./nets/mlp_weight.pkl'
torch.save(model, netname, )
#model = torch.load(netname)

