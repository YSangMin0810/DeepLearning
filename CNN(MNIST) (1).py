#!/usr/bin/env python
# coding: utf-8

# # Load packages

# In[2]:


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # MNIST train, test dataset 가져오기

# In[3]:


mnist_train=dset.MNIST("",train=True,transform=transforms.ToTensor(),
                      target_transform=None, download=True)           #train 용으로 쓰겠다
mnist_test=dset.MNIST("",train=False,transform=transforms.ToTensor(),
                      target_transform=None, download=True)           #test 용으로 쓰겠다


# # 대략적인 데이터 형태

# In[4]:


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


# # MNIST data 띄워보기

# In[7]:


print(mnist_train[0][1])  # label
print(mnist_train[0][0].size())  # image

for i in range(3):
    img=mnist_train[i][0].numpy()
    print(mnist_train[i][1])
    plt.imshow(img[0],cmap='gray')
    plt.show()


# # convolution 하나 씌워보기

# In[11]:


#mnist 의 첫 번째 이미지, 라벨 가져오기
image, label = mnist_train[0]
#view: tensor 의 사이즈 조절. -1: 해당 차원 차원 확장시켜라
#[1, 28, 28]->[1, 1, 28, 28]
image=image.view(-1, image.size()[0], image.size()[1], image.size()[2])
print(image.size())

print label

#convolution filter 정의
conv_layer=nn.Conv2d(in_channels=1,out_channels=3,kernel_size=3,padding=1)
#image 에 filter 적용
output=conv_layer(Variable(image))
print(output.size())

for i in range(3):
    plt.imshow(output[0,i,:,:].data.numpy(), cmap='gray')
    plt.show()


# # CNN 만들기
# 
# ## train, test data 가져오기 

# In[12]:


import numpy as np
import torch.optim as optim

batch_size = 16
learning_rate = 0.0002
num_epoch = 10  #1000


# In[14]:


# 후에 학습시킬 때 batch_size 단위로 학습시켜나감
train_loader = torch.utils.data.DataLoader(list(mnist_train)[:batch_size*100],
                                          batch_size=batch_size, # mnist_traind 을 트레인 시키자.
                                          shuffle=True, num_workers=2,
                                          drop_last=True)  # batch_size 만큼 나눌 때 나머지는 버려라
test_loader = torch.utils.data.DataLoader((mnist_test), batch_size=batch_size, 
                                          shuffle=False, num_workers=2,
                                          drop_last=True) 


# In[19]:


class CNN(nn.Module):  # nn.Module 상속받음
    def __init__(self):
        super(CNN, self).__init__() # 28 x 28
        self.layer=nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2),
            nn.ReLU(),
        
            nn.Conv2d(16, 32, 5, padding=2),  # 28 x 28
            nn.ReLU(),
            nn.MaxPool2d(2,2),   # 28 x 28-> 14 x 14
            
            nn.Conv2d(32, 64, 5, padding=2),  #14 x 14
            nn.ReLU(),
            nn.MaxPool2d(2,2)  # 14 x 14 -> 7 x 7
        )
        self.fc_layer=nn.Sequential(
            nn.Linear(64*7*7, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )
        
    def forward(self, x):
        out = self.layer(x)
        out = out.view(batch_size, -1)
        out = self.fc_layer(out)
        return out
    
model = CNN() #.cuda


# In[20]:


# 파라미터 체크하기
for parameter in model.parameters():
    #print(parameter)
    print(parameter.shape)


# In[22]:


# loss function, optimaizer 선언
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# # optimization

# In[23]:


for i in range(num_epoch):
    for j, [image, label] in enumerate(train_loader):
        x = Variable(image) #.cuda()
        y_ = Variable(label) #.cuda()
        
        optimizer.zero_grad() # optimizer 안에서 이전 gradient 들을 초기화
        output=model.forward(x)
        loss = loss_func(output, y_)
        loss.backward() # gradient 계산
        optimizer.step() # parameter 업데이트
        
        if j%50==0:
            print(loss, j, i)
        


# In[26]:


# 모델 저장시키기
torch.save(model, 'nets/mycnn_model_%d.pkl'%(num_epoch))


# In[27]:


try:
    #미리 학습시킨 네트워크의 파라미터 집합 [피클]이라 발음함
    model=torch.load('nets/mycnn_model_10.pkl')
    print("model restored")
except:
    print("model noe restored")


# In[28]:


def ComputeAccr(dloader, imodel):
    correct = 0
    total = 0
    
    for j, [imgs, labels] in enumerate(dloader):
        img = Variable(imgs) #.cuda  # x
        label = Variable(labels) # y
        #label = Variable(labels).cuda()
        # .cuda() : GPU 에 로드되기 위함. 만약 cpu로 설정되어 있다면 에러남
        
        output = imodel.forward(img) # forward prop.
        _, output_index = torch.max(output, 1)
        
        total += label.size(0)
        correct += (output_index == label).sum().float()
    print("Accuuracy of Test Data: {}".format(100*correct/total))


# In[29]:


ComputeAccr(test_loader, model)


# In[ ]:




