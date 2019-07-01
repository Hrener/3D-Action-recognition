#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import os
import time
import torch.nn as nn
from torch.utils.data import DataLoader
# from Dataset.NewDataset import NTUData
from Data_New.Dataset.RawDataset import NTUData
from Data_New.Model.model_3 import ConVNet
import matplotlib.pyplot as plt

start_time = time.time()
# -------------------------------------------- step 1/3 : 初始化 -------------------------------------------

cv_trn_path = 'cv_trn.hdf5'
cv_tst_path = 'cv_tst.hdf5'

Lr = 0.001 #  0.000547   #（60）
Epochs = 100
Batch_Size = 64
# -------------------------------------------- step 2/3 : 加载数据 -------------------------------------------

# 构建NTUData实例
train_data = NTUData(cv_trn_path)

# 构建DataLoder
train_loader = DataLoader(dataset=train_data, batch_size=Batch_Size, shuffle=True)

# -------------------------------------------- step 3/3 : 训练模型 -------------------------------------------
print('training...')

ConVNet = ConVNet()
ConVNet.cuda()
if os.path.exists('Net_params_Raw_model_3.pkl'):
    ConVNet.load_state_dict(torch.load('Net_params_Raw_model_3.pkl'))
    print('Successfully loaded model parameters...')


fc1_params = list(map(id, ConVNet.fc1.parameters()))
base_params = filter(lambda p: id(p) not in fc1_params, ConVNet.parameters())
print(base_params)
# optimizer = torch.optim.Adam(ConVNet.parameters(), lr=Lr, betas=(0.9, 0.99), weight_decay=0.0005)   # optimize all cnn parameters
optimizer = torch.optim.Adam([{'params': base_params},
                              {'params': ConVNet.fc1.parameters(), 'weight_decay':0.001}], lr=Lr, betas=(0.9, 0.99))
print(ConVNet.fc1.parameters())
# optimizer = torch.optim.Adam(ConVNet.parameters(), lr=Lr, betas=(0.9, 0.99), weight_decay=0.0005)
loss_func = nn.CrossEntropyLoss()                               # the target label is not one-hotted
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98)     # 设置学习率下降策略
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
ConVNet.train()

for epoch in range(Epochs):
    scheduler.step()  # 更新学习率
#     lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
# plt.plot(range(Epochs), lr_list, color='r')
# plt.show()
    epoch_train_correct = 0
    epoch_train_loss = 0

    for batch, data in enumerate(train_loader):
        # 获取数据和标签
        batch_train_correct = 0
        batch_n_frames = data[4].__len__()  # 每批的骨架序列数目，解决最后一批非64的问题
        output = ConVNet(data[0].cuda(), data[1].cuda(), data[2].cuda(), data[3].cuda())

        pred = torch.max(output.data, 1)[1].cuda()
        train_loss = loss_func(output, data[4].cuda())

        optimizer.zero_grad()  # clear gradients for this training step
        train_loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        batch_train_correct += torch.sum(pred == data[4].data.cuda())
        epoch_train_correct += batch_train_correct
        epoch_train_loss += train_loss
        print('epoch : {}/{} | batch : {}/{} | Loss : {:.4f} | num_correct : {} | batch_acc : {:.4f}'.format(epoch,
                Epochs, batch, train_data.__len__() // Batch_Size, train_loss, batch_train_correct, batch_train_correct.__float__() / batch_n_frames.__float__()))

    print('epoch : {}/{} | train_loss : {:.4f} | train_acc : {:.4f}'.format(epoch,Epochs,
                        epoch_train_loss / (train_data.__len__() // Batch_Size),
                        epoch_train_correct.__float__() / train_data.__len__(),))
    with open('train_Raw_model_3_Results', 'a') as f:
        f.write('\nepoch : {}/{} | train_loss : {:.4f} | train_acc : {:.4f}|'.format(epoch+1,300,
                        epoch_train_loss / (train_data.__len__() // Batch_Size),
                        epoch_train_correct.__float__() / train_data.__len__(),))

torch.save(ConVNet.state_dict(), 'Net_params_Raw_model_3.pkl')  # save entire net迭代35次
stop_time = time.time()
print("time is %s" % (stop_time - start_time))