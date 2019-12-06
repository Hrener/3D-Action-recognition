#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import os
import csv
import torch.nn as nn
from torch.utils.data import DataLoader
from data_generated import NTUData
from model1 import ConVNet
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# -------------------------------------------- step 1/3 : 初始化 -------------------------------------------
cv_trn_path = '/home1/rhhHD/Two-stream/cv_trn.hdf5'
cv_tst_path = '/home1/rhhHD/Two-stream/cv_tst.hdf5'
# cv_trn_path = 'D:\Hren Files\My Documents\pycharm projects\毕业设计\BiShe\BS\datantu\cross_view\cv_trn.hdf5'
# cv_tst_path = 'D:\Hren Files\My Documents\pycharm projects\毕业设计\BiShe\BS\datantu\cross_view\cv_tst.hdf5'

Lr = 0.0002
Epochs = 300
Batch_Size = 64
# -------------------------------------------- step 2/3 : 加载数据 -------------------------------------------
# 构建NTUData实例
train_data = NTUData(cv_trn_path)
test_data = NTUData(cv_tst_path)
# 构建DataLoder
train_loader = DataLoader(dataset=train_data, batch_size=Batch_Size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=Batch_Size, shuffle=False)

# -------------------------------------------- step 3/3 : 训练模型 -------------------------------------------
def train_test(convnet):

    epoch_test_correct = 0
    convnet.eval()
    for batch, data in enumerate(test_loader):
        # 获取数据和标签
        batch_test_correct = 0
        batch_n_frames = data[4].__len__()  # 每批的骨架序列数目，解决最后一批非64的问题

        output = convnet(data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device))
        pred = torch.max(output.data, 1)[1].to(device)

        batch_test_correct += torch.sum(pred == data[4].data.to(device))
        epoch_test_correct += batch_test_correct

        #print('batch : {}/{} |num_correct : {} | batch_acc : {:.4f}'.format(batch, test_data.__len__() // Batch_Size,
        #        batch_test_correct, batch_test_correct.__float__() / batch_n_frames.__float__()))

    #print('test_acc : {:.4f}'.format(epoch_test_correct.__float__() / test_data.__len__()))
    
    return epoch_test_correct.__float__() / test_data.__len__()

def train():
    print('training...')

    convnet = ConVNet().to(device)
    #if os.path.exists('/home1/rhhHD/Two-stream/model1_results/epoch29_loss0.2293_acc0.9236.pkl'):
    #    convnet.load_state_dict(torch.load('/home1/rhhHD/Two-stream/model1_results/epoch29_loss0.2293_acc0.9236.pkl'))
    #    print('Successfully loaded model parameters...')


    fc1_params = list(map(id, convnet.fc1.parameters()))
    base_params = filter(lambda p: id(p) not in fc1_params,
                         convnet.parameters())
    optimizer = torch.optim.Adam([{'params': base_params}, {'params': convnet.fc1.parameters(), 'weight_decay': 0.1}],
                                 lr=Lr, betas=(0.9, 0.99))
    #optimizer = torch.optim.Adam(convnet.parameters(), lr=Lr, betas=(0.9, 0.99), weight_decay=0.01)
    loss_func = nn.CrossEntropyLoss()                               # the target label is not one-hotted
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)     # 设置学习率下降策略

    best_acc = 0.
    lr_list = []
    for epoch in range(Epochs):
        scheduler.step(epoch)  # 更新学习率
    #     lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
    # plt.plot(lr_list, color='r')
    # plt.show()
        epoch_train_correct = 0
        epoch_train_loss = 0

        convnet.train()
        for batch, data in enumerate(train_loader):
            # 获取数据和标签
            batch_train_correct = 0
            batch_n_frames = data[4].__len__()  # 每批的骨架序列数目，解决最后一批非64的问题

            output = convnet(data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device))

            pred = torch.max(output.data, 1)[1].to(device)
            train_loss = loss_func(output, data[4].to(device))

            optimizer.zero_grad()  # clear gradients for this training step
            train_loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            batch_train_correct += torch.sum(pred == data[4].data.to(device))
            epoch_train_correct += batch_train_correct
            epoch_train_loss += train_loss
            if batch%5==0:
                print('epoch : {}/{} | batch : {}/{} | Loss : {:.4f} | num_correct : {} | batch_acc : {:.4f}'.format(epoch,
                    Epochs, batch, train_data.__len__() // Batch_Size, train_loss, batch_train_correct,
                            batch_train_correct.__float__() / batch_n_frames.__float__()))
            #with open('/home1/rhhHD/Two-stream/model2_results/' + 'loss_acc.csv', 'a', encoding='utf-8', newline='') as f:
            #    csv_writer = csv.writer(f)
            #    csv_writer.writerow([round(train_loss.item(), 4),
            #                        round(batch_train_correct.__float__() / batch_n_frames.__float__(), 4)])

        print('epoch : {}/{} | train_loss : {:.4f} | train_acc : {:.4f}'.format(
                            epoch,Epochs,
                            epoch_train_loss / (train_data.__len__() // Batch_Size),
                            epoch_train_correct.__float__() / train_data.__len__()))

        epoch_loss = round(epoch_train_loss.item() / (train_data.__len__() // Batch_Size), 4)
        epoch_acc = round(epoch_train_correct.__float__() / train_data.__len__(), 4)
        
        # test 
        testacc = train_test(convnet)
        

        with open('/home1/rhhHD/Two-stream/model1_results/' + 'train_results.csv', 'a', encoding='utf-8', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([epoch, epoch_loss, epoch_acc, testacc])

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            pkl_name = '/home1/rhhHD/Two-stream/model1_results/' + 'epoch' + str(epoch) + '_loss' + str(epoch_loss) + '_acc' + str(epoch_acc) + '.pkl'
            torch.save(convnet.state_dict(), pkl_name)


train()
