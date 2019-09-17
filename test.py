#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import os
import time
import torch.nn as nn
from torch.utils.data import DataLoader
from Data_New.Dataset.RawDataset import NTUData
# from Dataset.NewDataset import NTUData
from Data_New.Model.model_3 import ConVNet
import matplotlib.pyplot as plt


start_time = time.time()
# -------------------------------------------- step 1/3 : 初始化 -------------------------------------------

cv_trn_path = 'cv_trn.hdf5'
cv_tst_path = 'D:\Hren Files\My Documents\BiShe\BS\datantu\cross_view\cv_tst.hdf5'

Epochs = 1
Batch_Size = 64
# -------------------------------------------- step 2/3 : 加载数据 -------------------------------------------

# 构建NTUData实例
test_data = NTUData(cv_tst_path)

# 构建DataLoder
test_loader = DataLoader(dataset=test_data, batch_size=Batch_Size, shuffle=False)

# -------------------------------------------- step 3/3 : 训练模型 -------------------------------------------
print('training...')

ConVNet = ConVNet()
# print(ConVNet)
if os.path.exists('Net_params_200_Raw_model_3.pkl'):
    ConVNet.load_state_dict(torch.load('Net_params_200_Raw_model_3.pkl', map_location='cpu'))
    print('Successfully loaded model parameters...')



# loss_func = nn.CrossEntropyLoss()                               # the target label is not one-hotted
ConVNet.eval()
for epoch in range(Epochs):
#     lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
# plt.plot(range(Epochs), lr_list, color='r')
# plt.show()
    epoch_test_correct = 0
    epoch_test_loss = 0

    with torch.no_grad():
        for batch, data in enumerate(test_loader):
            # 获取数据和标签
            batch_test_correct = 0
            batch_n_frames = data[4].__len__()  # 每批的骨架序列数目，最后一批非64的问题
            output = ConVNet(data[0], data[1], data[2], data[3])

            pred = torch.max(output.data, 1)[1]
            # test_loss = loss_func(output, data[4])

            batch_test_correct += torch.sum(pred == data[4].data)
            epoch_test_correct += batch_test_correct
            # epoch_test_loss += test_loss
            print('epoch : {}/{} | batch : {}/{} | num_correct : {} | batch_acc : {:.4f}'.format(epoch,
                    Epochs, batch, test_data.__len__() // Batch_Size, batch_test_correct, batch_test_correct.__float__() / batch_n_frames.__float__()))
    print('epoch : {}/{} | test_acc : {:.4f}'.format(epoch,Epochs,
                        epoch_test_correct.__float__() / test_data.__len__()))
    with open('test_Raw_model_3_Results', 'a') as f:
        f.write('\nepoch : {}/{} |  test_acc : {:.4f}'.format(epoch+200,200,
                        epoch_test_correct.__float__() / test_data.__len__()))
stop_time = time.time()
print("time is %s" % (stop_time - start_time))
