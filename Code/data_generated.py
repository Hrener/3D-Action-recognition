#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import h5py
import random
import cv2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset

cv_trn_path = '/home1/rhhHD/Two-stream/cv_trn.hdf5'
cv_tst_path = '/home1/rhhHD/Two-stream/cv_tst.hdf5'
# cv_trn_path = 'D:\Hren Files\My Documents\pycharm projects\毕业设计\BiShe\BS\datantu\cross_view\cv_trn.hdf5'
# cv_tst_path = 'D:\Hren Files\My Documents\pycharm projects\毕业设计\BiShe\BS\datantu\cross_view\cv_tst.hdf5'


class NTUData(Dataset):
    def __init__(self, cv_path, transform=None):
        # TODO
        # 1. Initialize file path or list of file names.
        self.cv_path = cv_path
        self.transform = transform
        with h5py.File(self.cv_path, 'r') as file:
            self.group_name_list = [name for name in file]  # 数据集h5的data name
            # shape[1]取骨架序列的帧数
            self.num_per_frame = None
            self.data_len = file.keys().__len__()  # 骨架序列数据的数目

    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        # 这里需要注意的是，第一步：read one data，是一个data
        index_30 = self.get_frame30_index_list(index)
        # print(index_30.shape)
        frame30_body_0, frame30_body_1, label_action = self.get_frame30_data(index, index_30)   # get segment data ,label
        diff_body_0, diff_body_1 = self.get_diff_data(frame30_body_0, frame30_body_1)   # get skeleton Motion

        frame30_body_0 = Variable(torch.FloatTensor(frame30_body_0))
        frame30_body_1 = Variable(torch.FloatTensor(frame30_body_1))
        diff_body_0 = Variable(torch.FloatTensor(diff_body_0))
        diff_body_1 = Variable(torch.FloatTensor(diff_body_1))
        label_action = torch.LongTensor(label_action).squeeze().numpy().tolist().index(1)

        return frame30_body_0, frame30_body_1, diff_body_0, diff_body_1, label_action


    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return self.data_len

    def get_frame30_index_list(self, index):
        with h5py.File(self.cv_path, 'r') as file:
            self.num_per_frame = file[self.group_name_list[index]].shape[1]  # 每个骨架序列的总帧数
        # 生成31个数是为了得到30个区间
        random_list = np.linspace(0, self.num_per_frame, 31, dtype=int)  # 后面会减一，不会取到第109帧
        # 减一是取每一小区间的最后一位数，避免取到下一区间的第一个数，避免出现相同帧的情况
        frame30_list = [random.randint(random_list[i], random_list[i + 1] - 1) for i in range(30)]
        # frame30_list = sorted(random.sample(range(0, self.num_per_frame), 30))  # index list(1, 30)
        # frame30_index_list = sorted(np.arange(0, self.num_per_frame, 3))
        return frame30_list

    def get_frame30_data(self, index, frame30_list):
        with h5py.File(self.cv_path, 'r') as file:
            frame30_body_0 = file[self.group_name_list[index]][0, frame30_list, :, :]  # 写入30帧数据(30, 25, 3)
            frame30_body_1 = file[self.group_name_list[index]][1, frame30_list, :, :]  # 写入30帧数据
            label_action = file[self.group_name_list[index]].attrs['label']  # 写入label(60, 1)
        return frame30_body_0, frame30_body_1, label_action

    def get_diff_data(self, body0, body1):
        diff_body_0 = np.diff(body0, n=1, axis=0)   # (29, 25, 3)
        diff_body_1 = np.diff(body1, n=1, axis=0)
        diff_zero = np.zeros((1, 25, 3))
        diff_body_0 = np.r_[diff_body_0, diff_zero]
        diff_body_1 = np.r_[diff_body_1, diff_zero]
        return diff_body_0, diff_body_1




if __name__ == '__main__':
    myData = NTUData(cv_tst_path)
    frame_0, frame_1, diff_0, diff_1, label = myData[0]
    print(frame_0.shape, frame_1.shape, diff_0.shape, diff_1.shape, label)
    print(frame_0.dtype, frame_1.dtype, diff_0.dtype, diff_1.dtype)

    # fig = plt.figure()
    # ax = Axes3D(fig)
    #
    # ax.view_init(-85, 120)
    # plt.ion()  # 打开交互模式
    #
    # skeleton_Num = 60
    # for i in range(frame_0.shape[0]):
    #     plt.cla()
    #     x = frame_0[i, :, 0]
    #     y = frame_0[i, :, 1]
    #     z = frame_0[i, :, 2]
    #     xzg = [x[0], x[1], x[20], x[2], x[3]]
    #     yzg = [y[0], y[1], y[20], y[2], y[3]]
    #     zzg = [z[0], z[1], z[20], z[2], z[3]]
    #     xgb = [x[23], x[24], x[11], x[10], x[9], x[8], x[20], x[4], x[5], x[6], x[7], x[22], x[21]]
    #     ygb = [y[23], y[24], y[11], y[10], y[9], y[8], y[20], y[4], y[5], y[6], y[7], y[22], y[21]]
    #     zgb = [z[23], z[24], z[11], z[10], z[9], z[8], z[20], z[4], z[5], z[6], z[7], z[22], z[21]]
    #     xt = [x[19], x[18], x[17], x[16], x[0], x[12], x[13], x[14], x[15]]
    #     yt = [y[19], y[18], y[17], y[16], y[0], y[12], y[13], y[14], y[15]]
    #     zt = [z[19], z[18], z[17], z[16], z[0], z[12], z[13], z[14], z[15]]
    #
    #     ax.plot(xzg, yzg, zzg, color='b', marker='o', markerfacecolor='r')
    #     ax.plot(xt, yt, zt, color='b', marker='o', markerfacecolor='r')
    #     ax.plot(xgb, ygb, zgb, color='b', marker='o', markerfacecolor='r')
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     ax.set_zlabel('Z')
    #     ax.set_facecolor('none')
    #     plt.pause(0.1)
    # plt.ioff()
    # ax.axis('off')
    # plt.show()

    pass