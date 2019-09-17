#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import h5py
import random
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset

cv_trn_path = 'D:\Hren Files\My Documents\BiShe\BSpython\Hren-Action-recognition\cross_view_data\cv_trn.hdf5'
cv_tst_path = 'D:\Hren Files\My Documents\BiShe\BSpython\Hren-Action-recognition\cross_view_data\cv_tst.hdf5'


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
        index_30 = self.get_frame30_index_list(index)   # get the index of 30 frames
        frame30_body_0, frame30_body_1, label_action = self.get_frame30_data(index, index_30)   # get 30 frames data ,label
        diff_body_0, diff_body_1 = self.get_diff_data(frame30_body_0, frame30_body_1)   # get skeleton Motion

        frame30_body_0 = Variable(torch.FloatTensor(frame30_body_0))
        frame30_body_1 = Variable(torch.FloatTensor(frame30_body_1))
        diff_body_0 = Variable(torch.FloatTensor(diff_body_0))
        diff_body_1 = Variable(torch.FloatTensor(diff_body_1))

        label_action = torch.LongTensor(label_action)
        label_action = label_action.squeeze()
        label_action = label_action.numpy().tolist().index(1)
        #
        # T = Transformer(25, 30)
        #
        # frame_0 = torch.zeros((30, 30, 3))
        # frame_1 = torch.zeros((30, 30, 3))
        # diff_0 = torch.zeros((30, 30, 3))
        # diff_1 = torch.zeros((30, 30, 3))
        #
        # for frame in range(30):
        #     frame_0[frame, :, :] = T(frame30_body_0[frame, :, :])
        #     frame_1[frame, :, :] = T(frame30_body_1[frame, :, :])
        #     diff_0[frame, :, :] = T(diff_body_0[frame, :, :])
        #     diff_1[frame, :, :] = T(diff_body_1[frame, :, :])

        # frame_0 = frame_0.permute(2, 1, 0)
        # frame_1 = frame_1.permute(2, 1, 0)
        # diff_0 = diff_0.permute(2, 1, 0)
        # diff_1 = diff_1.permute(2, 1, 0)
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
    # my_data = MyDataSets(cv_trn_path)
    # for i in range(1):
    #     frame30_index = my_data.get_frame30_index_list(my_data.group_name_list[i])
    #     frame30_0, frame30_1, label = my_data.get_frame30_data(my_data.group_name_list[i], frame30_index)
    #     diff_0, diff_1 = my_data.get_diff_data(frame30_0, frame30_1)
    #
    #     print(frame30_index)
    #     print(frame30_0)
    #     print(label.shape)
    #     print(diff_0, diff_0.shape)
    myData = NTUData(cv_trn_path)
    frame_0, frame_1, diff_0, diff_1, label = myData[0]
    print(frame_0.shape, frame_1.shape, diff_0.shape, diff_1.shape, label)
    print(frame_0.dtype, frame_1.dtype, diff_0.dtype, diff_1.dtype)
    # pass
