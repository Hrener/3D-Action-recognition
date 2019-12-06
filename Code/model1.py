#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Hren
import torch
import torch.nn as nn


class ConVNet(nn.Module):
    def __init__(self, in_dim=25, out_dim=30):
        super(ConVNet, self).__init__()
        self.transform = nn.Linear(in_dim, out_dim, bias=False)  # W = (out_dim, in_dim) = (30, 25)
        self.ConVNet_up = nn.Sequential(  # 3*30*30
            torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),  # 32*28*28
            torch.nn.MaxPool2d(stride=2, kernel_size=2),  # 32*14*14
            
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),  # 64*12*12
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=2),  # 128*5*5
            
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),  # 128*10*10
        )
        self.ConVNet_down = nn.Sequential(  # 3*30*30
            torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),  # 32*28*28
            torch.nn.MaxPool2d(stride=2, kernel_size=2),  # 32*14*14
            
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),  # 64*12*12
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=2),  # 128*5*5
            
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),  # 128*10*10
        )
        self.fc1 = nn.Sequential(
            nn.Linear(2 * 128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc2 = nn.Sequential(
            nn.Linear(256, 60),
            #nn.ReLU(),
            #nn.Dropout(0.5),
            #nn.Linear(128, 60),
            # torch.nn.ReLU(),
            # torch.nn.Linear(96, 60),
            # torch.nn.Softmax(dim=1)
        )

    def forward(self, frame_0, frame_1, diff_0, diff_1):  # input(64, 30, 3, 25)
        batch_n_frames = frame_0.shape[0]   # batch_n_frames
        # <1> Skeleton Transformer
        frame_0 = frame_0.permute(0, 1, 3, 2).reshape(batch_n_frames * 30 * 3, 25)
        frame_1 = frame_1.permute(0, 1, 3, 2).reshape(batch_n_frames * 30 * 3, 25)
        diff_0 = diff_0.permute(0, 1, 3, 2).reshape(batch_n_frames * 30 * 3, 25)
        diff_1 = diff_1.permute(0, 1, 3, 2).reshape(batch_n_frames * 30 * 3, 25)
        frame_0 = self.transform(frame_0).reshape(batch_n_frames, 30, 3, 30).permute(0, 2, 1, 3)
        frame_1 = self.transform(frame_1).reshape(batch_n_frames, 30, 3, 30).permute(0, 2, 1, 3)
        diff_0 = self.transform(diff_0).reshape(batch_n_frames, 30, 3, 30).permute(0, 2, 1, 3)
        diff_1 = self.transform(diff_1).reshape(batch_n_frames, 30, 3, 30).permute(0, 2, 1, 3)
        # <2> Two-Stream-CNN
        frame_0_feature_maps = self.ConVNet_up(frame_0)
        frame_1_feature_maps = self.ConVNet_up(frame_1)
        diff_0_feature_maps = self.ConVNet_down(diff_0)
        diff_1_feature_maps = self.ConVNet_down(diff_1)
        # <3> Multi-person Maxout
        frame_feature_maps = torch.max(frame_0_feature_maps, frame_1_feature_maps)
        diff_feature_maps = torch.max(diff_0_feature_maps, diff_1_feature_maps)
        # <4> concat
        feature_maps = torch.cat((frame_feature_maps, diff_feature_maps), 1)
        # <5> ConvNetFlatten
        feature_maps = feature_maps.view(-1, 4096)
        # <6> Fullconnect
        output = self.fc2(self.fc1(feature_maps))
        return output

        # 定义权值初始化
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()
