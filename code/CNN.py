import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import pandas as pd


class CNN(nn.Module):
    def __init__(self, in_channels=1, out_channel=1, bboxes=4, debug=False):
        super().__init__()
        self.debug = debug
        hidden_chs1 = 32
        hidden_chs2 = 64
        hidden_chs3 = 128
        kernel_size = 5
        stride = 2 
        padding = 2
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                          out_channels=hidden_chs1, 
                          stride=stride, padding=padding, 
                          kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(in_channels=hidden_chs1, 
                          out_channels=hidden_chs2, 
                          stride=stride, padding=padding, 
                          kernel_size=kernel_size)
        self.conv3 = nn.Conv2d(in_channels=hidden_chs2, 
                          out_channels=hidden_chs3, 
                          stride=stride, padding=padding, 
                          kernel_size=kernel_size)
        self.batchNorm1 = nn.BatchNorm2d(num_features=hidden_chs1)
        self.batchNorm2 = nn.BatchNorm2d(num_features=hidden_chs2)
        self.batchNorm3 = nn.BatchNorm2d(num_features=hidden_chs3)
        self.MaxPool = nn.MaxPool2d(kernel_size=5, stride=2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        
        self.fcl = nn.Linear(in_features=128*8*8, out_features=256)
        self.cnn_layer = nn.Linear(in_features=256, out_features=out_channel)
        self.regressor = nn.Linear(in_features=256, out_features=bboxes)
    
    def cnn_layers(self, x):
        x = self.relu(x)
        x = self.MaxPool(x)
        return x
        
    def feature_extractor(self, x):
        x = self.conv1(x)
        if self.debug:
            print("after conv1:", x.shape)
        x = self.batchNorm1(x)
        
        x = self.cnn_layers(x)
        if self.debug:
            print("after maxpool:", x.shape)
        
        x =  self.conv2(x)
        if self.debug:
            print("after conv2:", x.shape)
        
        x = self.batchNorm2(x)
        x = self.cnn_layers(x)
        if self.debug:
            print("after maxpool:", x.shape)
        
        x = self.conv3(x)
        if self.debug:
            print("after conv3:", x.shape)
        
        x = self.batchNorm3(x)
        x = self.cnn_layers(x)
        if self.debug:
            print("after maxpool:", x.shape)
        
        x = self.flatten(x)
        if self.debug:
            print("after flatten:", x.shape)
        
        x = self.fcl(x)
        if self.debug:
            print("after fcl:", x.shape)
        return x
    
    def forward(self, x):
        x = self.feature_extractor(x)
        classifier_op = torch.sigmoid(self.cnn_layer(x))
        regressor_op = self.regressor(x)
        return (regressor_op, classifier_op)
    
def loss(bbox_preds, class_preds, bbox_targets, class_targets):
    bbox_loss = F.smooth_l1_loss(bbox_preds, bbox_targets.view(-1, 4))
    class_loss = F.binary_cross_entropy(class_preds, class_targets)
    return bbox_loss, class_loss
        
            
        