'''
EECS 445 - Introduction to Machine Learning
Winter 2019 - Project 2
Challenge
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.challenge import Challenge
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

# Residual block
class block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class Challenge(nn.Module):
    def __init__(self, block):
        super(Challenge, self).__init__()
        self.in_channels = 3
        self.layer1 = self.make_layer(block, 64, 2)
        self.layer2 = self.make_layer(block, 32, 2)
        self.fc1 = nn.Linear(32768, 128)
        self.fc2 = nn.Linear(128, 10)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        N, C, H, W = x.shape
        z = F.elu(self.layer1(x))
        z = F.elu(self.layer2(z))
        z = z.view(N, -1)
        z = F.elu(self.fc1(z))
        z = F.elu(self.fc2(z))
        return z
