import sys
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

def getVGG3LOutputDimension(inputDimension, outputChannel=128):

    outputDimension = np.ceil(np.array(inputDimension, dtype=np.float32)/2)
    outputDimension = np.ceil(np.array(outputDimension, dtype=np.float32)/2)
    outputDimension = np.ceil(np.array(outputDimension, dtype=np.float32)/2)
    return int(outputDimension) * outputChannel

def getVGG4LOutputDimension(inputDimension, outputChannel=128):

    outputDimension = np.ceil(np.array(inputDimension, dtype=np.float32)/2)
    outputDimension = np.ceil(np.array(outputDimension, dtype=np.float32)/2)
    outputDimension = np.ceil(np.array(outputDimension, dtype=np.float32)/2)
    outputDimension = np.ceil(np.array(outputDimension, dtype=np.float32)/2)
    return int(outputDimension) * outputChannel

class VGG3L(torch.nn.Module):

    def __init__(self, kernel_size):
        super(VGG3L, self).__init__()

        self.conv11 = torch.nn.Conv2d(1, int(kernel_size/4), 3, stride=1, padding=1)
        self.conv12 = torch.nn.Conv2d(int(kernel_size/4), int(kernel_size/4), 3, stride=1, padding=1)
        self.conv21 = torch.nn.Conv2d(int(kernel_size/4), int(kernel_size/2), 3, stride=1, padding=1)
        self.conv22 = torch.nn.Conv2d(int(kernel_size/2), int(kernel_size/2), 3, stride=1, padding=1)
        self.conv31 = torch.nn.Conv2d(int(kernel_size/2), int(kernel_size), 3, stride=1, padding=1)
        self.conv32 = torch.nn.Conv2d(int(kernel_size), int(kernel_size), 3, stride=1, padding=1)
        
    def forward(self, paddedInputTensor):

        paddedInputTensor =  paddedInputTensor.view( paddedInputTensor.size(0),  paddedInputTensor.size(1), 1, paddedInputTensor.size(2)).transpose(1, 2)

        encodedTensorLayer1 = F.relu(self.conv11(paddedInputTensor))
        encodedTensorLayer1 = F.relu(self.conv12(encodedTensorLayer1))
        encodedTensorLayer1 = F.max_pool2d(encodedTensorLayer1, 2, stride=2, ceil_mode=True)

        encodedTensorLayer2 = F.relu(self.conv21(encodedTensorLayer1))
        encodedTensorLayer2 = F.relu(self.conv22(encodedTensorLayer2))
        encodedTensorLayer2 = F.max_pool2d(encodedTensorLayer2, 2, stride=2, ceil_mode=True)

        encodedTensorLayer3 = F.relu(self.conv31(encodedTensorLayer2))
        encodedTensorLayer3 = F.relu(self.conv32(encodedTensorLayer3))
        encodedTensorLayer3 = F.max_pool2d(encodedTensorLayer3, 2, stride=2, ceil_mode=True)
        outputTensor = encodedTensorLayer3.transpose(1, 2)
        outputTensor = outputTensor.contiguous().view(outputTensor.size(0), outputTensor.size(1), outputTensor.size(2) * outputTensor.size(3))

        return outputTensor

class VGG4L(torch.nn.Module):

    def __init__(self, kernel_size):
        super(VGG4L, self).__init__()

        self.conv11 = torch.nn.Conv2d(1, int(kernel_size/8), 3, stride=1, padding=1)
        self.conv12 = torch.nn.Conv2d(int(kernel_size/8), int(kernel_size/8), 3, stride=1, padding=1)
        self.conv21 = torch.nn.Conv2d(int(kernel_size/8), int(kernel_size/4), 3, stride=1, padding=1)
        self.conv22 = torch.nn.Conv2d(int(kernel_size/4), int(kernel_size/4), 3, stride=1, padding=1)
        self.conv31 = torch.nn.Conv2d(int(kernel_size/4), int(kernel_size/2), 3, stride=1, padding=1)
        self.conv32 = torch.nn.Conv2d(int(kernel_size/2), int(kernel_size/2), 3, stride=1, padding=1)
        self.conv41 = torch.nn.Conv2d(int(kernel_size/2), int(kernel_size), 3, stride=1, padding=1)
        self.conv42 = torch.nn.Conv2d(int(kernel_size), int(kernel_size), 3, stride=1, padding=1)
        
    def forward(self, paddedInputTensor):

        paddedInputTensor =  paddedInputTensor.view( paddedInputTensor.size(0),  paddedInputTensor.size(1), 1, paddedInputTensor.size(2)).transpose(1, 2)

        encodedTensorLayer1 = F.relu(self.conv11(paddedInputTensor))
        encodedTensorLayer1 = F.relu(self.conv12(encodedTensorLayer1))
        encodedTensorLayer1 = F.max_pool2d(encodedTensorLayer1, 2, stride=2, ceil_mode=True)

        encodedTensorLayer2 = F.relu(self.conv21(encodedTensorLayer1))
        encodedTensorLayer2 = F.relu(self.conv22(encodedTensorLayer2))
        encodedTensorLayer2 = F.max_pool2d(encodedTensorLayer2, 2, stride=2, ceil_mode=True)

        encodedTensorLayer3 = F.relu(self.conv31(encodedTensorLayer2))
        encodedTensorLayer3 = F.relu(self.conv32(encodedTensorLayer3)) 
        encodedTensorLayer3 = F.max_pool2d(encodedTensorLayer3, 2, stride=2, ceil_mode=True)

        encodedTensorLayer4 = F.relu(self.conv41(encodedTensorLayer3))
        encodedTensorLayer4 = F.relu(self.conv42(encodedTensorLayer4))
        encodedTensorLayer4 = F.max_pool2d(encodedTensorLayer4, 2, stride=2, ceil_mode=True)

        outputTensor = encodedTensorLayer4.transpose(1, 2)
        outputTensor = outputTensor.contiguous().view(outputTensor.size(0), outputTensor.size(1), outputTensor.size(2) * outputTensor.size(3))

        return outputTensor

class ResNet(nn.Module):

    def __init__(self, block, layers):
        super().__init__()
        
        self.inplanes = 64

        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None  
   
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        
        self.inplanes = planes
        
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.conv1(x)           
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)        

        x = self.layer1(x)          # 56x56
        x = self.layer2(x)          # 28x28
        x = self.layer3(x)          # 14x14
        x = self.layer4(x)          # 7x7

        x = self.avgpool(x)         # 1x1
        x = torch.flatten(x, 1)     # remove 1 X 1 grid and make vector of tensor shape 

        return x



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def resnet34():
    layers=[3, 4, 6, 3]
    model = ResNet(BasicBlock, layers)
    return model
