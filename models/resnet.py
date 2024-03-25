import torch
import torch.nn as nn
from typing import Optional, Union, List

def conv3x3(_in, _out, stride: int = 1, pad: int = 1):
    return nn.Conv2d(_in,
                     _out,
                     kernel_size = 3,
                     padding = pad,
                     stride= stride)
    
def conv1x1(_in, _out, stride: int = 1, pad: int = 0):
    return nn.Conv2d(_in,
                     _out,
                     kernel_size = 1,
                     padding = pad,
                     stride= stride)
    
class ResBlock(nn.Module):
    def __init__(self, 
                 input_channel:int,
                 output_channel:int,
                 stride:int = 1,
                 downsample: Optional[nn.Module] = None):
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.downsample = downsample
        
        self.conv1 = conv3x3(input_channel, output_channel, stride)
        self.norm1 = nn.BatchNorm2d(output_channel)
        self.act = nn.ReLU()
        self.conv2 = conv3x3(output_channel, output_channel)
        self.norm2 = nn.BatchNorm2d(output_channel)
    
    def forward(self, x):
        
        init = x
        y = self.conv1(x)
        y = self.norm1(y)
        y = self.act(y)
        y = self.conv2(y)
        y = self.norm2(y)
        if self.downsample is not None:
            init = self.downsample(x)
        y += init
        y = self.act(y)
        
        return y

class ResNet18(nn.Module):
    def __init__(self,
                 layers:List[int],
                 input_channel:int = 3,
                 num_classes:int = 10):
        super(ResNet18, self).__init__()
        self.num_classes = num_classes
        self.init_channel = 64
        self.dilation = 1
        
        ## init layer
        self.init_layer = nn.Sequential()
        self.init_layer.add_module('conv1', nn.Conv2d(input_channel, self.init_channel, kernel_size = 7, stride = 2, padding = 3, bias = False))
        self.init_layer.add_module('init norm', nn.BatchNorm2d(self.init_channel))
        self.init_layer.add_module('init activation', nn.ReLU(inplace = True))
        self.init_layer.add_module('init pooling', nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1))
        
        ## residual block
        self.layer1 = self._make_layer(_in = 64, blocks = layers[0])
        self.layer2 = self._make_layer(_in = 128, blocks = layers[1])
        self.layer3 = self._make_layer(_in = 256, blocks = layers[2])
        self.layer4 = self._make_layer(_in = 512, blocks = layers[3])
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self,
                    _in:int,
                    blocks:int,
                    stride:int=1,
                    dilate:bool=False):
        downsample = None
        if stride != 1 or self.init_channel != _in:
            downsample = nn.Sequential()
            downsample.add_module('conv downsample', conv1x1(self.init_channel, _in))
            downsample.add_module('activation', nn.BatchNorm2d(_in))
        layers = []
        layers.append(ResBlock(self.init_channel, _in, stride, downsample = downsample))
        self.init_channel = _in
        for _ in range(1, blocks):
            layers.append(ResBlock(self.init_channel, _in))
        return nn.Sequential(*layers)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.init_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x