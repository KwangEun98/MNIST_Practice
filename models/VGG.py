import torch
import torch.nn as nn
from typing import Optional, Union, List

def vgg_conv3(in_channel, out_channel, kernel_size = 3, padding = 1) -> torch.Tensor:
    return nn.Conv2d(in_channels = in_channel,
                     out_channels = out_channel,
                     kernel_size = kernel_size,
                     padding = padding)

def downsample_pooling() -> torch.Tensor:
    return nn.MaxPool2d(kernel_size = 2, stride = 2)

class VGG_19(nn.Module):
    def __init__(self, init_channel = 3, blocks = [2,2,4,4,4], channels = [64, 128, 256, 512, 512], dropout_p = 0.1):
        super().__init__()
        self.init_channel = init_channel
        self.channels = channels
        self.layer1 = self._make_layer(self.init_channel, 64, blocks[0])
        self.layer2 = self._make_layer(128, 128, blocks[1])
        self.layer3 = self._make_layer(256, 256, blocks[2])
        self.layer4 = self._make_layer(512, 512, blocks[3])
        self.layer5 = self._make_layer(512, 512, blocks[4])
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        self.fc = nn.Sequential()
        self.add_module("Linear1", nn.Linear(512 * 7 * 7, 4096))
        self.add_module("ReLU1", nn.ReLU(inplace = True))
        self.add_module("Dropout1", nn.Dropout(p=dropout_p))
        self.add_module("Linear2", nn.Linear(4096, 4096))
        self.add_module("ReLU1", nn.ReLU(inplace = True))
        self.add_module("Dropout1", nn.Dropout(p=dropout_p))

    def _make_layer(self, _in, conv_channel, block):
        layers = []
        init = _in
        for i in range(1, block+1):
            if (i == block) & (init != 512):
                layers.append(vgg_conv3(init, conv_channel * 2))
                layers.append(nn.BatchNorm2d(num_features = conv_channel*2))
                if init == 512:
                    continue
            else:
                layers.append(vgg_conv3(init, conv_channel))
                layers.append(nn.BatchNorm2d(num_features = conv_channel))
            layers.append(nn.ReLU(inplace = True))   
            init = conv_channel
        layers.append(downsample_pooling())
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        ## MNIST에서 차원 문제로 layer5는 생략
        if x.shape[2] < 2:
            pass
        else:
            x = self.layer5(x)

        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
if __name__ == "__main__":
    model = VGG_19(init_channel = 1, blocks = [2,2,2,2,4])
    print(model)
    sample_tensor = torch.randn(32, 1, 28, 28)
    model(sample_tensor)