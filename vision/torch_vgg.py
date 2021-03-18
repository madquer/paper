import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
import torchinfo
import torchsummary
from time import time

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn'
]
# python __all__ : import 할 때 한번에 불러오는 모든 변수   

# pytorch version
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

# class definition
class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights = True):
        super(VGG, self).__init__()
        
        self.features = features

        self.avgpool = nn.AdaptiveAvgPool2d((7,7)) 
        # 이 사이즈를 어떻게 먼저 알지? 애초에 먼저 설계하는건가?
        # 논문에 따르면, conv layer 에 모두 padding 을 주었으므로, layer 가 바뀌더라도 계산이 쉽게 가능
        # average pooling 이 논문에 나왔었나? 왜 굳이 두번 일을 하지?
        # TODO: why average pooling?

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7 , 4096),
            nn.ReLU(inplace = True), # 함수형이 아니므로, inplace = True
            nn.Dropout(), # AlexNet 으로부터 시작
            
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(),

            nn.Linear(4096, num_classes)
        )
        if init_weights:
            self._initialize_weights()
            
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x) 
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
        

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') # kaiming He 초깃값
                # He : ReLU / Xavier : sigmoid or tanh
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1) # TODO: why?
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean = 0, std = 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3 # RGB
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

        


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}



# 'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn'


# VGG basic
def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs): # **kwargs : dict
    if pretrained:
        kwargs['init_weights'] = False # weight initialization = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs) # arg : features, (num_classes, init_weights)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model

def vgg11(pretrained=False, progress=True, **kwargs):
    r""" VGG 11-layer model (configuration "A") from
    "Very Deep Convolutional Networks For Large-Scale Image Recognition"

    Args:
        :param pretrained: If True, returns a model pre-trained on ImageNet
        :param progress: If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11', 'A', False, pretrained, progress, **kwargs)

def vgg11_bn(pretrained=False, progress=True, **kwargs):
    return _vgg('vgg11_bn', 'A', True, pretrained, progress, **kwargs)

def vgg13(pretrained=False, progress=True, **kwargs):
    return _vgg('vgg13', 'B', False, pretrained, progress, **kwargs)

def vgg13_bn(pretrained=False, progress=True, **kwargs):
    return _vgg('vgg13_bn', 'B', True, pretrained, progress, **kwargs)

def vgg16(pretrained=False, progress=True, **kwargs):
    return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)

def vgg16_bn(pretrained=False, progress=True, **kwargs):
    return _vgg('vgg16_bn', 'D', True, pretrained, progress, **kwargs)

def vgg19(pretrained=False, progress=True, **kwargs):
    return _vgg('vgg19', 'E', False, pretrained, progress, **kwargs)

def vgg19_bn(pretrained=False, progress=True, **kwargs):
    return _vgg('vgg19_bn', 'E', True, pretrained, progress, **kwargs)



if __name__ == '__main__':
    length = 5
    sample = torch.randn((length, 3, 224, 224))

    vgg11_bn_net = vgg11_bn(pretrained=False, progress=True, init_weights=False)
    vgg19_net = vgg19(pretrained=False, progress=True, init_weights = False)

    torchinfo.summary(vgg11_bn_net, input_size=sample.size())
    print('\n**********************************************************************\n')
    torchinfo.summary(vgg19_net, input_size=sample.size())
    print('\n**********************************************************************\n')

    start = time()
    output11 = vgg11_bn_net(sample)
    mid = time()
    output19 = vgg19_net(sample)
    end = time()
    print('vgg11bn :', output11.size(), f'time : {mid-start:.2f}')
    print('vgg19 :', output19.size(), f'time : {end-mid:.2f}')


"""
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
├─Sequential: 1-1                        [5, 512, 7, 7]            --
|    └─Conv2d: 2-1                       [5, 64, 224, 224]         1,792
|    └─BatchNorm2d: 2-2                  [5, 64, 224, 224]         128
|    └─ReLU: 2-3                         [5, 64, 224, 224]         --
|    └─MaxPool2d: 2-4                    [5, 64, 112, 112]         --
|    └─Conv2d: 2-5                       [5, 128, 112, 112]        73,856
|    └─BatchNorm2d: 2-6                  [5, 128, 112, 112]        256
|    └─ReLU: 2-7                         [5, 128, 112, 112]        --
|    └─MaxPool2d: 2-8                    [5, 128, 56, 56]          --
|    └─Conv2d: 2-9                       [5, 256, 56, 56]          295,168
|    └─BatchNorm2d: 2-10                 [5, 256, 56, 56]          512
|    └─ReLU: 2-11                        [5, 256, 56, 56]          --
|    └─Conv2d: 2-12                      [5, 256, 56, 56]          590,080
|    └─BatchNorm2d: 2-13                 [5, 256, 56, 56]          512
|    └─ReLU: 2-14                        [5, 256, 56, 56]          --
|    └─MaxPool2d: 2-15                   [5, 256, 28, 28]          --
|    └─Conv2d: 2-16                      [5, 512, 28, 28]          1,180,160
|    └─BatchNorm2d: 2-17                 [5, 512, 28, 28]          1,024
|    └─ReLU: 2-18                        [5, 512, 28, 28]          --
|    └─Conv2d: 2-19                      [5, 512, 28, 28]          2,359,808
|    └─BatchNorm2d: 2-20                 [5, 512, 28, 28]          1,024
|    └─ReLU: 2-21                        [5, 512, 28, 28]          --
|    └─MaxPool2d: 2-22                   [5, 512, 14, 14]          --
|    └─Conv2d: 2-23                      [5, 512, 14, 14]          2,359,808
|    └─BatchNorm2d: 2-24                 [5, 512, 14, 14]          1,024
|    └─ReLU: 2-25                        [5, 512, 14, 14]          --
|    └─Conv2d: 2-26                      [5, 512, 14, 14]          2,359,808
|    └─BatchNorm2d: 2-27                 [5, 512, 14, 14]          1,024
|    └─ReLU: 2-28                        [5, 512, 14, 14]          --
|    └─MaxPool2d: 2-29                   [5, 512, 7, 7]            --
├─AdaptiveAvgPool2d: 1-2                 [5, 512, 7, 7]            --
├─Sequential: 1-3                        [5, 1000]                 --
|    └─Linear: 2-30                      [5, 4096]                 102,764,544
|    └─ReLU: 2-31                        [5, 4096]                 --
|    └─Dropout: 2-32                     [5, 4096]                 --
|    └─Linear: 2-33                      [5, 4096]                 16,781,312
|    └─ReLU: 2-34                        [5, 4096]                 --
|    └─Dropout: 2-35                     [5, 4096]                 --
|    └─Linear: 2-36                      [5, 1000]                 4,097,000
==========================================================================================
Total params: 132,868,840
Trainable params: 132,868,840
Non-trainable params: 0
Total mult-adds (G): 7.74
==========================================================================================
Input size (MB): 3.01
Forward/backward pass size (MB): 594.45
Params size (MB): 531.48
Estimated Total Size (MB): 1128.94
==========================================================================================

**********************************************************************

==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
├─Sequential: 1-1                        [5, 512, 7, 7]            --
|    └─Conv2d: 2-1                       [5, 64, 224, 224]         1,792
|    └─ReLU: 2-2                         [5, 64, 224, 224]         --
|    └─Conv2d: 2-3                       [5, 64, 224, 224]         36,928
|    └─ReLU: 2-4                         [5, 64, 224, 224]         --
|    └─MaxPool2d: 2-5                    [5, 64, 112, 112]         --
|    └─Conv2d: 2-6                       [5, 128, 112, 112]        73,856
|    └─ReLU: 2-7                         [5, 128, 112, 112]        --
|    └─Conv2d: 2-8                       [5, 128, 112, 112]        147,584
|    └─ReLU: 2-9                         [5, 128, 112, 112]        --
|    └─MaxPool2d: 2-10                   [5, 128, 56, 56]          --
|    └─Conv2d: 2-11                      [5, 256, 56, 56]          295,168
|    └─ReLU: 2-12                        [5, 256, 56, 56]          --
|    └─Conv2d: 2-13                      [5, 256, 56, 56]          590,080
|    └─ReLU: 2-14                        [5, 256, 56, 56]          --
|    └─Conv2d: 2-15                      [5, 256, 56, 56]          590,080
|    └─ReLU: 2-16                        [5, 256, 56, 56]          --
|    └─Conv2d: 2-17                      [5, 256, 56, 56]          590,080
|    └─ReLU: 2-18                        [5, 256, 56, 56]          --
|    └─MaxPool2d: 2-19                   [5, 256, 28, 28]          --
|    └─Conv2d: 2-20                      [5, 512, 28, 28]          1,180,160
|    └─ReLU: 2-21                        [5, 512, 28, 28]          --
|    └─Conv2d: 2-22                      [5, 512, 28, 28]          2,359,808
|    └─ReLU: 2-23                        [5, 512, 28, 28]          --
|    └─Conv2d: 2-24                      [5, 512, 28, 28]          2,359,808
|    └─ReLU: 2-25                        [5, 512, 28, 28]          --
|    └─Conv2d: 2-26                      [5, 512, 28, 28]          2,359,808
|    └─ReLU: 2-27                        [5, 512, 28, 28]          --
|    └─MaxPool2d: 2-28                   [5, 512, 14, 14]          --
|    └─Conv2d: 2-29                      [5, 512, 14, 14]          2,359,808
|    └─ReLU: 2-30                        [5, 512, 14, 14]          --
|    └─Conv2d: 2-31                      [5, 512, 14, 14]          2,359,808
|    └─ReLU: 2-32                        [5, 512, 14, 14]          --
|    └─Conv2d: 2-33                      [5, 512, 14, 14]          2,359,808
|    └─ReLU: 2-34                        [5, 512, 14, 14]          --
|    └─Conv2d: 2-35                      [5, 512, 14, 14]          2,359,808
|    └─ReLU: 2-36                        [5, 512, 14, 14]          --
|    └─MaxPool2d: 2-37                   [5, 512, 7, 7]            --
├─AdaptiveAvgPool2d: 1-2                 [5, 512, 7, 7]            --
├─Sequential: 1-3                        [5, 1000]                 --
|    └─Linear: 2-38                      [5, 4096]                 102,764,544
|    └─ReLU: 2-39                        [5, 4096]                 --
|    └─Dropout: 2-40                     [5, 4096]                 --
|    └─Linear: 2-41                      [5, 4096]                 16,781,312
|    └─ReLU: 2-42                        [5, 4096]                 --
|    └─Dropout: 2-43                     [5, 4096]                 --
|    └─Linear: 2-44                      [5, 1000]                 4,097,000
==========================================================================================
Total params: 143,667,240
Trainable params: 143,667,240
Non-trainable params: 0
Total mult-adds (G): 19.78
==========================================================================================
Input size (MB): 3.01
Forward/backward pass size (MB): 594.45
Params size (MB): 574.67
Estimated Total Size (MB): 1172.13
==========================================================================================

**********************************************************************

vgg11bn : torch.Size([5, 1000]) time : 0.98
vgg19 : torch.Size([5, 1000]) time : 2.21
"""