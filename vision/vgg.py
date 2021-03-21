"""
Implementation of Basic structure for VGG
"""

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
import torchinfo
from time import time


vgg_config = { # without batch normalization
    'vgg11' : (64, 'max', 128, 'max', 256, 256, 'max', 512, 512, 'max', 512 ,512, 'max'),
    'vgg13' : (64, 64, 'max', 128, 128, 'max', 256, 256, 'max', 512, 512, 'max', 512 ,512, 'max'),
    'vgg16' : (64, 64, 'max', 128, 128, 'max', 256, 256, 256, 'max', 512, 512, 512, 'max', 512 ,512, 512, 'max'),
    'vgg19' : (64, 64, 'max', 128, 128, 'max', 256, 256, 256, 256, 'max', 512, 512, 512, 512, 'max', 512 ,512, 512, 512, 'max'),
}

# class definition
class VGG(nn.Module):
    def __init__(self, model_selection, dim = 224, num_classes = 1000):
        """VGG model configuration

        Args:
            :param model_config: vgg model configuration
            :param dim: size of photo
            :param num_classes: number of classes
        """
        super().__init__()

        self.vgg, dim_shrink_rate, last_channels = self.make_layers(model_selection)
        dim_size = dim // dim_shrink_rate
        # vgg 의 경우 padding = 1 이므로 구현 dim_shrink_rate 와 같은 방법으로 구현 가능
        if dim_size == 0:
            print(self.vgg)
            raise ValueError('Image dimension too small for this network: ',
                             f'Should have dimenstions larger than {dim_shrink_rate}')

        self.linear_input_size = dim_size * dim_size * last_channels

        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.linear_input_size, out_features=4096),
            nn.ReLU(inplace = True),
            nn.Dropout(p=0.5),

            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(in_features=4096, out_features=num_classes)
        )
        self.apply(self.init_weights)

    def forward(self, x):
        x = self.vgg(x)
        x = x.view(-1, self.linear_input_size)
        x = self.classifier(x)
        return x

    def make_layers(self, config): # select key in vgg_config dict
        """
        Borrowed this idea from
        https://github.com/dansuh17/vgg-pytorch/blob/master/model.py
        https://github.com/chengyangfu/pytorch-vgg-cifar10
        -> dim_shrink_rate
        """
        layers = []
        channels = 3
        dim_shrink_rate = 1

        for layer in config:
            if layer == 'max':
                layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
                dim_shrink_rate *= 2
            elif isinstance(layer, int):
                layers.append(nn.Conv2d(in_channels=channels, out_channels=layer, kernel_size=3, stride=1, padding=1))
                layers.append(nn.ReLU(inplace=True))
                channels = layer

        return nn.Sequential(*layers), dim_shrink_rate, channels

    @staticmethod # 정적 메소드 : staticmethod, classmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std = 0.01)
            nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    for vgg_model in ['vgg16', 'vgg19']:
        model_selection = vgg_config['vgg19']
        model = VGG(model_selection, dim=224, num_classes=1000)
        num_samples = 20
        sample = torch.randn(num_samples, 3, 224, 224)
        torchinfo.summary(model, input_data=sample)
        start = time()
        output = model(sample)
        end = time()
        print(output.shape)
        print(f"time : {end-start:.2f}s")

"""
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
├─Sequential: 1-1                        [20, 512, 7, 7]           --
|    └─Conv2d: 2-1                       [20, 64, 224, 224]        1,792
|    └─ReLU: 2-2                         [20, 64, 224, 224]        --
|    └─Conv2d: 2-3                       [20, 64, 224, 224]        36,928
|    └─ReLU: 2-4                         [20, 64, 224, 224]        --
|    └─MaxPool2d: 2-5                    [20, 64, 112, 112]        --
|    └─Conv2d: 2-6                       [20, 128, 112, 112]       73,856
|    └─ReLU: 2-7                         [20, 128, 112, 112]       --
|    └─Conv2d: 2-8                       [20, 128, 112, 112]       147,584
|    └─ReLU: 2-9                         [20, 128, 112, 112]       --
|    └─MaxPool2d: 2-10                   [20, 128, 56, 56]         --
|    └─Conv2d: 2-11                      [20, 256, 56, 56]         295,168
|    └─ReLU: 2-12                        [20, 256, 56, 56]         --
|    └─Conv2d: 2-13                      [20, 256, 56, 56]         590,080
|    └─ReLU: 2-14                        [20, 256, 56, 56]         --
|    └─Conv2d: 2-15                      [20, 256, 56, 56]         590,080
|    └─ReLU: 2-16                        [20, 256, 56, 56]         --
|    └─Conv2d: 2-17                      [20, 256, 56, 56]         590,080
|    └─ReLU: 2-18                        [20, 256, 56, 56]         --
|    └─MaxPool2d: 2-19                   [20, 256, 28, 28]         --
|    └─Conv2d: 2-20                      [20, 512, 28, 28]         1,180,160
|    └─ReLU: 2-21                        [20, 512, 28, 28]         --
|    └─Conv2d: 2-22                      [20, 512, 28, 28]         2,359,808
|    └─ReLU: 2-23                        [20, 512, 28, 28]         --
|    └─Conv2d: 2-24                      [20, 512, 28, 28]         2,359,808
|    └─ReLU: 2-25                        [20, 512, 28, 28]         --
|    └─Conv2d: 2-26                      [20, 512, 28, 28]         2,359,808
|    └─ReLU: 2-27                        [20, 512, 28, 28]         --
|    └─MaxPool2d: 2-28                   [20, 512, 14, 14]         --
|    └─Conv2d: 2-29                      [20, 512, 14, 14]         2,359,808
|    └─ReLU: 2-30                        [20, 512, 14, 14]         --
|    └─Conv2d: 2-31                      [20, 512, 14, 14]         2,359,808
|    └─ReLU: 2-32                        [20, 512, 14, 14]         --
|    └─Conv2d: 2-33                      [20, 512, 14, 14]         2,359,808
|    └─ReLU: 2-34                        [20, 512, 14, 14]         --
|    └─Conv2d: 2-35                      [20, 512, 14, 14]         2,359,808
|    └─ReLU: 2-36                        [20, 512, 14, 14]         --
|    └─MaxPool2d: 2-37                   [20, 512, 7, 7]           --
├─Sequential: 1-2                        [20, 1000]                --
|    └─Linear: 2-38                      [20, 4096]                102,764,544
|    └─ReLU: 2-39                        [20, 4096]                --
|    └─Dropout: 2-40                     [20, 4096]                --
|    └─Linear: 2-41                      [20, 4096]                16,781,312
|    └─ReLU: 2-42                        [20, 4096]                --
|    └─Dropout: 2-43                     [20, 4096]                --
|    └─Linear: 2-44                      [20, 1000]                4,097,000
==========================================================================================
Total params: 143,667,240
Trainable params: 143,667,240
Non-trainable params: 0
Total mult-adds (G): 19.78
==========================================================================================
Input size (MB): 12.04
Forward/backward pass size (MB): 2377.81
Params size (MB): 574.67
Estimated Total Size (MB): 2964.52
==========================================================================================
torch.Size([20, 1000])
time : 8.48s



==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
├─Sequential: 1-1                        [20, 512, 7, 7]           --
|    └─Conv2d: 2-1                       [20, 64, 224, 224]        1,792
|    └─ReLU: 2-2                         [20, 64, 224, 224]        --
|    └─Conv2d: 2-3                       [20, 64, 224, 224]        36,928
|    └─ReLU: 2-4                         [20, 64, 224, 224]        --
|    └─MaxPool2d: 2-5                    [20, 64, 112, 112]        --
|    └─Conv2d: 2-6                       [20, 128, 112, 112]       73,856
|    └─ReLU: 2-7                         [20, 128, 112, 112]       --
|    └─Conv2d: 2-8                       [20, 128, 112, 112]       147,584
|    └─ReLU: 2-9                         [20, 128, 112, 112]       --
|    └─MaxPool2d: 2-10                   [20, 128, 56, 56]         --
|    └─Conv2d: 2-11                      [20, 256, 56, 56]         295,168
|    └─ReLU: 2-12                        [20, 256, 56, 56]         --
|    └─Conv2d: 2-13                      [20, 256, 56, 56]         590,080
|    └─ReLU: 2-14                        [20, 256, 56, 56]         --
|    └─Conv2d: 2-15                      [20, 256, 56, 56]         590,080
|    └─ReLU: 2-16                        [20, 256, 56, 56]         --
|    └─Conv2d: 2-17                      [20, 256, 56, 56]         590,080
|    └─ReLU: 2-18                        [20, 256, 56, 56]         --
|    └─MaxPool2d: 2-19                   [20, 256, 28, 28]         --
|    └─Conv2d: 2-20                      [20, 512, 28, 28]         1,180,160
|    └─ReLU: 2-21                        [20, 512, 28, 28]         --
|    └─Conv2d: 2-22                      [20, 512, 28, 28]         2,359,808
|    └─ReLU: 2-23                        [20, 512, 28, 28]         --
|    └─Conv2d: 2-24                      [20, 512, 28, 28]         2,359,808
|    └─ReLU: 2-25                        [20, 512, 28, 28]         --
|    └─Conv2d: 2-26                      [20, 512, 28, 28]         2,359,808
|    └─ReLU: 2-27                        [20, 512, 28, 28]         --
|    └─MaxPool2d: 2-28                   [20, 512, 14, 14]         --
|    └─Conv2d: 2-29                      [20, 512, 14, 14]         2,359,808
|    └─ReLU: 2-30                        [20, 512, 14, 14]         --
|    └─Conv2d: 2-31                      [20, 512, 14, 14]         2,359,808
|    └─ReLU: 2-32                        [20, 512, 14, 14]         --
|    └─Conv2d: 2-33                      [20, 512, 14, 14]         2,359,808
|    └─ReLU: 2-34                        [20, 512, 14, 14]         --
|    └─Conv2d: 2-35                      [20, 512, 14, 14]         2,359,808
|    └─ReLU: 2-36                        [20, 512, 14, 14]         --
|    └─MaxPool2d: 2-37                   [20, 512, 7, 7]           --
├─Sequential: 1-2                        [20, 1000]                --
|    └─Linear: 2-38                      [20, 4096]                102,764,544
|    └─ReLU: 2-39                        [20, 4096]                --
|    └─Dropout: 2-40                     [20, 4096]                --
|    └─Linear: 2-41                      [20, 4096]                16,781,312
|    └─ReLU: 2-42                        [20, 4096]                --
|    └─Dropout: 2-43                     [20, 4096]                --
|    └─Linear: 2-44                      [20, 1000]                4,097,000
==========================================================================================
Total params: 143,667,240
Trainable params: 143,667,240
Non-trainable params: 0
Total mult-adds (G): 19.78
==========================================================================================
Input size (MB): 12.04
Forward/backward pass size (MB): 2377.81
Params size (MB): 574.67
Estimated Total Size (MB): 2964.52
==========================================================================================
torch.Size([20, 1000])
time : 8.51s

"""

"""comment
parameters : 143,667,240
nn.Linear(Fully connected layer) 층에서 약 120,000,000개의 파라미터가 생성됨.
이 파라미터의 개수를 추후에 다른 모델에서 어떤 방식으로 감소시킬까? 
추후 적용하는 'Global Average Pooling' 은 단순 Flatten 하는 것에 비해서 파라미터와 성능 관점에서 어떤, 어느정도의 영향을 미칠까?

"""

