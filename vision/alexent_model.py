"""
This code to be praticed is refered by 'paper with code'.
link : https://github.com/dansuh17/alexnet-pytorch/blob/d0c1b1c52296ffcbecfbf5b17e1d1685b4ca6744/model.py#L40
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

# define pytorch device - useful for device-agnostic execution
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define model parameters
NUM_EPOCHS = 90 # original paper
BATCH_SIZE = 128
MOMENTUM = 0.9
LR_DECAY = 0.0005
LR_INIT = 0.01
IMAGE_DIM = 227 # pixels
NUM_CLASSES = 1000 # 1000 classes for imagenet 2012 dataset
DEVICE_IDS = [0,1,2,3] # GPUs to use

# modify this to point to your data directory
INPUT_ROOT_DIR = 'alexnet_data_in'
TRAIN_IMG_DIR = 'alexnet_data_in/imagenet'
OUTPUT_DIR = 'alexnet_data_out'
LOG_DIR = OUTPUT_DIR + '/tblogs' # tensorboard logs
CHECKPOINT_DIR = OUTPUT_DIR + '/models' # model checkpoints

# make checkpoint path directory
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# model
class AlexNet(nn.Module):
    """
    Neural network model consisting of layers propsed by AlexNet paper
    """
    def __init__(self, num_classes=1000):
        """
        Define and allocate layers for this neural net.

        :param num_classes (int) : number of classes to predict with this model
        """
        super().__init__()
        # input size should be : (b x 3 x 227 x 227)
        # The image in the original paper states that width and height
        # The dimensions after first convolution layer do not lead to 55 x 55.

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4), # (b x 96 x 55 x 55)
            # output_size(O) = (I - K + 2P)/S + 1 = (227 - 11 + 2 * 0)/4 + 1 = 55
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.001, beta=0.75, k=2), # section 3.3
            nn.MaxPool2d(kernel_size=3, stride=2), # (b x 96 x 27 x 27)
            # output_size(O) = (55 - 3 + 2 * 0)/2 + 1 = 27 : pooling 도 동일

            nn.Conv2d(96, 256, 5, padding=2), # (b x 256 x 27 x 27)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k = 2),
            nn.MaxPool2d(kernel_size=3, stride=2), # (b x 256 x 13 x 13)

            nn.Conv2d(256, 384, 3, padding=1), # (b x 384 x 13 x 13)
            nn.ReLU(),

            nn.Conv2d(384, 384, 3, padding=1), # (b x 384 x 13 x 13)
            nn.ReLU(),

            nn.Conv2d(384, 256, 3, padding=1), # (b x 256 x 13 x 13)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2) # (b x 256 x 6 x 6)
        )

        # classifier is just a name for linear layers : 3 FC layers
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            # sequential model 로 layer 를 쌓아가고 있고, 따로 클래스로 할당하지 않았으므로, inplace=True 를 해준다.
            nn.Linear(in_features=(256*6*6), out_features=4096),
            nn.ReLU(),

            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),

            nn.Linear(in_features=4096, out_features=1000)
        )

        self.init_bias() # initialize bias layer : 따로 torch 에 구성된 것이 없으므로, 직접 아래에 생성해준다.

    def init_bias(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv2d): # layer 가 nn.Conv2d instance 일 경우,
                nn.init.normal_(layer.weight, mean=0, std=0.01) # mean, std 값으로 정규화 시키기
                nn.init.constant_(layer.bias, 0) # 0으로 대체하기
                # weight 는 정규분포로 초기화, bias 는 0 으로 초기화
                # Conv2d 일 때마다 weight, bias 초기화 시키기

        # original paper = 1 for Conv2d layers 2nd, 4th, and 5th conv layers
        nn.init.constant_(self.net[4].bias, 1)
        nn.init.constant_(self.net[10].bias, 1)
        nn.init.constant_(self.net[12].bias, 1)

    def forward(self, x):
        """
        Pass the input through the net.

        :param x(Tensor) : input tensor
        :return:
                output (Tensor) : output tensor
        """
        x = self.net(x)
        x = x.view(-1, 256 * 6 * 6) # reduce the dimensions for lienar layer input (like Flatten)
        return self.classifier(x)

if __name__ == '__main__':
    # print the seed value
    seed = torch.initial_seed()
    print(f'Used seed : {seed}')

    tbwriter = SummaryWriter(log_dir=LOG_DIR)
    print('TensorboardX summary writer created')


    # 1. create model
    alexnet = AlexNet(num_classes=NUM_CLASSES).to(device)
    # train on multiple GPUs
    alexnet = torch.nn.parallel.DataParallel(alexnet, device_ids=DEVICE_IDS)
    print(alexnet)
    print('Alexnet created')


    # 2. create dataset and data loader
    dataset = datasets.ImageFolder(TRAIN_IMG_DIR, transforms.Compose([
        # transforms.RandomResizedCrop(IMAGE_DIM, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
        transforms.CenterCrop(IMAGE_DIM),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ]))
    print('Dataset created')

    dataloader = data.DataLoader(
        dataset,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        drop_last=True,
        batch_size=BATCH_SIZE
    )
    print('Dataloader created')


    # 3. create optimizer
    # the one that WORKS
    optimizer = optim.Adam(params=alexnet.parameters(), lr = 0.0001)
    ### BELOW is the setting proposed by the original paper = which doesn't train...
    # optimizer = optim.SGD(
    #     params=alexnet.parameters(),
    #     lr=LR_INIT,
    #     momentum=MOMENTUM,
    #     weight_decay=LR_DECAY
    # )
    print('Optimizer created')

    # multiply LR by 1/10 after every 30 epochs
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    print('LR Scheduler created')


    # 4. start training
    print('Starting training...')
    total_steps = 1
    for epoch in range(NUM_EPOCHS):
        lr_scheduler.step()
        for imgs, classes in dataloader:
            imgs, classes = imgs.to(device), classes.to(device)

            # calculate the loss
            output = alexnet(imgs)
            loss = F.cross_entropy(output, classes)

            # updata the parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log the information and add to tensorboard
            if total_steps % 10 == 0:
                with torch.no_grad():
                    _, preds = torch.max(output, dim=1) # return (values, indices)
                    accuracy = torch.sum(preds == classes)

                    print(f'Epoch: {epoch+1} \tStep: {total_steps} \tLoss: {loss.item():.4f} \tAcc: {accuracy.item()}')
                    tbwriter.add_scalar('loss', loss.item(), total_steps)
                    tbwriter.add_scalar('accuracy', accuracy.item(), total_steps)

            # print out gradient values and parameter average values
            if total_steps % 100 == 0:
                with torch.no_grad():
                    # print and save the grad of the parameters
                    # also print and save parameter values
                    print('*' * 10)
                    for name, parameter in alexnet.named_parameters():
                        if parameter.grad is not None:
                            avg_grad = torch.mean(parameter.grad)
                            print('\t{} - grad_avg: {}'.format(name, avg_grad))
                            tbwriter.add_scalar('grad_avg/{}'.format(name), avg_grad.item(), total_steps)
                            tbwriter.add_histogram('grad/{}'.format(name),
                                                   parameter.grad.cpu().numpy(), total_steps)
                        if parameter.data is not None:
                            avg_weight = torch.mean(parameter.data)
                            print('\t{} - param_avg: {}'.format(name, avg_weight))
                            tbwriter.add_histogram('weight/{}'.format(name),
                                                   parameter.data.cpu().numpy(), total_steps)
                            tbwriter.add_scalar('weight_avg/{}'.format(name), avg_weight.item(), total_steps)
            total_steps += 1

        # save checkpoints
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'alexnet_states_e{epoch + 1}.pkl')
        state = {
            'epoch' : epoch,
            'total_steps' : total_steps,
            'optimizer' : optimizer.state_dict(),
            'model' : alexnet.state_dict(),
            'seed' : seed,
        }
        torch.save(state, checkpoint_path)





