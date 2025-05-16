import torch.nn.functional as F
import torch.nn as nn
from resnet9 import ResNet9, ResNet9_tinyimagenet
from resnet import resnet18 
from resnet_cifar import ResNet18 as resnet18_cifar


def get_model(data='cifar10', model='resnet9'):
    if data == 'fmnist' or data == 'fedemnist':
        if model == 'CNN':
            return CNN_MNIST()
        elif model == 'CNN_imp':
            return CNN_MNIST_imp()
        elif model == "CNN_imp_v2":
            return CNN_MNIST_imp_v2()
    elif data == 'cifar10' :
        if model == 'resnet9':
            resnet = ResNet9(3,num_classes=10)
        elif model == 'resnet18':
            resnet = resnet18(num_classes=10)
        elif model == 'resnet_ci':
            resnet = resnet18_cifar()
        elif model == 'CNN_cif':
            return CNN_CIFAR()
        # for name,param in resnet.named_parameters():
        #     logging.info(name)
        return resnet
    elif data == 'GTSRB':
        if model == 'resnet_ci':
            return resnet18_cifar(num_classes=43)
        elif model == "CNN_cif":
            return CNN_CIFAR(num_classes=43)
    elif data == 'cifar100':
        resnet = resnet18_cifar(num_classes=100)
        return resnet
        # return CNN_CIFAR()
    elif data == 'tinyimagenet':
        resnet = ResNet9_tinyimagenet(3,num_classes=200)
        return resnet
        # return SimpleCNNTinyImagenet(input_dim=(16 * 3 * 13 * 13), hidden_dims=[120*3, 84*3],
        #                                       output_dim=200)

    elif data == 'mnist':
        mlp = MLP(num_classes=10)
        return mlp


class MLP(nn.Module):
    def __init__(self, num_classes=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 100, bias=False)
        self.fc2 = nn.Linear(100, num_classes, bias=False)

    def forward(self, x):
        x= x.view(-1,28*28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SimpleCNNTinyImagenet(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNNTinyImagenet, self).__init__()
        self.conv1 = nn.Conv2d(3, 18, 5)
        self.bn1 = nn.BatchNorm2d(18)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(18, 48, 5)
        self.bn2 = nn.BatchNorm2d(48)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.linear = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 16 * 3 * 13 * 13)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.linear(x)
        return x

class CNN_MNIST(nn.Module):
    def __init__(self):
        super(CNN_MNIST, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.norm1 = nn.BatchNorm2d(32)
        nn.init.xavier_uniform(self.cnn1.weight)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.cnn2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.norm2 = nn.BatchNorm2d(64)
        nn.init.xavier_uniform(self.cnn2.weight)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(4096, 256)
        self.fcrelu = nn.ReLU()

        self.fc2 = nn.Linear(256, 10)

    def copy_params(self, state_dict, coefficient_transfer=100):
        own_state = self.state_dict()

        for name, param in state_dict.items():
            if name in own_state:
                shape = param.shape
                own_state[name].copy_(param.clone())

    def forward(self, x):
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.norm1(out)

        out = self.maxpool1(out)

        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.norm2(out)

        out = self.maxpool2(out)

        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        out = self.fcrelu(out)

        out = self.fc2(out)
        return out
    
    def get_features(self, x):
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.norm1(out)

        out = self.maxpool1(out)

        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.norm2(out)

        out = self.maxpool2(out)

        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        out = self.fcrelu(out)
        
        return out





class CNN_CIFAR(nn.Module):
    def __init__(self, num_classes = 10):
        super(CNN_CIFAR, self).__init__()
        self.conv1 = nn.Conv2d(3,   64,  3,bias=False)
        self.bn1 = nn.BatchNorm2d(64,track_running_stats=False)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64,  128, 3,bias=False)
        self.bn2 = nn.BatchNorm2d(128,track_running_stats=False)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(128, 256, 3,bias=False)
        self.bn3 = nn.BatchNorm2d(256,track_running_stats=False)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(64 * 4 * 4, 128,bias=False)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc2 = nn.Linear(128, 256,bias=False)
        self.drop3 = nn.Dropout2d(p=0.5)
        self.fc3 = nn.Linear(256, num_classes, bias=False)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 64 * 4 * 4)
        # x = self.drop1(x)
        x = F.relu(self.fc1(x))
        # x = self.drop2(x)
        x = F.relu(self.fc2(x))
        # x = self.drop3(x)
        x = self.fc3(x)
        return x
    
    def get_features(self, x) :
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 64 * 4 * 4)
        # x = self.drop1(x)
        x = F.relu(self.fc1(x))
        # x = self.drop2(x)
        return x

class CNN_MNIST_imp(nn.Module):
    def __init__(self):
        super(CNN_MNIST_imp, self).__init__()

        self.cnn1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)     # 28x28 → 28x28
        self.norm1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)                               # → 14x14

        self.cnn2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)    # 14x14 → 14x14
        self.norm2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)                               # → 7x7

        self.cnn3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)    # 7x7 → 7x7
        self.norm3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)                               # → 3x3

        self.flatten = nn.Flatten()                                # → 64 * 3 * 3 = 576
        self.fc = nn.Linear(576, 256)
        self.relu_fc = nn.ReLU()

        self.classifier = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.norm1(self.cnn1(x))))
        x = self.pool2(self.relu2(self.norm2(self.cnn2(x))))
        x = self.pool3(self.relu3(self.norm3(self.cnn3(x))))
        x = self.flatten(x)
        x = self.relu_fc(self.fc(x))
        x = self.classifier(x)
        return x

    def get_features(self, x):
        x = self.pool1(self.relu1(self.norm1(self.cnn1(x))))
        x = self.pool2(self.relu2(self.norm2(self.cnn2(x))))
        x = self.pool3(self.relu3(self.norm3(self.cnn3(x))))
        x = self.flatten(x)
        x = self.relu_fc(self.fc(x))
        return x
    
    def copy_params(self, state_dict, coefficient_transfer=100):
        own_state = self.state_dict()

        for name, param in state_dict.items():
            if name in own_state:
                shape = param.shape
                own_state[name].copy_(param.clone())

class CNN_MNIST_imp_v2(nn.Module):
    def __init__(self):
        super(CNN_MNIST_imp_v2, self).__init__()

        self.cnn1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)     # 28x28 → 28x28
        self.norm1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)                               # → 14x14

        self.cnn2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)    # 14x14 → 14x14
        self.norm2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)                               # → 7x7

        self.cnn3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)    # 7x7 → 7x7
        self.norm3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)                               # → 3x3

        self.flatten = nn.Flatten()                                # → 64 * 3 * 3 = 576
        self.fc = nn.Linear(576, 256)
        self.norm4 = nn.BatchNorm1d(256)
        self.relu_fc = nn.ReLU()

        self.classifier = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.norm1(self.cnn1(x))))
        x = self.pool2(self.relu2(self.norm2(self.cnn2(x))))
        x = self.pool3(self.relu3(self.norm3(self.cnn3(x))))
        x = self.flatten(x)
        x = self.relu_fc(self.norm4(self.fc(x)))
        x = self.classifier(x)
        return x

    def get_features(self, x):
        x = self.pool1(self.relu1(self.norm1(self.cnn1(x))))
        x = self.pool2(self.relu2(self.norm2(self.cnn2(x))))
        x = self.pool3(self.relu3(self.norm3(self.cnn3(x))))
        x = self.flatten(x)
        x = self.relu_fc(self.norm4(self.fc(x)))
        return x
    
    def copy_params(self, state_dict, coefficient_transfer=100):
        own_state = self.state_dict()

        for name, param in state_dict.items():
            if name in own_state:
                shape = param.shape
                own_state[name].copy_(param.clone())