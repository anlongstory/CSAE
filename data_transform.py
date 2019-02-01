################################################
#
#  Author: Chris Zhang
#  Date : 2019/1/10
#  Function: Used to pre-process some data-sets
#
#################################################

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST,FashionMNIST
import torchvision.transforms as transform
import torch.nn.functional as F
from data_read import ImageFolder_L
from config import BatchSize


### invert transform
class DeNormalize(object):
    def __init__(self,mean,std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t,m,s in zip(tensor,self.mean,self.std):
            t.mul_(s).add_(m)
        return tensor


class Normalize(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return F.normalize(tensor, self.mean, self.std)


############## MNIST

data_transform = transform.Compose([
    transform.Pad(padding=2,fill=0),
    transform.ToTensor(),
    transform.Normalize(mean=[0.5],std=[0.5])
])

inver_transform_MNIST = transform.Compose([
    DeNormalize([0.5],[0.5]),
    lambda x: x.cpu().numpy()*255.,
])
############## Read from Standard API

MNIST_root = r"./data/mnist"
MNIST_train_set = MNIST(MNIST_root,train=True,transform=data_transform,download=True)
MNIST_test_set = MNIST(MNIST_root,train=False,transform=data_transform)

MNIST_train_data = DataLoader(MNIST_train_set,batch_size = BatchSize,shuffle = True)
MNIST_test_data =  DataLoader(MNIST_test_set,batch_size = BatchSize,shuffle = False)

############## Fashion-MNIST


fashionMNIST_root=r"./data/fashionMNIST"

fashionMNIST_train_set=FashionMNIST(fashionMNIST_root,train=True,transform=data_transform,download=True)
fashionMNIST_test_set=FashionMNIST(fashionMNIST_root,train=False,transform=data_transform)

fashionMNIST_train_data=DataLoader(fashionMNIST_train_set,batch_size=BatchSize,shuffle=True)
fashionMNIST_test_data = DataLoader(fashionMNIST_test_set,batch_size=BatchSize,shuffle=True)



############## Subset of EMNIST

LETTER_root = r"./data/letter"
L_test = ImageFolder_L(LETTER_root,transform=data_transform)
letter_test_data = DataLoader(L_test,batch_size=BatchSize,shuffle=True)

