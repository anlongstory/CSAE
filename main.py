from two_stage_model import *
from data_transform import *
import torch
from utils import train_en_de_C
from config import base_lr,epoches,lr_step,latent_variable_dim


net1 = encoder_C(1,latent_variable_dim)
net2 = decoder_C(latent_variable_dim)

optimizer_encoder = torch.optim.SGD(net1.parameters(),lr=base_lr,momentum=0.9,weight_decay=0.0005)
optimizer_decoder = torch.optim.SGD(net2.parameters(),lr=base_lr,momentum=0.9,weight_decay=0.0005)


criterion = nn.MSELoss()

def adjust_lr(optimizer,epoch):
    lr = base_lr*(0.1**(epoch//lr_step))
    for parameter in optimizer.param_groups:
        parameter['lr'] = lr


print(" #### Start training ####")

for epoch in range(1,epoches+1):
    adjust_lr(optimizer_encoder,epoch)
    adjust_lr(optimizer_decoder, epoch)
    train_en_de_C(net1,net2,MNIST_train_data,MNIST_test_data,epoch,optimizer_encoder,optimizer_decoder,criterion)

print("Done!")
