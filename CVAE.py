import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
from data_transform import *
device_ids = [0,1,2,3]

use_cuda = True
batch_size = 512
latent_size = 40 # z dim

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/fashionMNIST', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/fashionMNIST', train=False,
                       transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True, **kwargs)

def to_var(x):
    x = Variable(x)
    if use_cuda:
        x = x.cuda()
    return x

def one_hot(labels, class_size):
    targets = torch.zeros(labels.size(0), class_size)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return to_var(targets)

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 28*28))
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD



class CVAE(nn.Module):
    def __init__(self, feature_size, latent_size, class_size):
        super(CVAE, self).__init__()
        self.feature_size = feature_size
        self.class_size = class_size
        self.training=True

        # encode
        self.fc1  = nn.Linear(feature_size + class_size, 400)
        self.fc21 = nn.Linear(400, latent_size)
        self.fc22 = nn.Linear(400, latent_size)

        # decode
        self.fc3 = nn.Linear(latent_size + class_size, 400)
        self.fc4 = nn.Linear(400, feature_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x, c): # Q(z|x, c)
        '''
        x: (bs, feature_size)
        c: (bs, class_size)
        '''
        inputs = torch.cat([x, c], 1) # (bs, feature_size+class_size)
        h1 = self.relu(self.fc1(inputs))
        z_mu = self.fc21(h1)
        z_var = self.fc22(h1)
        return z_mu, z_var

    def reparametrize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std) + mu
        else:
            return mu

    def decode(self, z, c): # P(x|z, c)
        '''
        z: (bs, latent_size)
        c: (bs, class_size)
        '''
        inputs = torch.cat([z, c], 1) # (bs, latent_size+class_size)
        h3 = self.relu(self.fc3(inputs))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x, c):
        mu, logvar = self.encode(x.view(-1, 28*28), c)
        z = self.reparametrize(mu, logvar)
        return self.decode(z, c), mu, logvar


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = to_var(data)
        labels = one_hot(labels, 10)
        recon_batch, mu, logvar = model(data, labels)
        optimizer.zero_grad()
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if epoch % 20 == 0 and epoch!=0:
            torch.save(model,'./model/fashionMNIST/CVAE/cvae'+str(epoch)+".pth")
        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))


def test2(model1,epoch,data):
    for i, (data, labels) in enumerate(data):
        data = data.cuda()
        # labels = one_hot(labels, 10)
        recon_batch,mena,var = model1(data)
        if i == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n],
                                  recon_batch.view(BatchSize, 1, 28, 28)[:n]])
            save_image(comparison.data.cpu(),
                     './results/CVAE/letter/120/reconstruction_' + str(epoch) + '.png', nrow=n)


def test(model,epoch):
    model.cuda()
    model.eval()
    test_loss = 0
    for i, (data, labels) in enumerate(test_loader):
        data = to_var(data)
        labels = one_hot(labels, 10)
        recon_batch, mu, logvar = model(data, labels)
        # test_loss += loss_function(recon_batch, data, mu, logvar).item()
        if i == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n],
                                  recon_batch.view(batch_size, 1, 28, 28)[:n]])
            save_image(comparison.data.cpu(),
                     './results/CVAE/fashionMNIST/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))



model = CVAE(28*28, latent_size, 10)
if use_cuda:
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# model = torch.load('./model/MNIST/CVAE/cvae100.pth')
# for epoch in range(1, 201):
#     train(epoch)
#     test(model,epoch)

# Generate images with condition labels

model = torch.load('./model/fashionMNIST/CVAE/cvae200.pth')
# model = torch.nn.DataParallel(model)
if isinstance(model, torch.nn.DataParallel):
    model = model.module

# model.load_state_dict(torch.load('./model/MNIST/CVAE/cvae100.pth'))
model.eval()
model = model.cuda()

samples=torch.Tensor([]).cuda()
for i in range(10):
    c = torch.eye(10, 10) # [one hot labels for 0-9]
    c = to_var(c)
    z = to_var(torch.randn(10, latent_size))
        # samples = model.decode(z, c).data.cpu().numpy()
    samples = torch.cat((model.decode(z, c),samples),0)

samples = samples.data.cpu().numpy()
print(samples.shape)

data_num = samples.shape[0]

cnt2=0
image_all = np.zeros([28*int((data_num**0.5)),28*(int(data_num**0.5))])

for j in range(int(data_num ** 0.5)):
    for q in range(int(data_num ** 0.5)):
        # input1=torch.from_numpy(np.array([samples[cnt2]])).float()
        # input1 = input1.cuda()
        # out2 = net2(input1)
        # ii=out2.view(1,1,28,28).cpu().numpy()
        # ii = inver_transform_MNIST(samples[cnt2].view(1,1,28,28))
        # ii[0][0][ii[0][0]<0]=0
        # ii[0][0][ii[0][0]> 255] = 255
        image_all[j*28:(j+1)*28,q*28:(q+1)*28] = samples[cnt2].reshape(28,28)
        cnt2+=1
# cv2.imwrite('cvae200.png',image_all)
plt.imshow(image_all,cmap='Greys_r')
plt.axis('off')
plt.show()
#
# fig = plt.figure(figsize=(10, 10))
# gs = gridspec.GridSpec(10, 10)
# gs.update(wspace=0.05, hspace=0.05)
# for i, sample in enumerate(samples):
#         ax = plt.subplot(gs[i])
#         plt.axis('off')
#         ax.set_xticklabels([])
#         ax.set_yticklabels([])
#         ax.set_aspect('equal')
#         plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
# plt.show()

