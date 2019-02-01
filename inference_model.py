
from utils import read_pkl,make_dir
import matplotlib.pyplot as plt
from data_transform import *
import pickle,torch
import numpy as np
import cv2
from config import *

make_dir(reconstruction_path)
make_dir(generate_samples_path)
make_dir(mean_var_path)

def gen_data(mean,cov,num):
    mean=mean.cpu()
    data = np.random.multivariate_normal(mean,cov,num)
    return np.round(data,4)


map_dict = read_pkl()
mean=map_dict[0].float()

num = 40
net1 = torch.load(os.path.join(model_path,'encoder_sigma_'+str(num)+'.pth'))
net2 = torch.load(os.path.join(model_path,'decoder_sigma_'+str(num)+'.pth'))

for i in net1.parameters():
    i.requires_grad=False

for i in net2.parameters():
    i.requires_grad=False

net1 = net1.cuda()
net2 = net2.cuda()

net1 = net1.eval()
net2 = net2.eval()

def generate_mix_samples(net1,net2):
    subset=os.path.join(mean_var_path,str(num)+'_mean_var.pkl')
    if os.path.exists(subset):
        have_pkl=True
    else:
        have_pkl=False

    if not have_pkl:
        map_mean_var = {}
        train_root=r"./data/MNIST_img/train"

        '''
        the images are arranged in this way: ::

        root/0/0/xxx.png
        root/0/0/xxy.png
        root/0/0/xxz.png

        root/1/1/123.png
        root/1/1/nsdf3.png
        root/1/1/asd932_.png

        '''
        folder_list = os.listdir(train_root)
        for i in sorted(folder_list):
            train_folder = os.path.join(train_root,i)
            data_set = ImageFolder_L(train_folder, transform=data_transform)
            data_data = DataLoader(data_set, batch_size=BatchSize, shuffle=False)
            out_all=torch.Tensor([]).cuda()
            out_sum = torch.Tensor([0]*latent_variable_dim).cuda()
            with torch.no_grad():
                for im,label in data_data :
                    im = im.cuda()
                    out = net1(im)
                    out_all = torch.cat((out_all,out),0)
                    for ii in out:
                        out_sum+=ii
            print(i)
            mean_out = out_sum / out_all.size(0)
            cov =  np.cov(torch.t(out_all).cpu().numpy())
            map_mean_var[mean_out] = cov

        f=open(subset,'wb')
        pickle.dump(map_mean_var,f)
        f.close()

    f=open(subset,'rb')
    mean_var=pickle.load(f)
    f.close()

    data_num = 10
    data=[]
    cnt=0
    for keys,values in mean_var.items():
        if  cnt == 0:
            data = gen_data(keys,values,data_num)
            cnt+=1
        else:
            data = np.concatenate([data,gen_data(keys,values,data_num)],0)
            cnt+=1

    cnt2=0
    image_all = np.zeros([32*int((len(data)**0.5)),32*(int(len(data)**0.5))])
    with torch.no_grad():
        for j in range(int(len(data) ** 0.5)):
            for q in range(int(len(data) ** 0.5)):
                input1=torch.from_numpy(np.array([data[cnt2]])).float()
                input1 = input1.cuda()
                out2 = net2(input1)
                ii = inver_transform_MNIST(out2)
                ii[0][0][ii[0][0]<0]=0
                ii[0][0][ii[0][0]> 255] = 255
                image_all[q*32:(q+1)*32,j*32:(j+1)*32] = ii[0][0]
                cnt2+=1
        cv2.imwrite(os.path.join(generate_samples_path, str(num) + '_generate.png'), image_all)
        plt.imshow(image_all,cmap='Greys_r')
        plt.axis('off')
        plt.show()


def reconstruction(model1, model2, batch_num, data):
    for i, (data, labels) in enumerate(data):
        data = data.cuda()
        recon = model1(data)
        recon_batch = model2(recon)
        n = min(data.size(0), 8)
        empty_img = np.zeros((2*32,32*n))
        for i in range(n):
            empty_img[0:32,i*32:(i+1)*32] = inver_transform_MNIST(data[i])
        for j in range(n):
            empty_img[32:64,j*32:(j+1)*32] = inver_transform_MNIST(recon_batch[j])

        cv2.imwrite(os.path.join(reconstruction_path, str(num) + '_reconstruction_' + str(batch_num) + '.png'),empty_img)

if generate_flag:
    generate_mix_samples(net1,net2)

if recon_flag:
    MNIST_test_data_shuffle=DataLoader(MNIST_test_set,batch_size = BatchSize,shuffle = True)
    for i in range(len(MNIST_test_data_shuffle)):
        reconstruction(net1,net2,i,MNIST_test_data_shuffle)
