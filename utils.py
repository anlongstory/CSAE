from datetime import datetime
import torch,pickle,math
from tqdm import tqdm
import numpy as np
import os
import torch.nn.functional as F
from config import latent_variable_dim,PEDCC_ui,model_path,epoches


device_ids = [i for i in range(torch.cuda.device_count())]



def make_dir(path):
    tmp_list=os.path.split(path)
    subset=""
    for i in tmp_list:
        subset=os.path.join(subset,i)
        if not os.path.isdir(subset):
            os.makedirs(subset)

make_dir(model_path)

def gen_data(mean,cov,num):
    data = np.random.multivariate_normal(mean,cov,num)
    return np.round(data,4)

def generator_noise(batchsize,out_dim):
    mean = np.zeros(out_dim)
    cov = np.eye(out_dim)
    noise = np.random.multivariate_normal(mean,cov,batchsize)
    return noise


def get_cc_ic(output, label, ui):
    total = output.shape[0]

    total_distance = list()
    for i in range(total):
        distance_list = list()
        for ui_label in ui.values():
            distance = sum((ui_label[0].float().cuda()-output[0][i])**2)
            distance_list.append(distance.item())
        idx = distance_list.index(min(distance_list))
        total_distance.append(idx)
    pred_label = torch.Tensor(total_distance).long().cuda()

    num_correct = (pred_label == label).sum().item()
    return num_correct / total

def read_pkl():
    f = open(PEDCC_ui,'rb')
    a = pickle.load(f)
    f.close()
    return a


def sobel(im):
    weight_x = np.array([[[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]]])
    weight_y = np.array([[[[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]]])

    weight_x = torch.from_numpy(weight_x).float().cuda()
    weight_y = torch.from_numpy(weight_y).float().cuda()

    sobel_x = F.conv2d(im,weight=weight_x,stride=1,padding=1)
    sobel_y = F.conv2d(im,weight=weight_y,stride=1,padding=1)

    return sobel_x+sobel_y

def train_en_de_C(net1,net2, train_data, valid_data, epoch, optimizer_en,optimizer_de, criterion):
    '''
    For CSAE-C training

    '''
    map_dict = read_pkl()
    if torch.cuda.is_available():
        net1 =  torch.nn.DataParallel(net1, device_ids=device_ids)
        net2 =  torch.nn.DataParallel(net2, device_ids=device_ids)
        net1 = net1.cuda()
        net2 = net2.cuda()
    prev_time = datetime.now()

    train_loss = 0
    train_acc = 0
    train_loss1 = 0
    train_loss2 = 0
    train_loss3 = 0
    net1 = net1.train()
    net2 = net2.train()
    for im, label in tqdm(train_data,desc="Processing train data: "):
        if torch.cuda.is_available():
            im = im.cuda()
            label = label.cuda()
            tensor_empty = torch.Tensor([]).cuda()
            for label_index in label:
                tensor_empty = torch.cat((tensor_empty, map_dict[label_index.item()].float().cuda()), 0)

            label_tensor = tensor_empty.view(-1, latent_variable_dim)
            label_tensor = label_tensor.cuda()

        # forward
        output_classifier=net1(im)
        loss1 = criterion(output_classifier, label_tensor)

        sigma = generator_noise(output_classifier.size(0),output_classifier.size(1))
        new_out = output_classifier + torch.from_numpy(sigma*0.04*(output_classifier.size(1)**0.5)).float().cuda()
        output_deconv = net2(new_out)
        loss2 = criterion(output_deconv,im)

        sobel_im = sobel(im)
        sobel_deconv = sobel(output_deconv)

        zeros = np.zeros([sobel_im.size(0), sobel_im.size(1), sobel_im.size(2), sobel_im.size(3)])
        loss3_1 =  criterion(sobel_im, torch.from_numpy(zeros).float().cuda())
        loss3_2 =  criterion(sobel_deconv, torch.from_numpy(zeros).float().cuda())
        loss3 = 0.02*torch.abs(loss3_1 - loss3_2)

        loss = loss1 + loss2 + loss3

        optimizer_en.zero_grad()
        optimizer_de.zero_grad()
        loss.backward()
        optimizer_en.step()
        optimizer_de.step()
        train_loss  +=  loss.item()
        train_loss1 += loss1.item()
        train_loss2 += loss2.item()
        train_loss3 += loss3.item()

        if (epoch % epoches == 0):
            train_acc += get_cc_ic(output_classifier, label, ui=map_dict)

    curr_time = datetime.now()
    h, remainder = divmod((curr_time - prev_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    time_str = " Time %02d:%02d:%02d" % (h, m, s)

    if valid_data is not None:
        valid_loss = 0
        valid_acc = 0
        val_loss1 = 0
        val_loss2 = 0
        val_loss3 = 0
        net1 = net1.eval()
        net2 = net2.eval()
        for im, label in tqdm(valid_data,desc="Processing val data: "):

            if torch.cuda.is_available():
                im = im.cuda()
                label = label.cuda()
                tensor_empty_test = torch.Tensor([]).cuda()
                for label_index in label:
                    tensor_empty_test = torch.cat((tensor_empty_test, map_dict[label_index.item()].float().cuda()), 0)

                label_tensor_test = tensor_empty_test.view(-1, latent_variable_dim)
                label_tensor_test = label_tensor_test.cuda()
            output1 = net1(im)
            loss1 = criterion(output1, label_tensor_test)

            output2 = net2(output1)
            loss2 = criterion(output2, im)


            sobel_im = sobel(im)
            sobel_deconv = sobel(output2)
            zeros = np.zeros([sobel_im.size(0), sobel_im.size(1), sobel_im.size(2), sobel_im.size(3)])
            loss3_1 = criterion(sobel_im, torch.from_numpy(zeros).float().cuda())
            loss3_2 = criterion(sobel_deconv, torch.from_numpy(zeros).float().cuda())
            loss3 = 0.02*torch.abs(loss3_1 - loss3_2)


            loss = loss1 + loss2 + loss3
            valid_loss += loss.item()
            val_loss1 += loss1.item()
            val_loss2 += loss2.item()
            val_loss3 += loss3.item()
            if (epoch % epoches == 0):
                valid_acc += get_cc_ic(output1, label, ui=map_dict)
        epoch_str = ("Epoch %d. Train Loss: %f, Train.Acc: %f, Valid Loss: %f, Valid Acc: %f,"
                     % (epoch, train_loss / len(train_data), train_acc / len(train_data),
                        valid_loss / len(valid_data), valid_acc / len(valid_data)))
        Loss = ("Train Loss1: %f, Train Loss2: %f, Train Loss3: %f,Val_Loss1: %f, Val_Loss2: %f, Val_Loss3: %f"
                %(train_loss1/len(train_data),train_loss2/len(train_data),train_loss3/len(train_data),
                  val_loss1/len(valid_data),val_loss2/len(valid_data),val_loss3/len(valid_data)))
    else:
        epoch_str = ("Epoch %d. Train Loss: %f, Train.Acc: %f,  "
                     % (epoch, train_loss / len(train_data), train_acc / len(train_data)))
        Loss = ("Train Loss1: %f, Train Loss2: %f,Train Loss3: %f,"
                % (train_loss1 / len(train_data), train_loss2 / len(train_data),train_loss3 / len(train_data)))

    prev_time = curr_time
    if epoch % 20 == 0:
        torch.save(net1, os.path.join(model_path,'encoder_sigma_' + str(epoch) + '.pth'))
        torch.save(net2,os.path.join(model_path,'decoder_sigma_' + str(epoch) + '.pth'))
    f = open(os.path.join(model_path,'en_de.txt'), 'a+')
    print(" ")
    print(epoch_str + time_str)
    print(Loss+"---------------")
    f.write(epoch_str + time_str + '\n')
    f.write(Loss+'\n')
    f.close()