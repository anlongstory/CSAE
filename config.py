import os

''' Hyperparameter '''

latent_variable_dim = 40
BatchSize = 4096
base_lr = 0.1
epoches = 120
lr_step = 30

''' model '''

model_path = r'./model/MNIST'
mean_var_path = r'./mean_var_pkl'
reconstruction_path = r"./reconstruction/MNIST"
recon_flag = True
generate_samples_path = r"./generate_samples/MNIST"
generate_flag = True

'''   PEDCC setting '''

class_num=10
PEDCC_root=r"./ui_pkl/"
PEDCC_ui=os.path.join(PEDCC_root,str(class_num)+"_"+str(latent_variable_dim)+".pkl")

