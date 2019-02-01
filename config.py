import os

''' Hyperparameter '''

latent_variable_dim = 40  # The output dimension of encoder, and the dimention of predefined class centriods
BatchSize = 4096  
base_lr = 0.1  
epoches = 120  # max epochs
lr_step = 30   

''' model '''

model_path = r'./model/MNIST'    # path to save model
mean_var_path = r'./mean_var_pkl' # path to save mean and covariance of eacg class
reconstruction_path = r"./reconstruction/MNIST"  # path to save reconstruction results
recon_flag = True     #  whether test reconstruction or not
generate_samples_path = r"./generate_samples/MNIST"  # path to save generate samples
generate_flag = True  #   whether generate samples or not

'''   PEDCC setting '''

class_num=10  # The number of class centroids
PEDCC_root=r"./ui_pkl/"
PEDCC_ui=os.path.join(PEDCC_root,str(class_num)+"_"+str(latent_variable_dim)+".pkl")

