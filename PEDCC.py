import numpy as np
import pickle
import torch
import os
from config import class_num,latent_variable_dim,PEDCC_root,PEDCC_ui

if not os.path.isdir(PEDCC_root):
    os.makedirs(PEDCC_root)

mu=np.zeros(latent_variable_dim)
sigma = np.eye(latent_variable_dim)

u_init = np.random.multivariate_normal(mu,sigma,class_num)
v = np.zeros(u_init.shape)
u=[]
for i in u_init:
    i=i/np.linalg.norm(i)
    u.append(i)
u=np.array(u)
G=1e-2

def countnext(u,v,G):
    num = u.shape[0]
    dd = np.zeros((latent_variable_dim,num,num))
    for m in range(num):
        for n in range(num):
            dd[:,m,n] = (u[m,:]-u[n,:]).T
            dd[:,n,m] = -dd[:,m,n]
    L=np.sum(dd**2,0)**0.5
    L=L.reshape(1,L.shape[0],L.shape[1])
    L[L<1e-2] = 1e-2
    a=np.repeat(L**3,latent_variable_dim,0)
    F=np.sum(dd/a,2).T
    tmp_F=[]
    for i in range(F.shape[0]):
        tmp_F.append(np.dot(F[i],u[i]))
    d=np.array(tmp_F).T.reshape(len(tmp_F),1)
    Fr = u*np.repeat(d, latent_variable_dim, 1)
    Ft = F-Fr
    un = u+v
    ll = np.sum(un**2,1)**0.5
    un=un/np.repeat(ll.reshape(ll.shape[0],1),latent_variable_dim,1)
    vn = v+G*Ft
    return un,vn

def generate_center(u,v,G):
    for i in range(200):
        un,vn=countnext(u,v,G)
        u=un
        v=vn
    return u*(latent_variable_dim)**0.5
    # return r

r1=generate_center(u,v,G)

f=open('./tmp.pkl','wb')
pickle.dump(r1,f)
f.close()

ff=open("./tmp.pkl",'rb')
b=pickle.load(ff)
ff.close()

os.remove("./tmp.pkl")

#### distance matrix
# result=np.zeros((len(b),len(b)))
# for a in range(len(b)):
#     for aa in range(a,len(b)):
#         result[a][aa]=np.linalg.norm((b[a]-b[aa]))
#         result[aa][a] = result[a][aa]

fff=open(PEDCC_ui,'wb')
map={}
for i in range(len(b)):
    map[i]=torch.from_numpy(np.array([b[i]]))
pickle.dump(map,fff)
fff.close()







