import scipy.io as sio
import scipy.stats as st
from sklearn import linear_model
import scipy.misc
import numpy as np
import math
from matplotlib import pyplot as plt

dimen=100  #原子数
iter=30    #迭代数上限
tol=1e-6   #容差
loads='train_data.mat'
load_data=sio.loadmat(loads)
load_matrix=load_data['Data']
a=load_matrix[2].reshape(3,32,32)
print(a.shape)
#plt.imshow(a.T)
#plt.show()
'''
load_label=load_data['Label']
R0=np.zeros([1000,1024])
G0=np.zeros([1000,1024])
B0=np.zeros([1000,1024])
R1=np.zeros([1024,1000])
G1=np.zeros([1024,1000])
B1=np.zeros([1024,1000])
for i in range(1000):
    for j in range(1024):
        R0[i][j]=load_matrix[i][j]
        G0[i][j]=load_matrix[i][j+1024]
        B0[i][j]=load_matrix[i][j+2048]
#图像SVD分解
u0,sigma0,v0=np.linalg.svd(R0)
u1,sigma1,v1=np.linalg.svd(G0)
u2,sigma2,v2=np.linalg.svd(B0)
#SVD取前dimne维重建图像
RS=np.zeros(R0.shape)
GS=np.zeros(G0.shape)
BS=np.zeros(B0.shape)
for i in range(dimen):
    RS+=sigma0[i]*np.dot(u0[:,i].reshape(-1,1),v0[i,:].reshape(1,-1))
for i in range(dimen):
    GS+=sigma1[i]*np.dot(u1[:,i].reshape(-1,1),v1[i,:].reshape(1,-1))
for i in range(dimen):
    BS+=sigma2[i]*np.dot(u2[:,i].reshape(-1,1),v2[i,:].reshape(1,-1))
#计算总的均方误差
a=0
b=0
c=0
for i in range(1000):
    for j in range(1024):
        a=a+(RS[i][j]-R0[i][j])*(RS[i][j]-R0[i][j])
        b=b+(GS[i][j]-G0[i][j])*(GS[i][j]-G0[i][j])
        c=c+(BS[i][j]-B0[i][j])*(BS[i][j]-B0[i][j])
print(a+b+c)
#字典学习K-SVD
R1=R0.transpose()
G1=R0.transpose()  
B1=R0.transpose() 
#迭代kvsd初始化
ur,sr,vr=np.linalg.svd(R1)
ug,sg,vg=np.linalg.svd(G1)
ub,sb,vb=np.linalg.svd(B1)
dicr=ur[:,:dimen]
dicg=ug[:,:dimen]
dicb=ub[:,:dimen]
nonzero_coe=None
#r做ksvd
for i in range (iter):
    xr=linear_model.orthogonal_mp(dicr,R1,n_nonzero_coefs=nonzero_coe)
    er=np.linalg.norm(R1-np.dot(dicr,xr))
    if er<tol:
        break
    #逐列更新字典，并更新非零编码
    for i in range(dimen):
        indexr=np.nonzero(xr[i,:])[0]
        if len(indexr)==0:
            continue
        dicr[:,i]=0
        rr=(R1-np.dot(dicr,xr))[:,indexr]
        ur,sr,vr=np.linalg.svd(rr,full_matrices=False)
        dicr[:,i]=ur[:,0].T
        xr[i,indexr]=sr[0]*vr[0,:]       
sparsecoder=linear_model.orthogonal_mp(dicr,R1,n_nonzero_coefs=nonzero_coe)
#g做ksvd
for i in range (iter):
    xg=linear_model.orthogonal_mp(dicg,G1,n_nonzero_coefs=nonzero_coe)
    eg=np.linalg.norm(G1-np.dot(dicg,xg))
    if eg<tol:
        break
    #更新字典
    for i in range(dimen):
        indexg=np.nonzero(xg[i,:])[0]
        if len(indexg)==0:
            continue
        dicg[:,i]=0
        rg=(G1-np.dot(dicg,xg))[:,indexg]
        ug,sg,vg=np.linalg.svd(rg,full_matrices=False)
        dicg[:,i]=ug[:,0].T
        xg[i,indexg]=sg[0]*vg[0,:]       
sparsecodeg=linear_model.orthogonal_mp(dicg,G1,n_nonzero_coefs=nonzero_coe)
#b做ksvd
for i in range (iter):
    xb=linear_model.orthogonal_mp(dicb,B1,n_nonzero_coefs=nonzero_coe)
    eb=np.linalg.norm(B1-np.dot(dicb,xb))
    if eb<tol:
        break
    #更新字典
    for i in range(dimen):
        indexb=np.nonzero(xb[i,:])[0]
        if len(indexb)==0:
            continue
        dicb[:,i]=0
        rb=(B1-np.dot(dicb,xb))[:,indexb]
        ub,sb,vb=np.linalg.svd(rb,full_matrices=False)
        dicb[:,i]=ub[:,0].T
        xb[i,indexb]=sb[0]*vb[0,:]       
sparsecodeb=linear_model.orthogonal_mp(dicb,B1,n_nonzero_coefs=nonzero_coe)
#计算字典学习的均方误差
a=0
b=0
c=0
R2=dicr.dot(sparsecoder)
G2=dicg.dot(sparsecodeg)
B2=dicb.dot(sparsecodeb)
for i in range(1024):
    for j in range(1000):
        a+=(R1[i][j]-round(R2[i][j]))*(R1[i][j]-round(R2[i][j]))
        b+=(G1[i][j]-round(G2[i][j]))*(G1[i][j]-round(G2[i][j]))
        c+=(B1[i][j]-round(B2[i][j]))*(B1[i][j]-round(B2[i][j]))
print(a+b+c)
'''