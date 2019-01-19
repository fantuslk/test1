import scipy.io as sio
import scipy.stats as st
import numpy as np
import math
from matplotlib import pyplot as plt

gaosk=8  #高斯核参数
dimen=200 #pca降维后维度
vcount=1000 #训练数据集每个类别样本数
#C=0.6#软间隔
class optStruct:
    def __init__(self,dataMatIn,classLabels,C,toler, kTup=('rbf', gaosk)):
#初始化所以参数
#dataMatIn:训练集矩阵
#labelMat：训练集标签矩阵
#C：惩罚参数
#toler：误差容忍度
#kTup：核函数
        self.X=dataMatIn
        self.labelMat=classLabels
        self.C=C
        self.tol=toler
        #样本数
        self.m=dataMatIn.shape[0]
        #待优化一组alpha
        self.alphas= np.mat(np.zeros((self.m,1)))
        self.b=0
        #第一列有效标志位，第二列误差值
        self.eCache= np.mat(np.zeros((self.m,2)))
        #计算出训练集train_x与每个样本X[i,:]核函数的转换值，并按列存储，方便查询，避免重复计算
        self.K=np.mat(np.zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i]=KernelTrans(self.X,self.X[i,:],kTup)
#计算第k个样本预测误差
def calcEk(os,k):
    fxk=float(np.multiply(os.alphas,os.labelMat).T*os.K[:,k]+os.b)
    Ek=fxk-float(os.labelMat[k])
    return Ek
#控制alpha在L到H间，
def clipAlpha(input,low,high):
    if input>high:
        input = high
    if input<low:
        input = low   
    return input
#选择m范围内部不等于i的整数
def selectJrand(i,m):
    j=i
    while j==i:
        j=int(np.random.uniform(0,m))
    return j
#寻找第二个优化变量，具有最大步长，第一个i对应ei，与外层配对的alpha下标
def selectJ(i,os,Ei):
    maxK=-1
    maxDeltaE=0
    Ej=0
    os.eCache[i]=[1,Ei]
    validEcacheList=np.nonzero(os.eCache[:,0].A)[0]#误差缓存矩阵得到记录所有样本的标志位列表
    if (len(validEcacheList))>1:#选最大步长的
        for k in validEcacheList:
            if k==i:
                continue
            Ek=calcEk(os,k)
            deltaE=abs(Ei-Ek)
            if (deltaE>maxDeltaE):
                maxK=k
                maxDeltaE=deltaE
                Ej=Ek
        return maxK,Ej
    else:#第一次循环采用随机选择
        j=selectJrand(i,os.m)
        Ej=calcEk(os,j)
        return j,Ej
#第k个样本误差存入缓存矩阵，选择第二个alpha用
def updateEk(os,k):
    Ek=calcEk(os,k)
    os.eCache[k]=[1,Ek]
#smo算法优化
def innerL(i,os):
    Ei=calcEk(os,i)
    #选择违反条件最严重alpha
    if((os.labelMat[i]*Ei<-os.tol)and(os.alphas[i]<os.C))or \
    ((os.labelMat[i]*Ei>os.tol)and(os.alphas[i]>0)):
        j,Ej=selectJ(i,os,Ei)
        alphaIold=os.alphas[i].copy()
        alphaJold=os.alphas[j].copy()
        #公式7
        if (os.labelMat[i]!=os.labelMat[j]):
            L=max(0,os.alphas[j]-os.alphas[i])
            H=min(os.C,os.C+os.alphas[j]-os.alphas[i])
            #print("H=11111111111111111111=L")
        else:
            L=max(0,os.alphas[j]+os.alphas[i]-os.C)
            H=min(os.C,os.alphas[j]+os.alphas[i])
            #print("H=222222222222222222=L")
        if H==L:
            #print("H===================L")
            return 0
        #公式8，9
        eta=2.0*os.K[i,j]-os.K[i,i]-os.K[j,j]
        if 0<=eta:
            #print("eta>==================0")
            return 0
        os.alphas[j]-=os.labelMat[j]*(Ei-Ej)/eta
        os.alphas[j]=clipAlpha(os.alphas[j],L,H)
        #updateEk(os,j)
        if(abs(os.alphas[j]-alphaJold)<0.00001):
            #print("j 变化太小")
            return 0
        os.alphas[i]+=os.labelMat[j]*os.labelMat[i]*(alphaJold-os.alphas[j])#alpha_j推出alpha_i
        updateEk(os,i)#更新样本i的预测误差
        #计算阙值b
        b1=os.b-Ei-os.labelMat[i]*(os.alphas[i]-alphaIold)*os.K[i,i]-\
        os.labelMat[j]*(os.alphas[j]-alphaJold)*os.K[i,j]
        b2=os.b-Ej-os.labelMat[i]*(os.alphas[i]-alphaIold)*os.K[i,j]-\
        os.labelMat[j]*(os.alphas[j]-alphaJold)*os.K[j,j]
        if (0<os.alphas[i] and os.C>os.alphas[i]):
            os.b=b1
        elif(0<os.alphas[j] and os.C>os.alphas[j]):
            os.b=b2
        else:
            os.b=(b1+b2)/2.0
        return 1
    else:
        return 0
#完整SMO算法,外层循环
def SMOP(dataMatIn,classLabels,C,toler,maxIter,kTup=('lin',0)):
    os=optStruct(np.mat(dataMatIn),np.mat(classLabels).transpose(),C,toler)
    iter=0
    entrireSet=True#是否遍历所有alpha
    alphaPairChanged=0
    while (iter<maxIter) and ((alphaPairChanged>0) or entrireSet):
        alphaPairChanged=0
        if entrireSet:#对整个训练集遍历
            for i in range(os.m):
                alphaPairChanged+=innerL(i,os)
                #print("fullset11111111111111")
            iter+=1
        else:#对非边界上的alpha遍历，0，C间
            nonBoundIs=np.nonzero((os.alphas.A>0)*(os.alphas.A<C))[0]
            for i in nonBoundIs:
                alphaPairChanged+=innerL(i,os)
                #print("fullset22222222222222")
            iter+=1
        if entrireSet:
            entrireSet=False
        elif (0==alphaPairChanged):
            entrireSet=True
        #print("itera===========%d"%(iter))
    return os.b,os.alphas
#获得权重w向量
def calcWs(alphas,dataArr,classLabels):
    X=np.mat(dataArr)
    labelMat=np.mat(classLabels)
    labelMat.transpose()
    rows,cols=np.shape(dataArr)
    w=np.zeros((cols,1))
    for i in range(rows):
        w+=np.multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w
def KernelTrans(DataIn,Sample,kTup):
    DataIn=np.mat(DataIn)
    rows,cols=np.shape(DataIn)
    k=np.mat(np.zeros((rows,1)))
    if "lin"==kTup[0]:#线性核
        k=DataIn*Sample.T 
    elif "rbf"==kTup[0]:#高斯核
        for i in range(rows):
            deltaRow=DataIn[i,:]-Sample
            k[i]=deltaRow*deltaRow.T 
        k=np.exp(k/(-1*kTup[1]**2))
    else:
        print("not legal")
    return k
#训练得到alphas和b
def train(dataset,target):
    Data,DataLabel=dataset,target
    b,alphas=SMOP(Data,DataLabel,200,0.001,50)  #软间隔C=0.6，容差tol=0.001，迭代上限itermax=50
    #计算训练集错误率
    '''
    dataMat=np.mat(Data)
    m,n=np.shape(dataMat)
    labelMat=np.mat(DataLabel)
    svInd=np.nonzero(alphas.A>0)[0]
    svs=dataMat[svInd]
    labelSV=labelMat[0,svInd]
    labelSV=labelSV.T
    err=0
    for i in range(m):
        kernelev=KernelTrans(svs,dataMat[i,:],('rbf',gaosk))
        predict=kernelev.T*np.multiply(labelSV,alphas[svInd])+b
        if np.sign(predict)!=np.sign(DataLabel[i]):
            err+=1
    print(err/m)
    '''
    return b,alphas
#主函数采用one vs one数据集有五类，所以一共有10种对比0-6，0-7，0-8，0-9，6-7，6-8，6-9，7-8，7-9，8-9
#先进行训练
loads='train_data3.mat'
load_data=sio.loadmat(loads)
load_matrixl=load_data['Data']
load_labell=load_data['Label']
load_matrixl=load_matrixl/255.0
m=load_matrixl.shape[0]
load_matrix=[]
load_label=[]
A=B=C=D=E=0
for i in range(m):
    if (load_labell[i]==0)and(A<vcount):
        load_matrix.append(load_matrixl[i])
        load_label.append(load_labell[i])
        A+=1
        continue
    if (load_labell[i]==6)and(B<vcount):
        load_matrix.append(load_matrixl[i])
        load_label.append(load_labell[i])
        B+=1
        continue
    if (load_labell[i]==7)and(C<vcount):
        load_matrix.append(load_matrixl[i])
        load_label.append(load_labell[i])
        C+=1
    if (load_labell[i]==8)and(D<vcount):
        load_matrix.append(load_matrixl[i])
        load_label.append(load_labell[i])
        D+=1
        continue
    if (load_labell[i]==9)and(E<vcount):
        load_matrix.append(load_matrixl[i])
        load_label.append(load_labell[i])
        E+=1
        continue
    if(A==vcount)and(B==vcount)and(C==vcount)and(D==vcount)and(E==vcount):
        break      
print("kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk")
print(A)
print(B)
#print(load_matrix)
#print(load_label)
#pca
meanVals=np.mean(load_matrix,axis=0)
meanRemoved=load_matrix-meanVals
covMat=np.cov(meanRemoved,rowvar=0)
eigVals,eigVects=np.linalg.eig(np.mat(covMat))
eigValInd=np.argsort(eigVals)
eigValInd=eigValInd[:-(dimen+1):-1]
redEigVects=eigVects[:,eigValInd]
lowDDataMat=meanRemoved*redEigVects
#得到10个分类器b，alpha
D=np.array([[0,6],[0,7],[0,8],[0,9],[6,7],[6,8],[6,9],[7,8],[7,9],[8,9]])
dictclass={}
trainset,trainlabel=load_matrix,load_label
for a,c in D:
    index=np.concatenate((np.nonzero(trainlabel==a)[0],np.nonzero(trainlabel==c)[0]))
    target11=np.array([trainlabel[k] for k in index])
    target1=np.array([-1 if k!=a else 1 for k in target11])
    trainset1=np.array([trainset[k] for k in index])
    b,alphas=train(trainset1,target1)
    dictclass[str(a)+str(c)]=(b,alphas)
#print(dictclass)
#对测试集进行分类
loads='test_data3.mat'
load_data=sio.loadmat(loads)
load_matrixt=load_data['Data']
load_labelt=load_data['Label']
load_matrixt=load_matrixt/255.0
#测试集进行pca降维
meanVals=np.mean(load_matrixt,axis=0)
meanRemoved=load_matrixt-meanVals
covMat=np.cov(meanRemoved,rowvar=0)
eigVals,eigVects=np.linalg.eig(np.mat(covMat))
eigValInd=np.argsort(eigVals)
eigValInd=eigValInd[:-(dimen+1):-1]
redEigVects=eigVects[:,eigValInd]
lowDDataMatt=meanRemoved*redEigVects
print("ggggggggggggggggggggggggggggggggggggggg")
testset,testlabel=load_matrixt,load_labelt
testset=np.mat(testset)
testlabel=np.mat(testlabel)
m=testset.shape[0]
corr=0
label=0
#print(dictclass)
for i in range(m):
    acount=np.zeros(5,dtype=int)
    for key,value in dictclass.items():
        b,alphas=value
        #print(alphas)
        #print("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
        #print(b)
        svInd=np.nonzero(alphas.A>0)[0]

        O=np.array([[int(key[0]),int(key[1])]])
        for a,c in O:
            #print(trainlabel==a)
            index=np.concatenate((np.nonzero(trainlabel==a)[0],np.nonzero(trainlabel==c)[0]))
            #print(index)
            target11=np.array([trainlabel[k] for k in index])
            target1=np.mat([-1 if k!=a else 1 for k in target11])
            target1=target1.T
            #print(target1)
            trainset1=np.array([trainset[k] for k in index])
            #print(trainset1)
            sVs=np.array([trainset1[k] for k in svInd])
            #print(np.shape(sVs))
    
            Kernelev=KernelTrans(sVs,testset[i,:],('rbf',gaosk))
            #print(np.shape(Kernelev.T))
            #print(np.shape(target1))
            #print(np.shape(alphas[svInd]))
            predict=Kernelev.T*np.multiply(target1[svInd],alphas[svInd])+b
            #ws=calcWs(alphas[svInd],sVs,labelSVg)
            #predict=Kernelev.T*np,mat(ws)+b
            #print(predict)
            if predict>0:
                if key == '06' or key == '07' or key == '08' or key == '09':
                    acount[0]+=1
                if key == '67' or key == '68' or key == '69':
                    acount[1]+=1
                if key == '78' or key == '79':
                    acount[2]+=1
                if key == '89':
                    acount[3]+=1
            elif predict<0:
                if key == '09' or key == '69' or key == '79' or key == '89':
                    acount[4]+=1
                if key == '08' or key == '68' or key == '78':
                    acount[3]+=1
                if key == '07' or key == '67':
                    acount[2]+=1
                if key == '06':
                    acount[1]+=1
    #计算错误率
    #print(acount)
    alist=acount.tolist()
    if alist.index(np.max(alist))==0:
        label=0
    else:
        label=alist.index(np.max(alist))+5
    print(label)
    if label == testlabel[i]:
        corr+=1
        #print(corr)
print(corr/m)