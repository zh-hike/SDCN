import scipy.sparse as sp
import paddle
from paddle.io import Dataset
import numpy as np
import pickle
import os
import zhouhai.calcute_graph as calcute_graph
import scipy.io as io
import paddle.nn as nn



def load_graph(view,graphname):
    """
    构建拉普拉斯矩阵 ,adj
    :param graphname:
    :return: 拉普拉斯矩阵
    """
    if not os.path.exists('zhouhai/graph/%s-%s.txt'%(graphname,view)):

        calcute_graph.cal_graph(mode="ncos", view=view, graphname=graphname)

    adj=np.loadtxt('zhouhai/graph/%s-%s.txt'%(graphname,view),dtype='float32')
    num=int(adj.max()+1)

    #创建稀疏矩阵
    adj=sp.coo_matrix((np.ones(adj.shape[0]),(adj[:,0],adj[:,1])),shape=(num,num))
    adj=((adj+adj.T)>0)*1
    indexs=list(range(num))
    I=sp.coo_matrix((np.ones(num),(indexs,indexs)),shape=(num,num))
    adj=adj+I

    adj=Laplacian(adj)
    adj=paddle.to_tensor(adj.toarray(),dtype=paddle.float32)
    return adj


def Laplacian(adj):
    D=np.array(adj.sum(1)).squeeze()
    indexs=list(range(len(D)))
    D=sp.coo_matrix((D,(indexs,indexs)),shape=(len(D),len(D)),dtype='float32')
    D=D.power(-0.5)
    D=D.dot(adj).dot(D)
    D.stop_gradient=True
    return D




class MultiViewDataLoader(Dataset):
    """
    导入多视图的数据，

    """
    def __init__(self,name):
        super(MultiViewDataLoader,self).__init__
        if name=="handwritten":
            views=[0,1,2,3,4]
            self.n_view=5
        if name=="mnist":
            views=[0,1,2]
            self.n_view=3
        self.xs=[]
        self.in_features=[]
        self.y=None
        self.n_clusters=None
        for view in views:
            feature=Features(view,name)
            self.xs.append(feature.x)
            self.y=feature.y
            self.in_features.append(feature.get_features_dim())
            self.n_clusters=feature.get_clusters_num()


class Features(Dataset):
    """
    Dataset子类，返回每个node的feature以及label
    """
    def __init__(self,view,graphname):
        super(Features,self).__init__()
        path="data"
        if graphname=="handwritten":
            self.x=io.loadmat('zhouhai/%s/%s.mat'%(path,graphname))['X'][0][view]
            self.y=io.loadmat('zhouhai/%s/%s.mat'%(path,graphname))['Y'].squeeze()
        elif graphname=="mnist":
            f=open('zhouhai/data/mnist.pkl','rb')
            d=pickle.load(f)
            f.close()
            self.x=d['X'][0][view]
            self.y=d['Y'].squeeze()
        self.x = paddle.to_tensor(self.x,dtype=paddle.float32,stop_gradient=True)
        self.y = paddle.to_tensor(self.y,dtype=paddle.float32,stop_gradient=True)
        
    def __getitem__(self, index):

        return self.x[index],self.y[index]


    def __len__(self):
        return self.x.shape[0]

    def get_features_dim(self):
        return self.x.shape[1]


    def get_clusters_num(self):

        return np.unique(self.y).shape[0]


