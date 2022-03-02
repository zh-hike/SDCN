from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics import accuracy_score as acc_score
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from data_loader import Features
from data_loader import MultiViewDataLoader
from data_loader import load_graph
from model import AE
from model import SDCN
from model import MultiViewAE
from config import ey1, ey2, ey3, dy1, dy2, dy3

import paddle
import numpy as np
from plot import plot_cluster_result
from munkres import Munkres



def eva_root_data(graphname):
    """
    对原始数据进行评估
    """
    dataLoader = MultiViewDataLoader(graphname)
    inputs, targets = dataLoader.xs, dataLoader.y
    n_clusters = dataLoader.n_clusters
    
    nmis = []
    accs=[]
    aris=[]
    f1s=[]
    for x in inputs:
        kmeans = KMeans(n_clusters, n_init=20)
        x_pred = kmeans.fit_predict(x)
        acc, nmi, ari, f1 = eva(targets, x_pred)
        nmis.append(nmi)
        accs.append(acc)
        aris.append(ari)
        f1s.append(f1)
    return accs, nmis, aris, f1s






def eva(y_true, y_pred):
    """
    计算评估结果，acc,nmi,ari,f1
    """
    y_true=y_true-y_true.min()
    nmi_score = normalized_mutual_info_score(y_true, y_pred)
    ari = ari_score(y_true,y_pred)
    
    y_true=y_true.astype(paddle.int16).tolist()
    try:
        y_pred=y_pred.astype(paddle.int16).tolist()
    except:
        pass
    l1=list(set(y_true))
    l2=list(set(y_pred))

    n_clusters=len(l1)
    cost=np.zeros((n_clusters,n_clusters),dtype=int)
    
    for i in l1:
        
        indexs=[i1 for i1 in range(len(y_true)) if y_true[i1]==i]   #记录i类别的索引
        for j in l2:
            c=[j1 for j1 in indexs if y_pred[j1]==j]
            cost[(i,j)]=len(c)

    m=Munkres()
    cost=-cost
    indexs=m.compute(cost)    #记录最佳match
    new_x=np.zeros_like(y_pred)
    for i in indexs:
        end=i[1]
        y_pred_index=[i1 for i1 in range(len(y_pred)) if y_pred[i1]==end]
        new_x[y_pred_index]=i[0]

    acc=acc_score(y_true,new_x)
    f1=f1_score(y_true,new_x.tolist(),average='macro')


    return acc, nmi_score, ari, f1


def eva_pre_train(graphname):
    """
    评估预训练后的数据
    """
    dataLoader = MultiViewDataLoader(graphname)
    inputs, targets = dataLoader.xs, dataLoader.y

    in_features = dataLoader.in_features
    n_clusters = dataLoader.n_clusters
    model = MultiViewAE(in_features)
    state_dict = paddle.load('AE_pretrain/%s.pkl' % (graphname))
    model.set_state_dict(state_dict)
    _, H, newxs = model(inputs)
    nmis = []
    accs = []
    aris = []
    f1s = []
    for x in newxs:
        kmeans = KMeans(n_clusters, n_init=20)
        x_pred = kmeans.fit_predict(x)
        acc, nmi, ari, f1 = eva(targets, x_pred)
        nmis.append(nmi)
        accs.append(acc)
        aris.append(ari)
        f1s.append(f1)
    return accs, nmis, aris, f1s




def plot_train_cluster(graphname):

    xs_acc,xs_nmi,_,_=eva_pre_train(graphname)

    nmis=[]

    for batch,i in enumerate(xs_acc,1):
        print("视图 %s : 预训练结果评估:  x_acc: %s "%(batch,i))
        nmis.append(i)
    dataLoader=MultiViewDataLoader(graphname)
    in_features=dataLoader.in_features
    n_clusters=dataLoader.n_clusters
    model=SDCN(in_features,n_clusters,graphname)
    state_dict=paddle.load("results_train/%s.pkl"%graphname)
    model.set_state_dict(state_dict)
    inputs,targets=dataLoader.xs,dataLoader.y
    adjs=[]
    views=list(range(dataLoader.n_view))
    for view in views:
        adj=load_graph(view,graphname)
        adjs.append(adj)
    Xs,Q,P,Z = model(inputs,adjs,nmis)
    plot_cluster_result(Z,targets,'%s-训练后聚类结果'%graphname)
    print("plot the cluster result of the train successfully")

def plot_root_data_cluster(graphname):
    dataLoader=MultiViewDataLoader(graphname)
    inputs,targets=dataLoader.xs,dataLoader.y
    plot_cluster_result(inputs[0],targets,'原始数据聚类结果1')
    print("plot the 1th view cluster result successfully")
    plot_cluster_result(inputs[1],targets,'原始数据聚类结果2')
    print("plot the 2th view cluster result successfully")
    plot_cluster_result(inputs[2],targets,'原始数据聚类结果3')
    print("plot the 3th view cluster result successfully")
    plot_cluster_result(inputs[3],targets,'原始数据聚类结果4')
    print("plot the 4th view cluster result successfully")
    plot_cluster_result(inputs[4],targets,'原始数据聚类结果5')
    print("plot the 5th view cluster result successfully")
if __name__=="__main__":

    #plot_train_cluster()
    #plot_root_data_cluster()
    plot_train_cluster("MNIST")
    accs,nmis,_,_=eva_root_data("MNIST")
    for (view,nmi) in enumerate(nmis,1):
        print("原始数据 第  %s  视图(nmi) : %s"%(view,nmi))

    
    for (view,acc) in enumerate(accs,1):
        print("原始数据 第  %s  视图(acc) : %s"%(view,acc))



    





        
