import numpy as np
import paddle
from config import topK
from config import path
from sklearn.preprocessing import normalize
import pickle
import scipy.io as io
def cal_graph(mode='dot',view=0,graphname=""):
    """
    给定数据所在的文件名，读出数据，计算node之间的相似度，取出前topK相似的节点，用稀疏矩阵的方式写入./graph
    ./graph中存放数据对应的邻接矩阵，即A，不包括同一节点之间的连线
    :param graphname: 存在于./data下的图所在的文件名（不包括后缀）
    :return: None
    """
    xs=None
    print(graphname)
    if graphname=="MNIST":
        f=open('data/MNIST.pkl','rb')
        d=pickle.load(f)
        f.close()
        xs=d['X'][0][view]
    elif graphname=="handwritten-5view":
        path1='data/%s.mat'%(graphname)
        xs=io.loadmat(path1)['X'][0][view]
    s=None   #相似度矩阵
    samples_size, num = xs.shape
    if mode=='heat':
        """
            ||xi-xj||**2        
          —  ————————————   
                t
        s=e
        """

        s=xs.reshape(samples_size,1,num)-xs
        s=-(np.sum(s,2)**2)/2
        s=np.exp(s)
    elif mode=='dot':
        s=xs.dot(xs.T)
    elif mode=='ncos':
        xs[xs>0]=1
        xs=normalize(xs,axis=1,norm="l1")
        s=np.dot(xs,xs.T)

        pass

    inds=[]
    #计算topk
    for i in range(samples_size):
        ind=np.argpartition(s[i,:],-(topK+1))[-(topK+1):]
        inds.append(ind)

    file=open('graph/%s-%s.txt'%(graphname,view),'w')

    #把特征提取出来把每条边存放在对应的文件中
    for i,v in enumerate(inds):
        for vv in v:
            if vv==i:
                pass
            else:
                file.write('%s %s\n'%(i,vv))
                file.flush()

    file.close()


