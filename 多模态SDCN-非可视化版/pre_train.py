from data_loader import Features
from data_loader import MultiViewDataLoader
from model import AE
from model import MultiViewAE
import paddle.nn as nn
import paddle
from config import ey1,ey2,ey3,dy1,dy2,dy3
from plot import show_AE_loss
import prepare
from model import get_Q
from model import get_P
from evaluate import eva
from sklearn.cluster import KMeans
from evaluate import eva_pre_train


def start(graphname="handwritten-5view",pretrain_epochs=2500):
    prepare.Prepare(graphname)
    epochs=pretrain_epochs
    dataLoader=MultiViewDataLoader(graphname)

    inputs=dataLoader.xs
    targets=dataLoader.y
    in_features=dataLoader.in_features

    n_clusters=dataLoader.n_clusters
    model=MultiViewAE(in_features)
    state_dict=None
    #model.train()

    criterion=nn.MSELoss(reduction='mean')
    optimizer=paddle.optimizer.Adam(parameters=model.parameters(),learning_rate=0.0005)
    #optimizer=paddle.optimizer.SGD(parameters=model.parameters(),learning_rate=0.1)
    losses=[]
    kmeans=KMeans(n_clusters=n_clusters,n_init=20)

    sign_loss=999999




    acc=0
    ss=0

    for epoch in range(epochs):
        optimizer.clear_grad()
        _,H,newxs=model(inputs)
        #x_pred=kmeans.fit_predict(outputs)
        #_,x_nmi,_,_=eva(targets,x_pred)
        loss=0
        for i,x in enumerate(newxs):
            loss=loss+criterion(x,inputs[i])

        loss=loss/5
        

        print('epoches: %s/%s  [=====================]  loss: %s' % (epoch+1, epochs,loss.item()))
        loss.backward()
        optimizer.step()
        ss+=1
        """if ss>1500 and ss%11==0:
            H_pred=kmeans.fit_predict(H)
            H_acc,_,_,_=eva(targets,H_pred)
            if acc < H_acc:
                state_dict=model.state_dict()
                acc=H_acc"""
        if sign_loss > loss.item():
            state_dict=model.state_dict()
            sign_loss=loss.item()


    paddle.save(state_dict,'AE_pretrain/%s.pkl'%(graphname))

    accs,xs_nmi,_,_=eva_pre_train(graphname)

    for batch,i in enumerate(xs_nmi,1):
        print("视图 %s : 预训练结果评估:  x_nmi: %s "%(batch,i))


    for batch,i in enumerate(accs,1):
        print("视图 %s : 预训练结果评估:  x_acc: %s "%(batch,i))


if __name__=="__main__":

    start(graphname='MNIST')    #启用MNIST数据集预训练
 
    #start()                    #启用handwritter预训练
