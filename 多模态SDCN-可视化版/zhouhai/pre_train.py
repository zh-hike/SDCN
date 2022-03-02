from zhouhai.data_loader import MultiViewDataLoader
from zhouhai.model import AE
from zhouhai.model import MultiViewAE
import paddle.nn as nn
import paddle
from zhouhai.plot import show_AE_loss
import zhouhai.prepare as prepare
from zhouhai.model import get_Q
from zhouhai.model import get_P
from zhouhai.evaluate import eva
from sklearn.cluster import KMeans
from zhouhai.evaluate import eva_pre_train
import pandas as pd
from pandas import Series,DataFrame
from zhouhai.evaluate import eva_root_data
##***********************************************************************
#streamlit

import streamlit as st





##***********************************************************************

def start(graphname="handwritten",pretrain_epochs=2500):
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
    bar = st.progress(0)
    iteration=st.empty()
    iteration.text("Epoch:  %s/%s"%(1,epochs))

    loss_text=st.empty()

    for epoch in range(epochs):
        iteration.text("Epoch:  %s/%s" % (epoch+1, epochs))
        bar.progress((epoch+1)/epochs)
        optimizer.clear_grad()
        _,H,newxs=model(inputs)
        #x_pred=kmeans.fit_predict(outputs)
        #_,x_nmi,_,_=eva(targets,x_pred)
        loss=0
        for i,x in enumerate(newxs):
            loss=loss+criterion(x,inputs[i])

        loss=loss/5
        
        #loss_text.text("loss: %s"%loss.item())
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


    paddle.save(state_dict,'zhouhai/AE_pretrain/%s.pkl'%(graphname))

    accs,xs_nmi,_,_=eva_pre_train(graphname,True)
    df=DataFrame()
    columns=[" "]+['第%s视图'%i for i in list(range(1,dataLoader.n_view+1))]
    for batch,i in enumerate(xs_nmi,1):
        print("视图 %s : 预训练结果评估:  x_nmi: %s "%(batch,i))
    _,root_accs,_,_=eva_root_data(graphname)
    ser=Series(['原始数据']+root_accs,index=columns)
    df=df.append(ser,ignore_index=True)
    ser=Series(['预训练后数据']+accs,index=columns)
    df=df.append(ser,ignore_index=True)
    df=df.set_index(" ")

    for batch,i in enumerate(accs,1):
        print("视图 %s : 预训练结果评估:  x_acc: %s "%(batch,i))


if __name__=="__main__":
    start(graphname='mnist')
    #start()
