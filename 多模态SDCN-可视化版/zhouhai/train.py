from zhouhai.model import SDCN
from zhouhai.model import get_clusters_center
import paddle
import paddle.nn as nn
from paddle.nn import initializer
from zhouhai.data_loader import Features
from zhouhai.data_loader import MultiViewDataLoader
from zhouhai.data_loader import load_graph
from sklearn.cluster import KMeans
from zhouhai.plot import show_SDCN_loss
from zhouhai.plot import show_SDCN_acc
from zhouhai.plot import plot_loss
import zhouhai.prepare
from zhouhai.plot import show_distributed
from zhouhai.plot import plot_train_state
from zhouhai.evaluate import eva
from zhouhai.evaluate import eva_pre_train
from zhouhai.evaluate import plot_train_cluster
import pandas as pd
from pandas import Series
from zhouhai import pre_train
from zhouhai import config
from zhouhai.plot import show_root_cluster

#**********************   streamlit   ************************
import streamlit as st
from zhouhai.utils import read_show
from zhouhai.utils import write_show

def start(graphname="handwritten", lr=1e-5, epochs=2500, b=5, c=1,pre_train_flag=False):
    zhouhai.prepare.Prepare(graphname)
    a=0.005
    if graphname=="mnist":
        a=100

    dataLoader = MultiViewDataLoader(graphname)
    n_clusters = dataLoader.n_clusters  # 类别数量
    inputs = dataLoader.xs
    adjs = []
    in_features = dataLoader.in_features

    adjs = []
    views = list(range(dataLoader.n_view))
    for view in views:
        adj = load_graph(view, graphname)
        adjs.append(adj)

    targets = dataLoader.y.astype(paddle.int64)
    num_samples = targets.shape[0]

    model = SDCN(in_features, n_clusters, graphname,pre_train_flag)

    model.train()
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=lr)
    # optimizer=paddle.optimizer.SGD(parameters=model.parameters(),learning_rate=0.1)
    Loss_res = nn.MSELoss(reduction="mean")
    Loss_clu = nn.KLDivLoss(reduction="batchmean")
    Loss_gcn = nn.KLDivLoss(reduction="batchmean")
    # Loss_gcn=nn.CrossEntropyLoss()
    metric = paddle.metric.Accuracy()

    losses = []
    all_losses = []
    res_losses = []
    clu_losses = []
    gcn_losses = []
    names = ['all_loss', 'loss_res', 'loss_clu', 'loss_gcn']
    accs = []
    clusters_center = get_clusters_center(inputs, in_features, n_clusters, graphname,pre_train_flag)
    init = initializer.Assign(clusters_center)
    parameter = paddle.create_parameter((n_clusters, config.n_z), dtype=paddle.float32, default_initializer=init)

    model.cluster_center = parameter

    print(model)

    losses = []
    nmi_Qs = []
    nmi_Ps = []
    nmi_Zs = []
    acc_Qs = []
    acc_Ps = []
    acc_Zs = []
    ari_Zs=[]
    #_, xs_nmi, _, _ = eva_pre_train(graphname,pre_train_flag)

    nmis = []
    bar = st.progress(0)
    iteration = st.empty()
    iteration.text("Epoch:  %s/%s" % (1, epochs))

    #state_text=st.empty()
    #state_patten="acc:%.4f          nmi:%.4f          ari:%.4f"
    #state_text.text(state_patten%(0,0,0))






    for epoch in range(1, epochs + 1):
        iteration.text("Epoch:  %s/%s" % (epoch, epochs))
        bar.progress(epoch/epochs)
        optimizer.clear_grad()
        # print(" *********************   模型前  **************************")
        Xs, Q, P, Z = model(inputs, adjs, nmis)
        """print("训练结束后:")
        print(Z)
        print()
        print()
        assert 1==0"""
        acc_Q, nmi_Q, ari_Q, f1_Q = eva(targets, Q.argmax(1))
        acc_P, nmi_P, ari_P, f1_P = eva(targets, P.argmax(1))
        acc_Z, nmi_Z, ari_Z, f1_Z = eva(targets, Z.argmax(1))
        """if nmi_Z<0.0000001:
            break"""
        # acc=acc.equal(targets).astype(paddle.float32).sum().item()/num_classes
        # print('epoches: %s/%s  [=====================]    loss: %s(%s,%s,%s)  ===========   acc:  %0.2f' % (epoch, epochs, loss.item(),loss_res.item(),loss_clu.item(),loss_gcn.item(),acc))
        patten = "epoche: %s [=========] %s:  acc: %.4f     nmi: %.4f    ari:%.4f    f1:%.4f"

        print(patten % (epoch, 'Q', acc_Q, nmi_Q, ari_Q, f1_Q))
        print(patten % (epoch, 'P', acc_P, nmi_P, ari_P, f1_P))
        print(patten % (epoch, 'Z', acc_Z, nmi_Z, ari_Z, f1_Z))
        print()
        aa = len(views)

        loss_res = 0
        for view in views:
            l = Loss_res(Xs[view], inputs[view])
            if view == 0:
                loss_res = l / aa
            else:
                loss_res = loss_res + l / aa

        loss_clu = Loss_clu(Q.log(), P)
        loss_gcn = Loss_gcn(Z.log(), P)
        # loss_gcn=Loss_gcn(Z,P.argmax(1))
        # loss=loss_res
        loss = a * loss_res + b * loss_clu + c * loss_gcn

        loss.backward()
        optimizer.step()
        print('epoches: %s/%s  [=====================]    loss: %s(%s,%s,%s)' % (epoch, epochs, loss.item(),a*loss_res.item(),b*loss_clu.item(),c*loss_gcn.item()))
        #print(Z)
        patten2="%s : %.4f     nmi: %.4f    ari:%.4f    f1:%.4f"

        #state_text.text(state_patten % (acc_Z, nmi_Z, ari_Z))
        losses.append(loss.item())
        nmi_Qs.append(nmi_Q)
        nmi_Ps.append(nmi_P)
        nmi_Zs.append(nmi_Z)
        acc_Qs.append(acc_Q)
        acc_Ps.append(acc_P)
        acc_Zs.append(acc_Z)
        ari_Zs.append(ari_Z)
    #cur = st.empty()
    #cur.text("正在加载损失函数图像")

    d=read_show(False)
    d[graphname]['loss']=losses
    d[graphname]['results']['acc'] = acc_Zs
    d[graphname]['results']['nmi'] = nmi_Zs
    d[graphname]['results']['ari'] = ari_Zs
    d[graphname]['results']['cluster'] = Z.numpy().tolist()
    write_show(d)

    paddle.save(model.state_dict(), 'zhouhai/results_train/%s.pkl' % graphname)
    #plot_train_state(acc_Qs, acc_Ps, acc_Zs, graphname)

    #plot_loss(losses, graphname)
    #plot_train_cluster(graphname,pre_train_flag)




if __name__ == "__main__":
    # start(graphname="handwritten-5view",epochs=2500,b=5,c=1)
    start(graphname="mnist", lr=1e-5, b=1, c=0.5, epochs=4000)
