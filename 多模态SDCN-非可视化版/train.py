from model import SDCN
from model import get_clusters_center
import paddle
import paddle.nn as nn
from paddle.nn import initializer
from data_loader import Features
from data_loader import MultiViewDataLoader
from data_loader import load_graph
from config import ey1, ey2, ey3, dy1, dy2, dy3
from config import a, b
import config
from sklearn.cluster import KMeans
from plot import show_SDCN_loss
from plot import show_SDCN_acc
from plot import plot_loss
import prepare
from plot import show_distributed
from plot import plot_train_state
from evaluate import eva
from evaluate import eva_pre_train
from evaluate import plot_train_cluster



def start(graphname="handwritten-5view", lr=1e-5, epochs=2500, a=0.005, b=5, c=1):
    prepare.Prepare(graphname)

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

    model = SDCN(in_features, n_clusters, graphname)

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
    clusters_center = get_clusters_center(inputs, in_features, n_clusters, graphname)
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
    _, xs_nmi, _, _ = eva_pre_train(graphname)

    nmis = []

    for batch, i in enumerate(xs_nmi, 1):
        print("视图 %s : 预训练结果评估:  x_nmi: %s " % (batch, i))
        nmis.append(i)
    for epoch in range(1, epochs + 1):
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
        losses.append(loss.item())
        nmi_Qs.append(nmi_Q)
        nmi_Ps.append(nmi_P)
        nmi_Zs.append(nmi_Z)
        acc_Qs.append(acc_Q)
        acc_Ps.append(acc_P)
        acc_Zs.append(acc_Z)

    paddle.save(model.state_dict(), 'results_train/%s.pkl' % graphname)
    plot_train_state(acc_Qs, acc_Ps, acc_Zs, graphname)

    plot_loss(losses, graphname)
    plot_train_cluster(graphname)


if __name__ == "__main__":
    # start(graphname="handwritten-5view",epochs=2500,a=0.005,b=5,c=1)
    start(graphname="MNIST", lr=1e-5, a=100, b=1, c=0.5, epochs=4000)
