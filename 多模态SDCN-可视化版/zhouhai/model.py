import paddle
import paddle.nn as nn
import numpy as np
from sklearn.cluster import KMeans
from zhouhai.config import ey1,ey2,ey3,dy1,dy2,dy3
import zhouhai.config as config


class GCN(nn.Layer):
    """
    计算gcn模型，

    """
    def __init__(self, in_features, out_features, graphname):
        """
        in_features为输入特征维度
        ou_features为输出维度
        graphname 为当前数据集的名称
        """
        super(GCN, self).__init__()
        div=1
        if graphname == "mnist":
            div = 1
        elif graphname == "handwritten":
            div = 1
        d = np.random.uniform(0,0.005,(in_features, out_features)) / div
        init = nn.initializer.Assign(d)
        self.weight = paddle.create_parameter((in_features, out_features), dtype=paddle.float32,
                                              default_initializer=init)
        # self.weight = paddle.randn((in_features,out_features),dtype=paddle.float32)
        # self.weight.stop_gradient=False

    def forward(self, x, adj, activate=True):
        """print(adj.shape,adj.dtype)
        print(x.shape,x.dtype)
        print(self.weight.shape,self.weight.dtype)"""

        x = adj.mm(x)
        x = x.mm(self.weight)
        if activate:
            x = nn.ReLU()(x)

        return x


class AE(nn.Layer):
    """
    单视图的自编码器,已废弃

    """
    def __init__(self, in_features, ey1, ey2, ey3, dy1, dy2, dy3):
        super(AE, self).__init__()
        # self.norm=nn.BatchNorm(1)
        self.eLinear1 = nn.Linear(in_features, ey1)
        self.eLinear2 = nn.Linear(ey1, ey2)
        self.eLinear3 = nn.Linear(ey2, ey3)

        self.mid_h = nn.Linear(ey3, config.n_z)

        self.dLinear1 = nn.Linear(config.n_z, dy1)
        self.dLinear2 = nn.Linear(dy1, dy2)
        self.dLinear3 = nn.Linear(dy2, dy3)
        self.last_linear = nn.Linear(dy3, in_features)

    def forward(self, x):
        # x=self.norm(x.unsqueeze(1)).squeeze()
        enc_h1 = nn.ReLU()(self.eLinear1(x))
        enc_h2 = nn.ReLU()(self.eLinear2(enc_h1))
        enc_h3 = nn.ReLU()(self.eLinear3(enc_h2))
        H = self.mid_h(enc_h3)
        dec_h1 = nn.ReLU()(self.dLinear1(H))
        dec_h2 = nn.ReLU()(self.dLinear2(dec_h1))
        dec_h3 = nn.ReLU()(self.dLinear3(dec_h2))
        new_x = self.last_linear(dec_h3)

        return new_x, enc_h1, enc_h2, enc_h3, H


class Encoder(nn.Layer):
    """
    多视图的encoder

    """
    def __init__(self, in_feature):
        super(Encoder, self).__init__()
        self.eLinear1 = nn.Linear(in_feature, ey1)
        self.eLinear2 = nn.Linear(ey1, ey2)
        self.eLinear3 = nn.Linear(ey2, ey3)

    def forward(self, x):
        enc_h1 = nn.ReLU()(self.eLinear1(x))
        enc_h2 = nn.ReLU()(self.eLinear2(enc_h1))
        enc_h3 = nn.ReLU()(self.eLinear3(enc_h2))

        return (enc_h1, enc_h2, enc_h3)


class Decoder(nn.Layer):
    """
    多视图的decoder
    """
    def __init__(self, out_feature):
        super(Decoder, self).__init__()
        self.dLinear1 = nn.Linear(config.n_z, dy1)
        self.dLinear2 = nn.Linear(dy1, dy2)
        self.dLinear3 = nn.Linear(dy2, dy3)
        self.last_linear = nn.Linear(dy3, out_feature)

    def forward(self, x):
        dec_h1 = nn.ReLU()(self.dLinear1(x))
        dec_h2 = nn.ReLU()(self.dLinear2(dec_h1))
        dec_h3 = nn.ReLU()(self.dLinear3(dec_h2))
        new_x = self.last_linear(dec_h3)

        return new_x


class MultiViewAE(nn.Layer):
    """
    多视图的自编码器

    """
    def __init__(self, in_features):
        super(MultiViewAE, self).__init__()
        self.n_view = len(in_features)  # 视图数量
        self.encoderList = nn.LayerList()  # encoder列表
        for i in in_features:
            self.encoderList.append(Encoder(i))
        """self.encoder_v1=Encoder(in_features[0])
        self.encoder_v2=Encoder(in_features[1])
        self.encoder_v3=Encoder(in_features[2])
        self.encoder_v4=Encoder(in_features[3])
        self.encoder_v5=Encoder(in_features[4])"""

        self.layer_H = nn.Linear(ey3 * self.n_view, config.n_z)
        self.decoderList = nn.LayerList()  # decoder列表
        for i in in_features:
            self.decoderList.append(Decoder(i))



    def forward(self, xs):
        encys = []  # 存放多个视图的encoder层的返回
        concatLayer = []  # 存放中间层的前一层
        for i in range(self.n_view):
            ency = self.encoderList[i](xs[i])
            encys.append(ency)
            concatLayer.append(ency[2])


        ency3 = paddle.concat(concatLayer, axis=1)
        H = nn.ReLU()(self.layer_H(ency3))
        newxs = []
        for i in range(self.n_view):
            newx = self.decoderList[i](H)
            newxs.append(newx)

        return encys, H, newxs


class GCN_5(nn.Layer):
    """
    多视图的gcn
    
    """
    def __init__(self, in_features, n_clusters, graphname):
        super(GCN_5, self).__init__()
        self.gcn1 = GCN(in_features, ey1, graphname)
        self.gcn2 = GCN(ey1, ey2, graphname)
        self.gcn3 = GCN(ey2, ey3, graphname)
        self.gcn4 = GCN(ey3, config.n_z, graphname)
        self.gcn5 = GCN(config.n_z, n_clusters, graphname)
        self.norm = nn.BatchNorm(1)

    def forward(self, x, adj, enc1, enc2, enc3, H):
        x = self.gcn1(x, adj)
        sigma = config.sigma
        x = self.gcn2((1 - sigma) * x + sigma * enc1, adj)
        x = self.gcn3((1 - sigma) * x + sigma * enc2, adj)
        x = self.gcn4((1 - sigma) * x + sigma * enc3, adj)
        x = self.gcn5((1 - sigma) * x + sigma * H, adj, False)
        # x = self.norm(x.unsqueeze(1)).squeeze()
        return x


class SDCN(nn.Layer):
    def __init__(self, in_features, clusters_num, graphname,pre_train_flag):
        """
        in_features:      输入5视图的维度，为数组
        clusters_num:     类别数量


        """

        super(SDCN, self).__init__()
        self.clusters_num = clusters_num
        self.ae = MultiViewAE(in_features)
        if pre_train_flag == False:
            graphname = graphname + "-best"
        state_dict = paddle.load('zhouhai/AE_pretrain/%s.pkl' % graphname)
        self.ae.set_state_dict(state_dict)
        self.gcns = nn.LayerList()
        self.n_views = len(in_features)
        # self.norms=nn.LayerList()
        self.norm = nn.BatchNorm(1)
        for i in in_features:
            # self.norms.append(nn.BatchNorm1D(1))
            self.gcns.append(GCN_5(i, self.clusters_num, graphname))
        """self.gcn1 = GCN_5(in_features[0], self.clusters_num)
        self.gcn2 = GCN_5(in_features[1], self.clusters_num)
        self.gcn3 = GCN_5(in_features[2], self.clusters_num)
        self.gcn4 = GCN_5(in_features[3], self.clusters_num)
        self.gcn5 = GCN_5(in_features[4], self.clusters_num)"""
        self.cluster_center = paddle.create_parameter((clusters_num, config.n_z), dtype=paddle.float32)

    def forward(self, xs, adjs, nmis):
        """
        xs:        输入的5视图的数据  为数组，每个元素为一个视图的数据
        adjs:      拉普拉斯矩阵，为数组，
        nmis:      每个视图的nmi评估

        return Xs,Q,P,Z
        """

        # x1,x2,x3,x4,x5=xs
        # adj1,adj2,adj3,adj4,adj5=adjs

        encys, H, newxs = self.ae(xs)

        
        Zs = []
        for i in range(self.n_views):
            # x=(xs[i]-xs[i].mean())/xs[i].std()
            z = self.gcns[i](xs[i], adjs[i], encys[i][0], encys[i][1], encys[i][2], H)
            """print(z)
            print()"""
            Zs.append(z)

        Xs = newxs

        Z = 0
        for i in Zs:
            Z = Z + i
        Z = Z / self.n_views
        Z = self.norm(Z.unsqueeze(1)).squeeze()
        sf = nn.Softmax(1)
        Z = sf(Z)

        
        clusters_center = self.cluster_center
        assert H.shape[1] == clusters_center.shape[1]
        Q = get_Q(H, clusters_center)
        P = get_P(Q)

        assert Q.shape == Z.shape
        assert P.shape == Q.shape
        Q = Q.astype(paddle.float32)
        P = P.astype(paddle.float32)
        Z = Z.astype(paddle.float32)
        return Xs, Q, P, Z


def get_P(Q):
    """
    计算p
    """
    q_f = Q.pow(2) / Q.sum(0)
    return q_f / q_f.sum(1).unsqueeze(1)


def get_Q(H, cluster_center):
    """
    计算q
    """
    v = config.v
    assert H.dim() == 2
    U = cluster_center
    m = -(v + 1) / 2  # 指数
    d = H.unsqueeze(1) - U
    d = d.pow(2).sum(2) / v + 1
    assert d.shape[0] == H.shape[0]
    assert d.shape[1] == U.shape[0]
    q = d.pow(m)
    return q / (q.sum(1).unsqueeze(1))


def get_clusters_center(inputs, in_features, n_clusters, graphname,pre_train_flag):
    """
    inputs:       输入数据  5视图，数组形式
    in_features:  输入的5个视图的维度，数组形式
    n_clusters:   表示类别的数量

    return:       自编码器中间H的聚类中心
    """

    ae = MultiViewAE(in_features)
    if pre_train_flag==False:
        graphname=graphname+"-best"
    state_dict = paddle.load('zhouhai/AE_pretrain/%s.pkl' % graphname)
    ae.set_state_dict(state_dict)
    _, H, _ = ae(inputs)
    model = KMeans(n_clusters, n_init=20)
    model.fit(H)
    return paddle.to_tensor(model.cluster_centers_, dtype=paddle.float32)

