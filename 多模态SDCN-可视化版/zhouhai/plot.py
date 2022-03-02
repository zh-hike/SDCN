
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from sklearn import manifold
import streamlit as st
import plotly.graph_objs as go
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np

def get_layout(title="",xtitle="",ytitle="",ytickvals=None):
    layout_comp = go.Layout(
        hovermode="closest",
        width=850,
        height=600,
        xaxis=dict(
            title=xtitle,
            zeroline=False,
            tickfont=dict(
                size=24,
            ),
            titlefont=dict(
                size=24,
            )
        ),
        yaxis=dict(
            title=ytitle,
            zeroline=False,
            nticks=11,
            tick0=0,
            tickvals=ytickvals,
            tickfont=dict(
                size=24,
            ),
            titlefont=dict(
                size=20,
            )
        ),
        legend=dict(
            font=dict(
                size=24,
                color="black",
            ),
            bordercolor="black",
        )
    )


    return layout_comp


def get_cluster(d,rd_method,best):
    pca=PCA(n_components=2)
    if best==False:
        if rd_method=="TSNE":
            pca=manifold.TSNE(n_components=2)

        data = pca.fit_transform(d)
    else:
        data=d

    model=KMeans(n_clusters=10)
    predict=model.fit_predict(d)
    predict=np.array(predict)
    predict=predict-predict.min()
    data=np.array(data)
    datas=[]
    print(data)
    for i in list(range(10)):
        d=data[np.where(predict==i)[0]]

        datas.append(go.Scatter(
            x=d.T[0],
            y=d.T[1],
            name='digit %s'%i,
            mode='markers',
            marker=dict(
                size=4,
            ),
        ))

    layout_comp = get_layout(xtitle="", ytitle="")
    f = go.Figure(data=datas, layout=layout_comp)
    return f,data.tolist()

def get_loss(data):
    layout_comp=get_layout(xtitle="轮次",ytitle="损失值")
    datas=[]
    datas.append(go.Scatter(
        x=list(range(1,len(data)+1)),
        y=data,
        mode='lines',
        marker=dict(
            color='red',
        )
    ))
    f=go.Figure(data=datas,layout=layout_comp)
    return f


def get_acc(accs,nmis,aris,loss):
    layout_comp = get_layout(xtitle="轮次", ytitle="数值")
    datas = []
    colors=['red','green','blue','pink']
    names=['ACC','NMI','ARI','Loss']
    data=[accs,nmis,aris,loss]
    for name,color,d in zip(names,colors,data):
        datas.append(go.Scatter(
            x=list(range(1, len(d) + 1)),
            y=d,
            mode='lines',
            name=name,
            marker=dict(
                color=color,
            )
        ))

    f = go.Figure(data=datas, layout=layout_comp)
    return f






def plot_cluster_result(data,label,filename):
    """
    绘制聚类结果，散点图表示
    data:  二维数据
    label: 数据标签
    
    """
    
    plt.figure(figsize=(12,10))
    tsne=manifold.TSNE(n_components=2,init='pca',random_state=1)
    xs=tsne.fit_transform(data)
    x=xs.T[0]
    y=xs.T[1]
    plt.scatter(x,y,c=label,marker='.')
    plt.savefig('zhouhai/plot/%s.png'%filename,dpi=400)

def plot_train_state(nmi_Qs,nmi_Ps,nmi_Zs,graphname):
    plt.figure(figsize=(12,10))
    plt.plot(list(range(1,len(nmi_Qs)+1)),nmi_Qs,label="acc_Q")
    plt.plot(list(range(1,len(nmi_Ps)+1)),nmi_Ps,label="acc_P")
    plt.plot(list(range(1,len(nmi_Zs)+1)),nmi_Zs,label="acc_Z")
    plt.legend(loc="lower right")
    plt.savefig('zhouhai/plot/%s-train_acc.png'%graphname,dpi=400)

def plot_loss(losses,graphname):


    plt.figure(figsize=(12,10))
    plt.plot(list(range(1,len(losses)+1)),losses,label="loss")
    plt.legend(loc=1)
    plt.savefig('zhouhai/plot/%s-train_loss.png'%graphname,dpi=400)



def show(data,filename):
    plt.figure(figsize=(12, 8))
    plt.plot(list(range(1, len(data) + 1)), data)
    plt.savefig('zhouhai/plot/%s.png'%filename,dpi=300)

def show_AE_loss(data,epochs):

    plt.figure(figsize=(12,8))
    plt.plot(list(range(1,len(data)+1)),data)
    plt.title('plot/%s_epochs__AE_loss.png')
    plt.xlabel('epochs')
    plt.savefig('zhouhai/plot/%s_epochs__AE_loss.png'%epochs,dpi=300)


def show_SDCN_loss(data,epochs,loss_name):
    #plt.figure(figsize=(12, 8))
    #plt.rcParams['font.sans-serif'] = ['Arial Unicode Ms']
    for i,name in zip(data,loss_name):
        plt.plot(i,label=name)

    plt.legend(loc=1,fontsize=10)

    #plt.plot(list(range(1,len(data)+1)), data)
    #plt.title('plot/%s_epochs__SDCN_loss.png' % epochs)
    plt.xlabel('epochs')
    plt.savefig('zhouhai/plot/%s_epochs__SDCN_loss.png' % epochs, dpi=400)


def show_SDCN_acc(data,epochs):
    plt.figure(figsize=(12, 8))
    plt.plot(list(range(1, len(data)+1)), data)
    plt.title('plot/%s_epochs__SDCN_acc.png' % epochs)
    plt.xlabel('epochs')
    plt.savefig('zhouhai/plot/%s_epochs__SDCN_acc.png' % epochs, dpi=400)

def show_distributed(data,subplot,filename):
    row,col=subplot[0],subplot[1]
    fig=plt.figure(figsize=(20,3*row))
    index=1
    for i,y in zip(range(1,row*col+1),data):
        axe=fig.add_subplot(row,col,i)
        x=list(range(1,len(y)+1))
        axe.bar(x,y)
        axe.set_xticks(x)
        axe.set_ylim([0,2000])
    plt.savefig('zhouhai/distribute/%s.png'%filename,dpi=300)




def show_root_cluster(graphname):
    if graphname=='handwritten':
        views=[0,1,2,3,4]
    elif graphname=='mnist':
        views=[0,1,2]

    for i in views:
        cur = st.empty()
        cur.text("正在加载视图 - %s"%(i+1))
        st.header("原始数据降维 视图 %s"%(i+1))
        st.image('zhouhai/plot/%s-原始数据聚类结果-%s.png' % (graphname,i))
        cur.text("")