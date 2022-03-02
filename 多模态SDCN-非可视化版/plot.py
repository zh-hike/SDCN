
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from sklearn import manifold


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
    plt.savefig('plot/%s.png'%filename,dpi=400)

def plot_train_state(nmi_Qs,nmi_Ps,nmi_Zs,graphname):
    plt.figure(figsize=(12,10))
    plt.plot(list(range(1,len(nmi_Qs)+1)),nmi_Qs,label="acc_Q")
    plt.plot(list(range(1,len(nmi_Ps)+1)),nmi_Ps,label="acc_P")
    plt.plot(list(range(1,len(nmi_Zs)+1)),nmi_Zs,label="acc_Z")
    plt.legend(loc="lower right")
    plt.savefig('plot/%s-train_acc.png'%graphname,dpi=400)

def plot_loss(losses,graphname):
    plt.figure(figsize=(12,10))
    plt.plot(list(range(1,len(losses)+1)),losses,label="loss")
    plt.legend(loc=1)
    plt.savefig('plot/%s-train_loss.png'%graphname,dpi=400)
def show(data,filename):
    plt.figure(figsize=(12, 8))
    plt.plot(list(range(1, len(data) + 1)), data)
    plt.savefig('plot/%s.png'%filename,dpi=300)

def show_AE_loss(data,epochs):

    plt.figure(figsize=(12,8))
    plt.plot(list(range(1,len(data)+1)),data)
    plt.title('plot/%s_epochs__AE_loss.png')
    plt.xlabel('epochs')
    plt.savefig('plot/%s_epochs__AE_loss.png'%epochs,dpi=300)


def show_SDCN_loss(data,epochs,loss_name):
    #plt.figure(figsize=(12, 8))
    #plt.rcParams['font.sans-serif'] = ['Arial Unicode Ms']
    for i,name in zip(data,loss_name):
        plt.plot(i,label=name)

    plt.legend(loc=1,fontsize=10)

    #plt.plot(list(range(1,len(data)+1)), data)
    #plt.title('plot/%s_epochs__SDCN_loss.png' % epochs)
    plt.xlabel('epochs')
    plt.savefig('plot/%s_epochs__SDCN_loss.png' % epochs, dpi=400)


def show_SDCN_acc(data,epochs):
    plt.figure(figsize=(12, 8))
    plt.plot(list(range(1, len(data)+1)), data)
    plt.title('plot/%s_epochs__SDCN_acc.png' % epochs)
    plt.xlabel('epochs')
    plt.savefig('plot/%s_epochs__SDCN_acc.png' % epochs, dpi=400)

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
    plt.savefig('distribute/%s.png'%filename,dpi=300)