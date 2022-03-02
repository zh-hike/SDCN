import json
import streamlit as st
from zhouhai.plot import get_loss, get_acc, get_cluster
from zhouhai.table import MyhtmlTable
import numpy as np
import copy
def read_show(best):
    if best==False:
        file=open("zhouhai/results_train/show","r")
    else:
        file = open("zhouhai/results_train/show-best", "r")
    d=json.loads(file.read())
    file.close()
    return d


def write_show(d):
    file=open("zhouhai/results_train/show","w")
    file.write(json.dumps(d))
    file.close()




def only_show_result(graphname,rd_method,best):


    data = read_show(best)
    accs = data[graphname]['results']['acc']
    nmis = data[graphname]['results']['nmi']
    aris = data[graphname]['results']['ari']

    acc_fig = get_acc(accs, nmis, aris,data[graphname]['loss'])

    st.title("原始多模态特征降维结果展示")
    names = ['']
    new_d=copy.deepcopy(data)

    new_d[graphname][rd_method]=[]

    for i, d in enumerate(data[graphname][rd_method], 1):
        st.header("模态 %s" % i)
        fig , new_data= get_cluster(d, rd_method,True)
        st.write(fig)
        names.append('模态%s' % i)
        new_d[graphname][rd_method].append(new_data)
    names.append("多模态结构化深度聚类网络")
    st.header("多模态结构化深度聚类网络降维结果展示")
    pp=rd_method
    if best==False:
        pp='cluster'
    fig , new_cluster= get_cluster(data[graphname]['results'][pp], rd_method,best)
    new_d[graphname]['results'][rd_method]=new_cluster
    if best==False:
        write_show(new_d)
    st.write(fig)
    r1 = np.array(data[graphname]['acc']).round(4).tolist()
    r1.append(round(accs[-1], 4))
    r2 = np.array(data[graphname]['nmi']).round(4).tolist()
    r2.append(round(nmis[-1], 4))
    r3 = np.array(data[graphname]['ari']).round(4).tolist()
    r3.append(round(aris[-1], 4))
    table = np.array([names,
                      ['acc'] + r1,
                      ['nmi'] + r2,
                      ['ari'] + r3,

                      ]
                     )




    #st.header("损失函数:")

    #loss_fig=get_loss(data[graphname]['loss'])
    #st.write(loss_fig)

    st.header("训练过程展示")

    st.write(acc_fig)


    table=MyhtmlTable(table.shape,table,'模型性能评估')
    st.write(table.get_table(),unsafe_allow_html=True)
    pass




def oper_introduction():

    file=open("zhouhai/welcome.md",'r')
    m=file.read()
    file.close()
    return st.markdown(m,unsafe_allow_html=True)





    pass