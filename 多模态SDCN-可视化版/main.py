# -*- utf-8 -*-
import streamlit as st

##***********************************************************************
##********************  周孩 start  **************************************

from zhouhai import pre_train
from zhouhai import train
from zhouhai.plot import show_root_cluster
from zhouhai import utils
from zhouhai.utils import oper_introduction
##********************   周孩 end  **************************************
##***********************************************************************



def run_CDIMC_net():
    # sidebar 部分
    st.sidebar.header("数据加载模块")
    st.sidebar.write("**数据集**")
    data_set_name = st.sidebar.selectbox("", ["其它", "handwritten", "mnist"])

    # 变量初始化
    views = None
    classes = None
    sample_nums = None

    if data_set_name == "其它":
        data_set_name = st.sidebar.text_input("请在下面输入相对或完整路径")
    elif data_set_name == "handwritten":
        views = 5
        classes = 10
        sample_nums = 2000
    elif data_set_name == "mnist":
        views = 2
        classes = 10
        sample_nums = 4000

    if data_set_name != "":
        try:
            st.sidebar.write("你已选择数据集" + data_set_name)
            st.sidebar.write("视图数:{} 类别数:{} 样例数:{}".format(views, classes, sample_nums))
        except Exception as E:
            st.sidebar.exception(E)

    st.sidebar.write("**数据缺失比例**")
    percentDel = st.sidebar.selectbox("", [0.1, 0.3, 0.5, 0.7])
    st.sidebar.write("**选择视图缺失索引矩阵**")
    ff = st.sidebar.selectbox("", [0,1,2,3,4,5,6,7,8,9])

    st.sidebar.write("**是否预训练**")
    pre_train_flag = st.sidebar.selectbox("", ["预训练", "加载已有模型"])
    pre_train_flag = pre_train_flag == "预训练"

    st.sidebar.write("**降维方法**")
    rd_method = st.sidebar.selectbox("", ["PCA", "TSNE"])

    st.sidebar.write("**是否只进行可视化**")
    only_visual = st.sidebar.selectbox("", ["否", "是"])
    only_visual = only_visual == "是"

    flag = st.sidebar.button("确定参数选择")

    if flag:
        if data_set_name == "handwritten":
            if not only_visual:
                # 运行模型
                run_handwritten(pre_train_flag, percentDel, ff, st)
            visualize_cdimc_result(st, rd_method)
        elif data_set_name == "mnist":
            if not only_visual:
                # 运行模型
                run_mnist(pre_train_flag, percentDel, ff, st)
            visualize_cdimc_result(st, rd_method)


def run_long_app():
    """
    sidebar可以在这里设置
    :return:
    """
    st.sidebar.success("已选择龙云飞的模型")


def run_zhou_app():
    """
    sidebar可以在这里设置
    :return:
    """
    #st.sidebar.success("已选择模型SDCN")
    readme_text=oper_introduction()
    st.sidebar.header("数据加载模块")
    st.sidebar.write("数据集")
    data_set_name = st.sidebar.selectbox("", ["handwritten", "mnist"])
    if data_set_name == "其它":
        data_set_name = st.sidebar.text_input("请在下面输入相对或完整路径")
    elif data_set_name == "handwritten":
        views = 5
        classes = 10
        sample_nums = 2000
    elif data_set_name == "mnist":
        views = 3
        classes = 10
        sample_nums = 1799

    if data_set_name != "":
        try:
            st.sidebar.write("你已选择数据集" + data_set_name)
            st.sidebar.write("视图数:{} 类别数:{} 样例数:{}".format(views, classes, sample_nums))
        except Exception as E:
            st.sidebar.exception(E)
    st.sidebar.write("**降维方法**")
    rd_method = st.sidebar.selectbox("", ["TSNE","PCA"])
    st.sidebar.write("**是否只进行可视化**")
    only_visual = st.sidebar.selectbox("", ["否", "是"])
    only_visual = only_visual == "是"
    #rd_method=None
    if only_visual==False:
        st.sidebar.write("是否预训练")
        pre_train_flag = st.sidebar.selectbox("", ["预训练", "加载已有模型"])
        pre_train_flag = pre_train_flag=="预训练"
        if pre_train_flag:
            pre_train_epochs=st.sidebar.number_input("输入预训练轮次",value=2000,step=100)




        st.sidebar.header("正式训练参数")
        lr=st.sidebar.number_input("请输入正式训练学习率",value=0.00001,step=1e-5,format="%.6f")
        train_epochs=st.sidebar.number_input("请输入正式训练轮次",value=2200,step=100)
        #st.sidebar.write("\nLoss = a\*Loss_res + b\*Loss_clu + c\*Loss_gcn")
        a=0
        b=0
        c=0
        if data_set_name=="handwritten":
            a=0.015
            #a=st.sidebar.number_input("请输入a",value=0.005,step=0.001,format="%.4f")
            b = st.sidebar.number_input("参数a", value=4.0, step=0.1)
            c = st.sidebar.number_input("参数b", value=1.0, step=0.1)
        elif data_set_name=="mnist":
            a=100.0
            #a=st.sidebar.number_input("请输入a",value=100.0,step=1.0)
            b = st.sidebar.number_input("参数a", value=1.0, step=0.1)
            c = st.sidebar.number_input("参数b", value=0.5, step=0.1)

    flag = st.sidebar.button("确定参数选择")

    if flag:
        readme_text.empty()
        st.title("多模态结构化深度聚类网络")
        if only_visual:
            plot=utils.only_show_result(data_set_name,rd_method,True)
            pass
        else:
            if pre_train_flag:
                st.header("预训练")
                pre_train.start(data_set_name,pre_train_epochs)
            st.header("正式训练")
            train.start(data_set_name,lr=lr,epochs=train_epochs,b=b,c=c,pre_train_flag=pre_train_flag)
            utils.only_show_result(data_set_name,rd_method,False)






def run_zang_app():
    """
    sidebar可以在这里设置
    :return:
    """
    st.sidebar.success("已选择龙云飞的模型")


def main():
    readme_text=st.markdown("# 开始演示")
    st.sidebar.title("选择模式")
    app_mode = st.sidebar.selectbox("", ["操作说明", "CDIMC-Net", "龙云飞的模型", "多模态结构化深度聚类网络", "臧煜的模型"])

    if app_mode == "操作说明":
        st.sidebar.success('如果继续，请选择不同的应用')
    elif app_mode == "CDIMC-Net":
        readme_text.empty()
        run_CDIMC_net()
    elif app_mode == "龙云飞的模型":
        readme_text.empty()
        run_long_app()
    elif app_mode == "多模态结构化深度聚类网络":
        readme_text.empty()
        run_zhou_app()
    elif app_mode == "臧煜的模型":
        readme_text.empty()
        run_zang_app()


if __name__ == "__main__":
    main()
