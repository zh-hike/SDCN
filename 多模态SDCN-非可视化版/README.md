<center> <big> SDCN </big></center>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Structural Deep Clustering Network*模型，简称 **SDCN**。目前，卷积在图像识别方面有了很好的表现，人们把卷积的思想带入了图神经网络，组成了*GCN*，在此基础上，此模型利用*Autoencoder*的预训练结果配合*GCN*使用，在深度聚类方面达到了很好的效果。此模型损失函数用到三部分，$L=Lres + a*Lclu + b*Lgcn$。$a，b$为超参数.
> *SDCN*由两部分组成，一部分是*Autoencoder*，另一部分是*GCN*。
* *Autoencoder* 被用来做预训练以及正式训练时和 *GCN* 配合使用。
* *GCN* 在正式训练阶段和 *Autoencoder* 一起训练。

> 超参数选择，此代码的超参数部分在 ***config.py*** 里进行修改
> 对于预训练和正式训练的超参数，例如学习率等，在对应的文件中最后一行进行修改

# 参数介绍
| 模型参数 | 简单描述 |
| :----: | :----: |
| 数据集| 可选handwritten和MNIST |
| 是否只进行可视化 | 若是，则不训练，只展示结果 |
| 是否预训练 | 若是，先进行预训练，但是得到一个好的预训练结果并不容易|
| 预训练轮次 | 默认2000轮，已默认选择最佳轮次 |
| 正式训练学习率 | 默认1e-5，已默认选择最佳学习率|
| 输入a|损失函数$Lclu$的权重|
| 输入b | 损失函数 $Lgcn$的权重 |
