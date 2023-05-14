1. 模型的参数量和其所能捕获的“知识“量之间并非稳定的线性关系(下图中的1)，而是接近边际收益逐渐减少的一种增长曲线
2. 完全相同的模型架构和模型参数量，使用完全相同的训练数据，能捕获的“知识”量并不一定完全相同，另一个关键因素是训练的方法。合适的训练方法可以使得在模型参数总量比较小时，尽可能地获取到更多的“知识”(下图中的3与2曲线的对比).

![img](https://pic2.zhimg.com/v2-f2fc2f02b87a38a9ff34a50664800045_r.jpg)

作者将问题限定在**分类问题**下，或者其他本质上属于分类问题的问题，该类问题的共同点是模型最后会有一个softmax层，其输出值对应了相应类别的概率值

> softmax层:$$S(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}, \quad \text{对于所有 } j = 1, 2, ..., K$$
>
> 其中：$z_i$ 是向量z的第i个元素，$e^{z_i}$ 是 $



知识蒸馏的关键点：如果回归机器学习最最基础的理论，我们可以很清楚地意识到一点(而这一点往往在我们深入研究机器学习之后被忽略): **机器学习最根本的目的**在于训练出在某个问题上泛化能力强的模型。

soft target分布的熵相对高时，其soft target蕴含的知识就更丰富

加入了温度后的softmax函数：

> $$S(z)_i = \frac{e^{z_i/T}}{\sum_{j=1}^{K} e^{z_j/T}}, \quad \text{对于所有 } j = 1, 2, ..., K$$

T越高，softmax的output probability distribution越趋于平滑，其分布的熵越大，负标签携带的信息会被相对地放大，模型训练将更加关注负标签。



通用知识蒸馏方法：

![img](https://pic2.zhimg.com/80/v2-d01f5142d06aa27bc5e207831b5131d9_720w.webp)

- step1：训练Net—T
- step2：在温度T下，蒸馏Net—T得到Net—S

训练Net-T的过程很简单，下面详细讲讲第二步:高温蒸馏的过程。高温蒸馏过程的目标函数由distill loss(对应soft target)和student loss(对应hard target)加权得到。示意图如上。

$$ L = \alpha L_{soft}  +  \beta L_{hard} $$

- Net-T产生的`softmax distribution (with high temperature)` 来作为`soft target`，Net-S在相同温度T条件下的`softmax`输出和soft target的cross entropy就是**Loss函数的第一部分**

$L_{soft} = -\sum_{j}^{N}{P_j}^Tlog({q_j}^T)$,其中$P_i^T=\frac{exp(v_i/T)}{\sum_{k}^{N}{exp(z_k/T)}}$,$q_i^T=\frac{exp(z_i^T/T)}{\sum_{k}^{N}exp(z_k/T)}$

- Net-S在T=1的条件下的`softmax`输出和`ground truth`的`cross entropy`就是**Loss函数的第二部分** $L_{hard}$

$L_{hard} = - \sum{j}^{N}{c_jlog(q_j^l)}$,$q_j^l=\frac{exp(z_i)}{\sum_{k}^{N}{exp(z_k)}}$

第二部分Loss ($L_{hard}$) 的必要性其实很好理解: Net-T也有一定的错误率，使用ground truth可以有效降低错误被传播给Net-S的可能。打个比方，老师虽然学识远远超过学生，但是他仍然有出错的可能，而这时候如果学生在老师的教授之外，可以同时参考到标准答案，就可以有效地降低被老师偶尔的错误“带偏”的可能性。



## 关于温度

1. 从有部分信息量的负标签中学习 --> 温度要高一些
2. 防止受负标签中噪声的影响 -->温度要低一些



关于代码：

