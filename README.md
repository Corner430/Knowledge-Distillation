### 知识蒸馏（Knowledge Distillation）

> model 文件夹下是一些已经训练好的 teacher 模型

1. Basic knowledge: [Distilling the Knowledge in a Neural Network](1.Distilling-the-Knowledge-in-a-Neural-Network/index.md)
    - $ L_{soft}$ 可是看作是一种正则，使得模型减少过拟合
    - 知识蒸馏和直接从头训练区别在哪？
        - 收敛的形式不一样，大模型的隐藏层比较复杂，搜索的空间比较大，收敛的位置更平滑，也就是得到的解空间更优美。让大模型指导小模型，能让二者的解空间尽量逼近，收敛的位置尽量接近。**说句专业的话，就是让二者的模型参数之间的散度尽量小。**
    - **`distill`时候的学习率要给大一些**
    - 调节合适的温度。通常取 [1, 2, 5, 10]，2最好
    - 对于大模型使用一定的方式，加快收敛速度
    - **对于 $L_{soft}$ 的相对权重，可以通过打印中间梯度来确定一下，这个超参非常重要。**
    - **可以动态调整学习率**
    - 可以避免 teacher 的错误，或者当无法 match teacher 的输出时，可以避免 student 的错误

2. [FITNETS: HINTS FOR THIN DEEP NETS](2.FITNETS-HINTS-FOR-THIN-DEEP-NETS/index.md)
    - **FitNets 更深更窄，深有助于泛化，窄有助于减少计算**
    - **一言以蔽之，KD 学习结果，FitNets 学习中间表示特征**
    - HT 和 KD 之间的**唯一区别**是参数空间的起点
    - **HT 是比 KD 更强的正则化器，因为它在测试集上带来更好的泛化性能**
    - **将窄而深的网络的内部层与教师网络的隐藏状态进行提示，比将它们与分类目标进行提示具有更好的泛化性能。**
    - **参数量差 10 倍**，一般没关系
    - **阶段训练并不会导致前面白折腾，参考 Curriculum Learning**
    - **选择的 hint/guided layer 如果是卷积层，需要用卷积核来做 regressor，否则参数太多**
    - 参考作者预处理的方式
    - 效果甚至会比 teacher model 更好

> **作者提出了另外四种架构，见原文，但是效果不如 FitNet**
