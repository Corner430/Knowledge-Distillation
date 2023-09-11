### 知识蒸馏（Knowledge Distillation）

> model 文件夹下是一些已经训练好的 teacher 模型

1. Basic knowledge: [Distilling the Knowledge in a Neural Network](1.Distilling-the-Knowledge-in-a-Neural-Network/index.md) (KD)
    - $ L_{soft}$ 可是看作是一种正则，使得模型减少过拟合
    - 知识蒸馏和直接从头训练区别在哪？
        - 收敛的形式不一样，大模型的隐藏层比较复杂，搜索的空间比较大，收敛的位置更平滑，也就是得到的解空间更优美。让大模型指导小模型，能让二者的解空间尽量逼近，收敛的位置尽量接近。**说句专业的话，就是让二者的模型参数之间的散度尽量小。**
    - **`distill`时候的学习率要给大一些**
    - 调节合适的温度。通常取 [1, 2, 5, 10]，2最好
    - 对于大模型使用一定的方式，加快收敛速度
    - **对于 $L_{soft}$ 的相对权重，可以通过打印中间梯度来确定一下，这个超参非常重要。**
    - **可以动态调整学习率**
    - 可以避免 teacher 的错误，或者当无法 match teacher 的输出时，可以避免 student 的错误

2. [FITNETS: HINTS FOR THIN DEEP NETS](2.FITNETS-HINTS-FOR-THIN-DEEP-NETS/index.md): (HT)
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
    - **对于 distill loss 的权重，可以使用一个衰减的方式**

> **作者提出了另外四种架构，见原文，但是效果不如 FitNet**

3. [Deep Mutual Learning](3.Deep-Mutual-Learning/index.md) (DML)
    - **dataset: CIFAR-100, Market-1501**
    - **不要 teacher-student，要 peer-to-peer。甚至效果更优**
    - 前人表明小模型可以和大模型达到相同的能力，**真正的难点在于网络的优化，而不是网络的规模**
    - 当 peer-to-peer时，**在所有小型和未经训练的学生网络的情况下，额外的知识从哪来？**
    - **为什么可以收敛到一个不错的解**，而不是 'the blind lead the blind'？
    - **一言以蔽之，student 的初始值不同，后验熵被增加，所以 work**
    - **网络越多，效果越好**
    - 可以混合大小异构网络
    - **蒸馏本质上就是增加 后验熵和进行正则**
    - **DML 的目的不是产生多样化的集成，而是让 student 之间尽可能的相似，这一点很难通过传统的监督学习完成**
    - **不使用温度系数**
    - 作者提出了一个替代方案，效果不如原始方案，**原因和高后验熵有关系**
    - 代码使用 Tensorflow 实现，具体实现细节见原文

4. [Knowledge Distillation by On-the-Fly Native Ensemble](4.Knowledge-Distillation-by-On-the-Fly-Native-Ensemble/index.md)  (ONE)
    - KD 是 two stage，ONE 是 one stage，DML 是 peer teach（缺乏一个权威的teacher）
    - grouped convolution
    - 使用 **Gate 集成** 而不是使用 **平均集成** , 并共享底层，用来控制每个branch的权重
    - KL divergence 的使用
    - 为什么 ONE 的泛化性能更好？
    - compact
        - Parameter Binarization
        - Filter Pruning
    - 代码用的Pytorch
    - **ONE 拥有一些正则的作用**

5. [Collaborative Learning for Deep Neural Networks](5.Collaborative-Learning-for-Deep-Neural-Networks/index.md)    (CL-ILR)

6. [Online knowledge Distillation with Diveres Peers](6.Online-knowledge-Distillation-with-Diverse-Peers/index.md)  (OKDDip)
    - dataset: CIFAR-10, CIFAR-100, ImageNet-2012
    - KD 并不总是可用
    - 最近提出的在线蒸馏可以让学生之间相互学习，并摒弃 teacher network，**问题在于过早的造成了学生之间的相似饱和**
    - **通过 *attention* 个性化定制 *teacher model*, 实现了一对一教学**
    - 可以将 $L(\mathbf{h_a})$ 和 $E(\mathbf{h_a})$ 分别看作 $Q$ 和 $K$
    - **非对称性给了 不太好的模型向好的模型去学习，好的模型不被不好的模型拉下水 的可能性**
    - 不同时期的 *group leader* 可以有不同的计算方式
    - *branch-based or network-based student models*
    - *[code](https://github.com/DefangChen/OKDDip-AAAI2020)*
    - 一般而言，*network-based* 的效果比 *branch-based* 的效果要好, 因为它们有着更多独立参数，使得 *peer* 更加多样性
    - *Homogenization problem tends to become more severe for dealing with easier dataset.*
    - **计算 peer 多样性**
    - **peer 多样性导致了 更好的 ensemble 效果**
