1. [知识蒸馏](1.知识蒸馏.ipynb)
2. [Fashion mnist code](2.fashion_mnist_main.ipynb)
3. [MNIST code](3.mnist_main.ipynb)

- **`distill`时候的学习率要给大一些**
- 调节合适的温度
- 对于大模型使用一定的方式，加快收敛速度
- **对于 $L_{soft}$ 的相对权重，可以通过打印中间梯度来确定一下，这个超参非常重要。**
- **可以动态调整学习率**