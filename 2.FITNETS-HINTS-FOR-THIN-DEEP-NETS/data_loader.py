import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data



def load_data_cifar_10(batch_size, resize=None):
    """Download the Cifar-10 dataset and then load it into memory"""
    # 数据预处理
    trans = [transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # 对图像进行归一化

    if resize:
        trans.insert(0, transforms.Resize(resize))
    
    trans = transforms.Compose(trans)
    
    # 加载CIFAR-10数据集
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True,download=True, transform=trans)
    testset = torchvision.datasets.CIFAR10(root='../data', train=False,download=True, transform=trans)

    return (data.DataLoader(trainset, batch_size, shuffle=True,num_workers=4),
            data.DataLoader(testset, batch_size, shuffle=False,num_workers=4))


def load_data_fashion_mnist(batch_size, resize=None):
    """Download the Fashion-MNIST dataset and then load it into memory."""
    trans = [transforms.ToTensor()]

    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)

    mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans, download=True)

    return (data.DataLoader(mnist_train, batch_size, shuffle=True,num_workers=4),
            data.DataLoader(mnist_test, batch_size, shuffle=False,num_workers=4))


def load_data_MNIST(batch_size, resize=None):
    """Download the MNIST dataset and then load it into memory."""
    trans = [transforms.ToTensor()]

    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)

    mnist_train = torchvision.datasets.MNIST(root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.MNIST(root="../data", train=False, transform=trans, download=False)

    return (data.DataLoader(mnist_train, batch_size, shuffle=True,num_workers=4),
            data.DataLoader(mnist_test, batch_size, shuffle=False,num_workers=4))