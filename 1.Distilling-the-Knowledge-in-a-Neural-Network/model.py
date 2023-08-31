import torch
import torch.nn as nn



class Student(nn.Module):
    def __init__(self, input_size=224, num_classes=10):
        super(Student, self).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(input_size ** 2, 1000)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.linear2 = nn.Linear(1000, num_classes)

    def forward(self, X):
        X = self.flatten(X)
        X = self.linear1(X)
        X = self.relu(X)
        X = self.dropout(X)
        X = self.linear2(X)
        return X