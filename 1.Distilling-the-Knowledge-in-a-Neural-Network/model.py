import torch
import torch.nn as nn


# Define TeacherModel
class TeacherModel(nn.Module):
    def __init__(self, dropout1=0.25, dropout2=0.5, num_classes=10):
        super(TeacherModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout1),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 625),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(625, num_classes),
        )

    def forward(self, X):
        X = self.features(X)
        X = self.classifier(X)
        return X


class Student(nn.Module):
    def __init__(self, input_size=28, num_classes=10):
        super(Student, self).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(input_size ** 2, 1000)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.linear2 = nn.Linear(1000, num_classes)

    def forward(self, X):
        X = self.flatten(X)
        X = self.linear1(X)
        X = self.relu(X)
        X = self.dropout(X)
        X = self.linear2(X)
        return X