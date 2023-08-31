import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

import model
import data_loader
import train
import evaluate


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)


# load data
batch_size = 256
train_iter, test_iter = data_loader.load_data_fashion_mnist(batch_size=batch_size, resize=224)


# Define teacher & stduent model, Move models and data to GPU, Initialize weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

teacher = torchvision.models.resnet18(pretrained=False)
teacher.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
teacher.fc = nn.Linear(512, 10)
teacher.load_state_dict(torch.load("../models/resnet18_fashion-mnist.pth"))
teacher = teacher.to(device)

student = model.Student().to(device)
student_distill = model.Student().to(device)
student.apply(weights_init)
student_distill.apply(weights_init)


# Define loss and optimizer
criterion_student = nn.CrossEntropyLoss()
criterion_student_distill = nn.CrossEntropyLoss()
optimizer_student = optim.SGD(student.parameters(), lr=0.01)
optimizer_student_distill = optim.SGD(student_distill.parameters(), lr=0.07)



# teacher_loss, teacher_acc = evaluate.evaluate(teacher, test_iter, device)
# print(f"[Teacher] Test Loss: {teacher_loss}, Acc: {100. * teacher_acc}%")


train.train_student(student, train_iter, test_iter, criterion_student, optimizer_student, device, num_epochs=500)

print("---------------" * 3)

train.train_distill(
    teacher,
    student_distill,
    optimizer_student_distill,
    criterion_student_distill,
    train_iter,
    test_iter,
    device,
    num_epochs=500
)