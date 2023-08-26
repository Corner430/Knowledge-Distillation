import torch
import torch.nn as nn
import torch.optim as optim

import model
import data_loader
import train


def temperature_softmax(input_tensor, temperature=1.0):
    return torch.softmax(input_tensor / temperature, dim=1)


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)


batch_size = 256
train_iter, test_iter = data_loader.load_data_fashion_mnist(batch_size=batch_size)


# Define teacher & stduent model, Move models and data to GPU, Initialize weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher = model.TeacherModel().to(device)
student = model.Student().to(device)
student_distill = model.Student().to(device)
student.apply(weights_init)
student_distill.apply(weights_init)


# Define loss and optimizer
criterion_teacher = nn.CrossEntropyLoss()
criterion_student = nn.CrossEntropyLoss()
criterion_student_distill = nn.CrossEntropyLoss()
optimizer_teacher = optim.RMSprop(teacher.parameters(), lr=1e-4)
optimizer_student = optim.SGD(student.parameters(), lr=0.01)
optimizer_student_distill = optim.SGD(student_distill.parameters(), lr=0.01)



train.train_teacher(teacher, optimizer_teacher, criterion_teacher, train_iter, test_iter, device,num_epochs=50)
print("----------"*10)
train.train_student(student, optimizer_student, criterion_student, train_iter, test_iter, device,num_epochs=350)
print("----------"*10)
train.train_distill(teacher, student_distill, optimizer_student_distill, criterion_student_distill, train_iter, test_iter, device, num_epochs=350)