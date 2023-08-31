import torch
from evaluate import evaluate
import torch.nn.functional as F

def temperature_softmax(input_tensor, temperature=1.0):
    return torch.softmax(input_tensor / temperature, dim=1)


# Define student training function
def train_student(net, train_iter, test_iter, criterion, optimizer, device, num_epochs=10):
    for epoch in range(num_epochs):
        runnning_loss = 0.0
        running_acc = 0.0
        for inputs, labels in train_iter:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            runnning_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            running_acc += (preds == labels).float().mean()
        eval_loss, eval_acc = evaluate(net, test_iter, device)
        print("epoch %d, loss %.4f, train acc %.3f, test loss %.4f, test acc %.3f" % (epoch + 1, runnning_loss / len(train_iter), running_acc / len(train_iter), eval_loss, eval_acc))
        # torch.save(net.state_dict(), f"net_epoch{epoch + 1}.pth")

    print("training finished")


# Define distillation training function
def train_distill(teacher, student, optimizer_student, criterion_student, trainloader, testloader, device, num_epochs=10):
    for epoch in range(num_epochs):
        teacher.eval()
        student.train()
        running_loss = 0.0
        running_acc = 0.0
        for X, y in trainloader:
            X = X.to(device)
            y = y.to(device)
            y_teacher = teacher(X)
            y_hat = student(X)

            l_soft = F.cross_entropy(temperature_softmax(y_hat, temperature=2), temperature_softmax(y_teacher, temperature=2))
            l_hard = criterion_student(temperature_softmax(y_hat), y)
            loss = l_soft + 0.25 * l_hard
            
            optimizer_student.zero_grad()
            loss.backward()
            optimizer_student.step()
            running_loss += loss.item()
            _, preds = torch.max(y_hat, 1)
            running_acc += (preds == y).float().mean()
        eval_loss, eval_acc = evaluate(student, testloader, device)
        print("epoch %d, loss %.4f, train acc %.3f, test loss %.4f, test acc %.3f" % (epoch + 1, running_loss / len(trainloader), running_acc / len(trainloader), eval_loss, eval_acc))
        # torch.save(net.state_dict(), f"net_epoch{epoch + 1}.pth")

    print("training finished")