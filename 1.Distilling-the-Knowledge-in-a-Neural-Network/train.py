import torch
import evaluate
import torch.nn.functional as F

def temperature_softmax(input_tensor, temperature=1.0):
    return torch.softmax(input_tensor / temperature, dim=1)


# Define teacher training function
def train_teacher(net, optimizer, criterion, trainloader, testloader, device, num_epochs=10):
    for epoch in range(num_epochs):
        net.train()	# Set model to training mode
        running_loss = 0.0
        for X, y in trainloader:
            X = X.to(device)
            y = y.to(device)
            # y_hat = net(X)
            y_hat = temperature_softmax(net(X))
            loss = criterion(y_hat, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # if epoch % 5 == 5 - 1:
        test_loss, accuracy = evaluate.evaluate_teacher(net, criterion, testloader, device)
        print(f"Epoch {epoch + 1}, loss = {running_loss / len(trainloader)}, test_loss = {test_loss}, accuracy = {accuracy}")


# Define student training function
def train_student(net, optimizer, criterion, trainloader,testloader, device, num_epochs=10):
    for epoch in range(num_epochs):
        net.train()	# Set model to training mode
        running_loss = 0.0
        for X, y in trainloader:
            X = X.to(device)
            y = y.to(device)
            # y_hat = net(X)
            y_hat = temperature_softmax(net(X))
            loss = criterion(y_hat, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # if epoch % 5 == 5 - 1:
        test_loss, accuracy = evaluate.evaluate_student(net, criterion, testloader, device)
        print(f"Epoch {epoch + 1}, loss = {running_loss / len(trainloader)}, test_loss = {test_loss}, accuracy = {accuracy}")


# Define distillation training function
def train_distill(teacher, student, optimizer_student, criterion_student, trainloader, testloader, device, num_epochs=10):
    for epoch in range(num_epochs):
        teacher.eval()
        student.train()
        running_loss = 0.0
        for X, y in trainloader:
            X = X.to(device)
            y = y.to(device)
            y_teacher = teacher(X)
            y_hat = student(X)

            l_soft = F.cross_entropy(temperature_softmax(y_hat, temperature=2), temperature_softmax(y_teacher, temperature=2))
            l_hard = criterion_student(temperature_softmax(y_hat), y)
            loss = l_soft + 0.5 * l_hard
            
            optimizer_student.zero_grad()
            loss.backward()
            optimizer_student.step()
            running_loss += loss.item()
        # if epoch % 5 == 5 - 1:
        test_loss, accuracy = evaluate.evaluate_student(student, criterion_student, testloader, device)
        print(f"Epoch {epoch + 1}, loss = {running_loss / len(trainloader)}, test_loss = {test_loss}, accuracy = {accuracy}")