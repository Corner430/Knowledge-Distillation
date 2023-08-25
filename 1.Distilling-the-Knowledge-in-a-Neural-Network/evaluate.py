import torch


def temperature_softmax(input_tensor, temperature=1.0):
    return torch.softmax(input_tensor / temperature, dim=1)


# Define teacher evaluation function
def evaluate_teacher(net, criterion, testloader, device):
    net.eval()	# Set model to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in testloader:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            y_hat = temperature_softmax(net(X),) 
            loss = criterion(y_hat, y)
            
            running_loss += loss.item()
            _, predicted = y_hat.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    return (running_loss / len(testloader), correct / total)


# Define student evaluation function
def evaluate_student(net, criterion, testloader, device):
    net.eval()	# Set model to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in testloader:
            X = X.to(device)
            y = y.to(device)
            # y_hat = net(X)
            y_hat = temperature_softmax(net(X))
            loss = criterion(y_hat, y)
            
            running_loss += loss.item()
            _, predicted = y_hat.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    return running_loss / len(testloader), correct / total