import torch
import torch.nn as nn



# Define evaluation function
def evaluate(net, test_iter, device, criterion=nn.CrossEntropyLoss()):
    net.eval()
    eval_loss = 0
    eval_acc = 0
    for inputs, labels in test_iter:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        eval_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        eval_acc += (preds == labels).float().mean()
    return eval_loss / len(test_iter), eval_acc / len(test_iter)
    print(f"Test Loss: {running_loss / len(test_iter):.4f}, Accuracy: {100. * correct / total:.2f}%")
