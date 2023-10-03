import torch
import torch.nn as nn
from torchsummary import summary
from torchvision.datasets import CIFAR10
import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm
import pandas as pd

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False).to(DEVICE)
# print(summary(model, (3, 244, 244)))

class LinearModel(nn.Module):
    def __init__(self, input_size=3*32*32, n_classes=2):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer = nn.Linear(input_size, n_classes)

    def forward(self, x):
        return self.layer(self.flatten(x))
        

def probability(counts):
    # counts = np.zeros((n_classes, n_classes, n_classes))
    # F_pred = np.argmax(F)
    # L_pred = np.argmax(L)

    probs = counts/np.sum(counts)
    return

def class_transform(label):
    return int(label > 4)

def get_data(batch_size):
    transform = transforms.ToTensor()
    
    trainset = CIFAR10(root='./data', train=True, download=True, transform=transform, target_transform=class_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = CIFAR10(root='./data', train=False, download=True, transform=transform, target_transform=class_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    
    

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader, classes
    
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def batch_loss(model, loss_func, xb, yb, optimizer = None):
    output = model(xb)
    loss = loss_func(output, yb)

    if optimizer is not None:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return loss.item(), len(xb)

def fit(conv_model, linear_model, epochs, loss_func, trainloader, testloader, optimizer, n_classes=2):
    
    for epoch in tqdm(range(epochs)):
        conv_model.train()
        linear_model.train()
        running_loss_conv = 0
        sample_nums = 0

        # for xb, yb in trainloader:
        #     xb = xb.to(DEVICE)
        #     yb = yb.to(DEVICE)
        #     loss_batch_conv, nums = batch_loss(conv_model, loss_func, xb, yb, optimizer)
        #     loss_batch_linear, _ = batch_loss(linear_model, loss_func, xb, yb, optimizer)

        #     running_loss_conv += loss_batch_conv * xb.size(0)
        #     sample_nums += nums
        #     loss_conv = running_loss_conv / sample_nums
        # print(
        #     f'EPOCH: {epoch+1:0>{len(str(epochs))}}/{epochs}',
        #     end=' '
        # )
        # print(f'LOSS: {loss_conv:.4f}',end=' ')

        
        conv_model.eval()
        linear_model.eval()

        # F, L, Y
        counts = np.zeros((n_classes, n_classes, n_classes))
        for xb, yb in testloader:
            xb = xb.to(DEVICE)
            conv_predictions = torch.argmax(conv_model(xb).cpu(), dim=1)
            linear_predictions = torch.argmax(linear_model(xb).cpu(), dim=1)
            for i in range(len(yb)):
                counts[conv_predictions[i], linear_predictions[i], yb[i]] += 1
            

        probs = counts/len(testloader.dataset)
        print(probs)


if __name__ == "__main__":
    batch_size = 64
    lr=0.001
    trainloader, testloader, classes = get_data(batch_size=batch_size)

    conv_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', num_classes=2, weights=None).to(DEVICE)
    linear_model = LinearModel().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(conv_model.parameters(), lr=lr)
    fit(conv_model=conv_model, linear_model=linear_model, epochs=10, loss_func=criterion, trainloader=trainloader, testloader=testloader, optimizer=optimizer)


    # # show some examples
    # dataiter = iter(trainloader)
    # images, labels = next(dataiter)
    # imshow(torchvision.utils.make_grid(images))
    # print(' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))