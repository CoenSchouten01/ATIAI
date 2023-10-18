import torch
import torch.nn as nn
from torchsummary import summary
from torchvision.datasets import MNIST
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
    def __init__(self, input_size=1*28*28, n_classes=2):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer = nn.Linear(input_size, n_classes)

    def forward(self, x):
        return self.layer(self.flatten(x))

class ConvModel(nn.Module):
    def __init__(self, input_channels = 1, n_hidden_channels=32, conv_kernel_size=3, pool_kernel_size = 2, dense_units = 2000, n_classes=2):
        super().__init__()
        self.model = nn.Sequential(nn.Conv2d(in_channels=input_channels, out_channels=n_hidden_channels, kernel_size=conv_kernel_size),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size = pool_kernel_size),
                                   nn.Conv2d(in_channels=n_hidden_channels, out_channels=n_hidden_channels, kernel_size=conv_kernel_size),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size = pool_kernel_size),
                                   nn.Conv2d(in_channels=n_hidden_channels, out_channels=n_hidden_channels, kernel_size=conv_kernel_size),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=n_hidden_channels, out_channels=n_hidden_channels, kernel_size=conv_kernel_size),
                                   nn.ReLU(),
                                   nn.Flatten(),
                                   nn.Linear(in_features=32, out_features=dense_units),
                                   nn.ReLU(),
                                   nn.Linear(in_features=dense_units, out_features = n_classes)
        )
        self.last_layer = nn.Softmax()

    def forward(self, x):
        return self.model(x)


def mutual_information(probs, conditioning=False, epsilon=0.0001):
    # Calculate I()
    n_classes = probs.shape[0]
    mi = 0

    joint_ly = np.sum(probs, axis=0)
    joint_fy = np.sum(probs, axis=1)
    joint_fl = np.sum(probs, axis=2)

    for f in range(n_classes):
        p_f = np.sum(joint_fy, axis=1)[f]
        for y in range(n_classes):
            p_fy = joint_fy[f, y]
            p_y = np.sum(joint_fy, axis=0)[y]
            if conditioning == False:
                # print(p_fy, p_f, p_y, epsilon)
                mi += p_fy * np.log2((p_fy + epsilon) / (p_f * p_y + epsilon))
            else:
                for l in range(n_classes):
                    p_fly = probs[f, l, y]
                    p_fl = joint_fl[f, l]
                    p_ly = joint_ly[l, y]
                    p_l = np.sum(joint_fl, axis=0)[l]
                    p_fy_given_l = p_fly / (p_l + epsilon)
                    p_f_given_l = p_fl / (p_l + epsilon)
                    p_y_given_l = p_ly / (p_l + epsilon)
                    mi += p_fly * np.log2((p_fy_given_l + epsilon) / (p_f_given_l * p_y_given_l + epsilon))
    return mi

def mutual_information_per_class(probs, conditioning=False, epsilon=0.0001):
    n_classes = probs.shape[0]
    mis = []

    joint_ly = np.sum(probs, axis=0)
    joint_fy = np.sum(probs, axis=1)
    joint_fl = np.sum(probs, axis=2)

    for value in range(n_classes):
        mi = 0
        p_y= np.sum(joint_fy, axis=0)[value]
        for f in range(n_classes):
          p_f = np.sum(joint_fy, axis=1)[f]
          p_fy = joint_fy[f, value]

          if conditioning == False:
              mi += p_fy * np.log2((p_fy + epsilon) / (p_f * p_y + epsilon))
          else: # TODO
              p_fly = probs[value, value, value]
              p_fl = joint_fl[value, value]
              p_ly = joint_ly[value, value]
              p_l = np.sum(joint_fl, axis=0)[value]
              p_fy_given_l = p_fly / (p_l + epsilon)
              p_f_given_l = p_fl / (p_l + epsilon)
              p_y_given_l = p_ly / (p_l + epsilon)
              mi += p_fly * np.log2((p_fy_given_l + epsilon) / (p_f_given_l * p_y_given_l + epsilon))
              mi += (1 - p_fly) * np.log2((1 - p_fy_given_l + epsilon) / ((1 - p_f_given_l) * (1 - p_y_given_l) + epsilon))
        mis.append(mi)
    return mis

def class_transform(label):
    return int(label > 4)

def get_data(batch_size):
    transform = transforms.ToTensor()

    trainset = MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = MNIST(root='./data', train=False, download=True, transform=transform)
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

def fit(conv_model, linear_model, epochs, loss_func, trainloader, testloader, conv_optimizer, linear_optimizer, n_classes=2):

    history_mi_train = []
    history_mi_test = []
    history_mu = []

    history_mis_train = []
    history_mis_test = []
    history_mus = []

    conv_model.eval()
    linear_model.eval()

    # F, L, Y
    counts = np.zeros((n_classes, n_classes, n_classes))
    test_counts = np.zeros((n_classes, n_classes, n_classes))

    for xb, yb in trainloader:
        xb = xb.to(DEVICE)
        conv_predictions = torch.argmax(conv_model(xb).cpu(), dim=1)
        linear_predictions = torch.argmax(linear_model(xb).cpu(), dim=1)
        for i in range(len(yb)):
            counts[conv_predictions[i], linear_predictions[i], yb[i]] += 1

    count_correct = 0
    count_corrects = 0
    for xb, yb in testloader:
        xb = xb.to(DEVICE)
        conv_predictions = torch.argmax(conv_model(xb).cpu(), dim=1)
        linear_predictions = torch.argmax(linear_model(xb).cpu(), dim=1)
        for i in range(len(yb)):
            test_counts[conv_predictions[i], linear_predictions[i], yb[i]] += 1
            if yb[i] == linear_predictions[i]:
              count_correct += 1
            if yb[i] == conv_predictions[i]:
              count_corrects += 1

    # print(count_correct/len(testloader.dataset))
    # print(count_corrects/len(testloader.dataset))


    probs_train = counts/len(trainloader.dataset)
    probs_test = test_counts/len(testloader.dataset)
    mi_fy_train = mutual_information(probs_train, False)
    mi_fy_test = mutual_information(probs_test, False)
    mu = mi_fy_test - mutual_information(probs_test, True)
    history_mi_train.append(mi_fy_train)
    history_mi_test.append(mi_fy_test)
    history_mu.append(mu)

    for epoch in tqdm(range(epochs)):
        conv_model.train()
        linear_model.train()
        running_loss_conv = 0
        running_loss_linear = 0
        sample_nums = 0

        for xb, yb in trainloader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            loss_batch_conv, nums = batch_loss(conv_model, loss_func, xb, yb, conv_optimizer)
            loss_batch_linear, _ = batch_loss(linear_model, loss_func, xb, yb, linear_optimizer)

            running_loss_conv += loss_batch_conv * xb.size(0)
            sample_nums += nums
            loss_conv = running_loss_conv / sample_nums

            running_loss_linear += loss_batch_linear * xb.size(0)
            sample_nums += nums
            loss_linear = running_loss_linear / sample_nums
        print(
            f'EPOCH: {epoch+1:0>{len(str(epochs))}}/{epochs}',
            end=' '
        )
        print(f'LOSS: {loss_conv:.4f}, {loss_linear:.4f} \n',end=' ')


        conv_model.eval()
        linear_model.eval()

        # F, L, Y
        counts = np.zeros((n_classes, n_classes, n_classes))
        test_counts = np.zeros((n_classes, n_classes, n_classes))

        for xb, yb in trainloader:
            xb = xb.to(DEVICE)
            conv_predictions = torch.argmax(conv_model(xb).cpu(), dim=1)
            linear_predictions = torch.argmax(linear_model(xb).cpu(), dim=1)
            for i in range(len(yb)):
                counts[conv_predictions[i], linear_predictions[i], yb[i]] += 1

        count_correct = 0
        count_corrects = 0
        for xb, yb in testloader:
            xb = xb.to(DEVICE)
            conv_predictions = torch.argmax(conv_model(xb).cpu(), dim=1)
            linear_predictions = torch.argmax(linear_model(xb).cpu(), dim=1)
            for i in range(len(yb)):
                test_counts[conv_predictions[i], linear_predictions[i], yb[i]] += 1
                if yb[i] == linear_predictions[i]:
                  count_correct += 1
                if yb[i] == conv_predictions[i]:
                  count_corrects += 1

        print(count_correct/len(testloader.dataset))
        print(count_corrects/len(testloader.dataset))


        probs_train = counts/len(trainloader.dataset)
        probs_test = test_counts/len(testloader.dataset)
        mi_fy_train = mutual_information(probs_train, False)
        mi_fy_test = mutual_information(probs_test, False)
        mu = mi_fy_test - mutual_information(probs_test, True)
        history_mi_train.append(mi_fy_train)
        history_mi_test.append(mi_fy_test)
        history_mu.append(mu)

        mis_fy_train = mutual_information_per_class(probs_train, False)
        mis_fy_test = mutual_information_per_class(probs_test, False)
        #mus = np.array(mis_fy_test) - np.array(mutual_information_per_class(probs_test, True))
        history_mis_train.append(mis_fy_train)
        history_mis_test.append(mis_fy_test)
        #history_mus.append(mus)

    plt.plot(history_mi_train)
    plt.plot(history_mi_test)
    plt.plot(history_mu)
    plt.legend(['MI train', 'MI test', 'mu'])
    plt.show()

    for i in range(n_classes):
        plt.title(f"Class number : {i}")
        plt.plot(np.array(history_mis_train)[:,i])
        plt.plot(np.array(history_mis_test)[:,i])
        #plt.plot(np.array(history_mus)[:,i])
        plt.legend(['MI train', 'MI test', 'mu'])
        plt.show()

    for i in range(n_classes):
        plt.plot(np.array(history_mis_test)[:,i])

    plt.legend(['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5',
                'class 6', 'class 7', 'class 8', 'class 9'])
    # plt.legend(['MI train', 'MI test', 'mu'])
    plt.show()

if __name__ == "__main__":
    batch_size = 64
    lr=0.0001
    trainloader, testloader, classes = get_data(batch_size=batch_size)

    # conv_model = torchvision.models.resnet18(num_classes=2, weights=None)
    conv_model = ConvModel(n_classes=10)
    conv_model.to(DEVICE)

    # MNIST has 1 channel whereas regular ResNet expects 3
    # conv_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # conv_model.to(DEVICE)
    linear_model = LinearModel(n_classes=10).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    conv_optimizer = optim.SGD(conv_model.parameters(), lr=0.001)
    linear_optimizer = optim.SGD(linear_model.parameters(), lr=0.01)
    fit(conv_model=conv_model, linear_model=linear_model, epochs=30, loss_func=criterion, trainloader=trainloader, testloader=testloader, conv_optimizer=conv_optimizer, linear_optimizer=linear_optimizer, n_classes=10)



    # # show some examples
    # dataiter = iter(trainloader)
    # images, labels = next(dataiter)
    # imshow(torchvision.utils.make_grid(images))
    # print(' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))