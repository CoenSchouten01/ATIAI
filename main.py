import torch
import torch.nn as nn
from torchvision.datasets import MNIST, CIFAR10
from torchvision.models.vision_transformer import VisionTransformer
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET = "CIFAR10"
MULTICLASS = True

class LinearModel(nn.Module):
    '''
    The linear model as used in section 3 of the paper "SGD on Neural Networks Learns Functions of Increasing Complexity" by Nakkiran et al.
    '''
    def __init__(self, input_size=1*28*28, n_classes=2):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer = nn.Linear(input_size, n_classes)

    def forward(self, x):
        return self.layer(self.flatten(x))

class ConvModel(nn.Module):
    '''
    The convulutional model as used in section 3 of the paper "SGD on Neural Networks Learns Functions of Increasing Complexity" by Nakkiran et al.
    '''
    def __init__(self, input_channels = 1, n_linear_in_features=32, n_hidden_channels=32, conv_kernel_size=3, pool_kernel_size = 2, dense_units = 2000, n_classes=2):
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
                                   nn.Linear(in_features=n_linear_in_features, out_features=dense_units),
                                   nn.ReLU(),
                                   nn.Linear(in_features=dense_units, out_features = n_classes)
        )
        self.last_layer = nn.Softmax()

    def forward(self, x):
        return self.model(x)


def mutual_information(probs, conditioning=False, epsilon=0.0001):
    '''
    This function calculates the mutual information between model 1 and Y,
    based on the conditioning argument this mutual information is either 
    conditioned on model 2 or not
    '''
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
    '''
    This function calculates the mutual information between model 1 and Y per class in Y,
    based on the conditioning argument this mutual information is either 
    conditioned on model 2 or not
    '''
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
          else:
            for l in range(n_classes):
              p_fly = probs[f, l, value]
              p_fl = joint_fl[f, l]
              p_ly = joint_ly[l, value]
              p_l = np.sum(joint_fl, axis=0)[l]
              p_fy_given_l = p_fly / (p_l + epsilon)
              p_f_given_l = p_fl / (p_l + epsilon)
              p_y_given_l = p_ly / (p_l + epsilon)
              mi += p_fly * np.log2((p_fy_given_l + epsilon) / (p_f_given_l * p_y_given_l + epsilon))
        mis.append(mi)
    return mis

def class_transform(label):
    '''
    This function is used to transform multiclass data to binary data
    by setting the first 5 classes to 0 and the other 5 to 1
    '''
    return int(label > 4)

def get_data(batch_size):
    '''
    This function returns the test and train dataloaders, based on the set dataset and
    wether or not multiclass is used
    '''
    transform = transforms.ToTensor()

    if DATASET == "MNIST":
        if MULTICLASS:
            trainset = MNIST(root='./data', train=True, download=True, transform=transform)
            testset = MNIST(root='./data', train=False, download=True, transform=transform)
        else:
            trainset = MNIST(root='./data', train=True, download=True, transform=transform, target_transform=class_transform)
            testset = MNIST(root='./data', train=False, download=True, transform=transform, target_transform=class_transform)
    elif DATASET == "CIFAR10":
        if MULTICLASS:
            trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
            testset = CIFAR10(root='./data', train=False, download=True, transform=transform)
        else:
            trainset = CIFAR10(root='./data', train=True, download=True, transform=transform, target_transform=class_transform)
            testset = CIFAR10(root='./data', train=False, download=True, transform=transform, target_transform=class_transform)
    else:
        raise Exception("Incorrect Dataset, must be either MNIST or CIFAR10")

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    return trainloader, testloader

def batch_loss(model, loss_func, xb, yb, optimizer = None):
    '''
    Calculate the batch loss for the given model using the given loss function
    Performs one backpropagation step using the given optimizer
    '''
    output = model(xb)
    loss = loss_func(output, yb)

    if optimizer is not None:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return loss.item(), len(xb)

def fit(model_1, model_2, optimizer_1, optimizer_2, epochs, loss_func, trainloader, testloader, n_classes):
    '''
    The main train-evaluation loop of the program
    '''

    # keep track of the mutual information and mu for the plots
    history_mi_train = []
    history_mi_test = []
    history_mu = []

    # keep track of the mutual information and mu per class for the plots
    history_mis_train = []
    history_mis_test = []
    history_mus = []

    # enter train mode
    model_1.eval()
    model_2.eval()

    # model 1, model 2, Y
    counts = np.zeros((n_classes, n_classes, n_classes))
    test_counts = np.zeros((n_classes, n_classes, n_classes))

    # perform initial evaluation before training
    for xb, yb in trainloader:
        xb = xb.to(DEVICE)
        m1_predictions = torch.argmax(model_1(xb).cpu(), dim=1)
        m2_predictions = torch.argmax(model_2(xb).cpu(), dim=1)
        for i in range(len(yb)):
            counts[m1_predictions[i], m2_predictions[i], yb[i]] += 1

    count_correct_1 = 0
    count_correct_2 = 0
    for xb, yb in testloader:
        xb = xb.to(DEVICE)
        m1_predictions = torch.argmax(model_1(xb).cpu(), dim=1)
        m2_predictions = torch.argmax(model_2(xb).cpu(), dim=1)
        for i in range(len(yb)):
            test_counts[m1_predictions[i], m2_predictions[i], yb[i]] += 1
            if yb[i] == m1_predictions[i]:
              count_correct_1 += 1
            if yb[i] == m2_predictions[i]:
              count_correct_2 += 1

    # calculate the emprical probabilities
    probs_train = counts/len(trainloader.dataset)
    probs_test = test_counts/len(testloader.dataset)

    # calculate the mutual information and the mu
    mi_fy_train = mutual_information(probs_train, False)
    mi_fy_test = mutual_information(probs_test, False)
    mu = mi_fy_test - mutual_information(probs_test, True)

    # update the histories
    history_mi_train.append(mi_fy_train)
    history_mi_test.append(mi_fy_test)
    history_mu.append(mu)

    # repeatedly train and evaluate for the indicated number of epochs
    for epoch in tqdm(range(epochs)):
        model_1.train()
        model_2.train()
        running_loss_m1 = 0
        running_loss_m2 = 0
        sample_nums = 0

        # perform a training epoch
        for xb, yb in trainloader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            loss_batch_m1, nums = batch_loss(model_1, loss_func, xb, yb, optimizer_1)
            loss_batch_m2, _ = batch_loss(model_2, loss_func, xb, yb, optimizer_2)

            sample_nums += nums
            running_loss_m1 += loss_batch_m1 * xb.size(0)
            loss_m1 = running_loss_m1 / sample_nums

            running_loss_m2 += loss_batch_m2 * xb.size(0)
            loss_m2 = running_loss_m2 / sample_nums

        print(
            f'EPOCH: {epoch+1:0>{len(str(epochs))}}/{epochs}',
            end=' '
        )
        print(f'LOSS: {loss_m1:.4f}, {loss_m2:.4f} \n',end=' ')

        model_1.eval()
        model_2.eval()

        # model_1, model_2, Y
        counts = np.zeros((n_classes, n_classes, n_classes))
        test_counts = np.zeros((n_classes, n_classes, n_classes))

        for xb, yb in trainloader:
            xb = xb.to(DEVICE)
            m1_predictions = torch.argmax(model_1(xb).cpu(), dim=1)
            m2_predictions = torch.argmax(model_2(xb).cpu(), dim=1)
            for i in range(len(yb)):
                counts[m1_predictions[i], m2_predictions[i], yb[i]] += 1

        count_correct_1 = 0
        count_correct_2 = 0
        for xb, yb in testloader:
            xb = xb.to(DEVICE)
            m1_predictions = torch.argmax(model_1(xb).cpu(), dim=1)
            m2_predictions = torch.argmax(model_2(xb).cpu(), dim=1)
            for i in range(len(yb)):
                test_counts[m1_predictions[i], m2_predictions[i], yb[i]] += 1
                if yb[i] == m1_predictions[i]:
                  count_correct_1 += 1
                if yb[i] == m2_predictions[i]:
                  count_correct_2 += 1

        # print the accuracies
        print(count_correct_1/len(testloader.dataset))
        print(count_correct_2/len(testloader.dataset))

        # calculate the emprical probabilities
        probs_train = counts/len(trainloader.dataset)
        probs_test = test_counts/len(testloader.dataset)

        # calculate the mutual information and the mu
        mi_fy_train = mutual_information(probs_train, False)
        mi_fy_test = mutual_information(probs_test, False)
        mu = mi_fy_test - mutual_information(probs_test, True)
        
        # update the histories
        history_mi_train.append(mi_fy_train)
        history_mi_test.append(mi_fy_test)
        history_mu.append(mu)

        # calculate the mutual information and the mu per class
        mis_fy_train = mutual_information_per_class(probs_train, False)
        mis_fy_test = mutual_information_per_class(probs_test, False)
        mus = np.array(mis_fy_test) - np.array(mutual_information_per_class(probs_test, True))
        
        # update the histories
        history_mis_train.append(mis_fy_train)
        history_mis_test.append(mis_fy_test)
        history_mus.append(mus)
    return history_mi_train, history_mi_test, history_mu, history_mis_train, history_mis_test, history_mus


def plot_results(history_mi_train, history_mi_test, history_mu, history_mis_train, history_mis_test, history_mus):
    
    mi_train_mean = np.mean(history_mi_train, axis=0)
    mi_test_mean = np.mean(history_mi_test, axis=0)
    mu_mean = np.mean(history_mu, axis=0)
    mis_train_mean = np.mean(history_mis_train, axis=0)
    mis_test_mean = np.mean(history_mis_test, axis=0)
    mus_mean = np.mean(history_mus, axis=0)

    mi_train_std = np.std(history_mi_train, axis=0)
    mi_test_std = np.std(history_mi_test, axis=0)
    mu_std = np.std(history_mu, axis=0)
    mis_train_std = np.std(history_mis_train, axis=0)
    mis_test_std = np.std(history_mis_test, axis=0)
    mus_std = np.std(history_mus, axis=0)

    # plot the mutual information over all classes on the train and test set, as well as the mu
    plt.plot(mi_train_mean, label = "MI train")
    plt.fill_between(x=np.arange(len(mi_train_mean)), y1=mi_train_mean-mi_train_std, y2=mi_train_mean+mi_train_std, alpha=0.2)
    plt.plot(mi_test_mean, label = "MI test")
    plt.fill_between(x=np.arange(len(mi_test_mean)), y1=mi_test_mean-mi_test_std, y2=mi_test_mean+mi_test_std, alpha=0.2)
    plt.plot(mu_mean, label = "mu")
    plt.fill_between(x=np.arange(len(mu_mean)), y1=mu_mean-mu_std, y2=mu_mean+mu_std, alpha=0.2)
    plt.legend()
    plt.title("MNIST Transformer and Linear")
    plt.xlabel("Epochs")
    plt.ylabel("Mutual Information")
    plt.savefig(DATASET + "-transformer-" + "all_classes" + ".png")
    plt.show()

    if MULTICLASS:
        if DATASET == "MNIST":
            classes = list(range(10))
        elif DATASET == "CIFAR10":
            classes = ['plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        else:
            raise Exception("Incorrect Dataset, must be either MNIST or CIFAR10")
            
        # plot the metrics per class in the data
        for i in range(n_classes):
            plt.title(f"MNIST Transformer and Linear: class {i}")
            plt.xlabel("Epochs")
            plt.ylabel("Mutual Information")
            mis_train_mean_i = np.array(mis_train_mean)[:,i]
            mis_train_std_i = np.array(mis_train_std)[:,i]
            plt.plot(mis_train_mean_i, label="MI Train")
            plt.fill_between(x=np.arange(len(mis_train_mean_i)), y1=mis_train_mean_i-mis_train_std_i, y2=mis_train_mean_i+mis_train_std_i, alpha=0.2)
            
            mis_test_mean_i = np.array(mis_test_mean)[:,i]
            mis_test_std_i = np.array(mis_test_std)[:,i]
            plt.plot(mis_test_mean_i, label = "MI test")
            plt.fill_between(x=np.arange(len(mis_test_mean_i)), y1=mis_test_mean_i-mis_test_std_i, y2=mis_test_mean_i+mis_test_std_i, alpha=0.2)

            mus_mean_i = np.array(mus_mean)[:,i]
            mus_std_i = np.array(mus_std)[:,i]
            plt.plot(mus_mean_i, label = "mu")
            plt.fill_between(x=np.arange(len(mus_mean_i)), y1=mus_mean_i-mus_std_i, y2=mus_mean_i+mus_std_i, alpha=0.2)
            plt.legend()
            plt.savefig(DATASET + "-transformer-" + "class" + str(classes[i]) + ".png")
            plt.show()
        
        # plot all mutual informations in one plot
        for i in range(n_classes):
            mis_test_mean_i = np.array(mis_test_mean)[:,i]
            mis_test_std_i = np.array(mis_test_std)[:,i]
            plt.plot(mis_test_mean_i, label = classes[i])
            plt.fill_between(x=np.arange(len(mis_test_mean_i)), y1=mis_test_mean_i-mis_test_std_i, y2=mis_test_mean_i+mis_test_std_i, alpha=0.2)
        
        plt.title("MNIST Transformer and Linear: MI all classes")
        plt.xlabel("Epochs")
        plt.ylabel("Mutual Information")
        plt.legend()
        plt.savefig(DATASET + "-transformer-" + "all_mis" + ".png")
        plt.show()

        # plot all mus in one plot
        for i in range(n_classes):
            mus_mean_i = np.array(mus_mean)[:,i]
            mus_std_i = np.array(mus_std)[:,i]
            plt.plot(mus_mean_i, label = classes[i])
            plt.fill_between(x=np.arange(len(mus_mean_i)), y1=mus_mean_i-mus_std_i, y2=mus_mean_i+mus_std_i, alpha=0.2)
            
        plt.title("MNIST Transformer and Linear: mu all classes")
        plt.xlabel("Epochs")
        plt.ylabel("Mutual Information")
        plt.legend()
        plt.savefig(DATASET + "-transformer-" + "all_mus" + ".png")
        plt.show()

if __name__ == "__main__":
    # hyperparameters
    n_experiments = 2
    batch_size = 64
    epochs = 2
    
    # fetch the training data
    trainloader, testloader = get_data(batch_size=batch_size)

    if MULTICLASS:
        n_classes = 10
    else: 
        n_classes = 2    

    history_mi_train = []
    history_mi_test = []
    history_mu = []
    history_mis_train = []
    history_mis_test = []
    history_mus = []

    for _ in range(n_experiments):
        # initialize the models
        if DATASET == "MNIST":
            linear_model = LinearModel(input_size=1*28*28, n_classes=n_classes).to(DEVICE)
            conv_model = ConvModel(input_channels=1, n_linear_in_features=32, n_classes=n_classes).to(DEVICE)
            transformer_model = VisionTransformer(image_size = 28,
                patch_size = 4,
                num_layers = 4,
                num_heads = 4,
                hidden_dim = 64,
                mlp_dim = 128,
                num_classes = n_classes)
            transformer_model.conv_proj = nn.Conv2d(
                        in_channels=1, out_channels=64, kernel_size=4, stride=4
                    )
        elif DATASET == "CIFAR10":
            linear_model = LinearModel(input_size=3*32*32, n_classes=n_classes).to(DEVICE)
            conv_model = ConvModel(input_channels=3, n_linear_in_features=128, n_classes=n_classes).to(DEVICE)
            transformer_model = VisionTransformer(image_size = 32,
                patch_size = 4,
                num_layers = 4,
                num_heads = 4,
                hidden_dim = 64,
                mlp_dim = 128,
                num_classes = n_classes)
            transformer_model.conv_proj = nn.Conv2d(
                        in_channels=3, out_channels=64, kernel_size=4, stride=4
                    )
        else: 
            raise Exception("Incorrect Dataset, must be either MNIST or CIFAR10")
        
        transformer_model.to(DEVICE)

        # initialize criterion and optimizers
        criterion = nn.CrossEntropyLoss()
        linear_optimizer = optim.SGD(linear_model.parameters(), lr=0.01)
        conv_optimizer = optim.SGD(conv_model.parameters(), lr=0.001)
        transformer_optimizer = optim.SGD(transformer_model.parameters(), lr=0.001)

        # call fit on the desired configuration
        mi_train, mi_test, mu, mis_train, mis_test, mus = fit(model_1=conv_model, model_2=linear_model, optimizer_1=conv_optimizer, 
            optimizer_2=linear_optimizer, epochs=epochs, loss_func=criterion, trainloader=trainloader, testloader=testloader, 
            n_classes=n_classes)
        
        history_mi_train.append(mi_train)
        history_mi_test.append(mi_test)
        history_mu.append(mu)
        history_mis_train.append(mis_train)
        history_mis_test.append(mis_test)
        history_mus.append(mus)
        
    plot_results(history_mi_train, history_mi_test, history_mu, history_mis_train, history_mis_test, history_mus)
