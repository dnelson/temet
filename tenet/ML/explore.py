"""
* Misc ML exploration.
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
from os.path import isfile

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
from torch.utils.data.dataloader import default_collate

from ..util import simParams
from ..plot.config import *

path = '/u/dnelson/data/torch/'

def mnist_tutorial():
    """ Playing with the MNIST Fashion dataset. """

    # check gpu
    print(f'GPU is available: {torch.cuda.is_available()} [# devices = {torch.cuda.device_count()}]')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    labels = {0: "T-Shirt",
              1: "Trouser",
              2: "Pullover",
              3: "Dress",
              4: "Coat",
              5: "Sandal",
              6: "Shirt",
              7: "Sneaker",
              8: "Bag",
              9: "Ankle Boot"}

    # one-hot encoding transformation for labels
    target_trans = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))

    # download/load data
    training_data = datasets.FashionMNIST(root=path, train=True, download=True,
                                          transform=ToTensor(), target_transform=target_trans)
    test_data = datasets.FashionMNIST(root=path, train=False, download=True,
                                      transform=ToTensor(), target_transform=target_trans)

    # plot some images
    fig = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3

    for i in range(1, cols * rows + 1):
        sample_ind = torch.randint(len(training_data), size=(1,)).item()
        img, label_vec = training_data[sample_ind]
        label = labels[np.where(label_vec == 1)[0][0]]

        fig.add_subplot(rows, cols, i)
        plt.title(label)
        plt.axis("off")
        plt.imshow(img.squeeze(), origin='upper', cmap="gray")

    fig.savefig('mnist_tutorial.pdf')

    # create data loaders
    def collate_fn(x):
        """ Helper to automatically transfer vectors to the device, when loaded in a DataLoader. """
        return tuple(x_.to(device) for x_ in default_collate(x))
    
    batch_size = 64
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, 
                                  drop_last=False, pin_memory=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, 
                                 drop_last=False, pin_memory=False, collate_fn=collate_fn)

    #train_dataloader = DeviceDataLoader(train_dataloader, device=device)
    #test_dataloader = DeviceDataLoader(test_dataloader, device=device)

    print(f'Total training samples [{len(train_dataloader)}]. ', end='')
    print(f'For [{batch_size = }], number of training batches [{len(train_dataloader)}].')

    # load?
    if 0 and isfile('mnist_model.pth'):
        # load pickled model class, as well as the model weights
        model = torch.load('mnist_model.pth')
        print('Loaded: [mnist_model.pth].')
        # make sure to call model.eval() before inferencing to set the dropout and batch 
        # normalization laers to evaluate mode. otherwise, inconsistent inference results.
        # model.eval() # not for additional training?
    else:
        # instantiate i.e. initialize a new model
        model = mnist_network()

        if isfile('mnist_weights.pth'):
            model.load_state_dict(torch.load('mnist_weights.pth')) # load weights only
            print('Loaded: [mnist_weights.pth].')

    # move model to available device
    model.to(device)

    # inspect model structure
    print(model)

    #for name, param in model.named_parameters(): # also have .parameters()
    #    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

    # evaluate a forward pass (on a random input)
    if 0:
        X = torch.rand(1, 28, 28, device=device)
        logits = model(X)
        pred_probab = nn.Softmax(dim=1)(logits) # transform from [-inf,inf] to [0,1]
        y_pred = pred_probab.argmax(1) # choose largest probability as the predicted class
        print(f"Predicted class: {y_pred} [name = {labels[y_pred.item()]}]")

    # training hyperparameters
    learning_rate = 1e-3 # i.e. prefactor on grads for gradient descent
    batch_size = 64 # number of data samples propagated through the network before params are updated
    epochs = 5 # number of times to iterate over the dataset

    # loss function
    loss = nn.CrossEntropyLoss() # cross entropy = LogSoftMax and NLLLoss (negative log likelihood)

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # training loop
    for i in range(epochs):
        print(i)
        _train_model(train_dataloader, model, loss, optimizer, batch_size, device)
        _test_model(test_dataloader, model, loss, device)

    # save trained model
    torch.save(model.state_dict(), 'mnist_weights.pth')
    torch.save(model, 'mnist_model.pth')
    print('Saved: [mnist_model.pth] and [mnist_weights.pth].')

class DeviceDataLoader:
    """ https://github.com/pytorch/pytorch/issues/11372 """
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device
    
    def __len__(self):
        return len(self.dataloader)
    
    def __iter__(self):
        for batch in self.dataloader:
            yield tuple(tensor.to(self.device) for tensor in batch)

def _train_model(dataloader, model, loss_fn, optimizer, batch_size, device):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Move to device
        #X.to(device)
        #y.to(device)

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss = loss.item()
            current = batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def _test_model(dataloader, model, loss_fn, device):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            #X.to(device)
            #y.to(device)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(dim=1) == y.argmax(dim=1)).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test. Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}.\n")

class CustomDataset(Dataset):
    """ A custom dataset. Stores samples and their corresponding labels. (Not used). """
    def __init__(self, param1, param2, transform=None, target_transform=None):
        pass
        self.labels = [] # e.g. read from disk
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        """ Return a single sample at index i."""
        image = 0.0 # todo
        label = self.labels[i]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

class mnist_network(nn.Module):
    """ Playing with the MNIST Fashion dataset. """
    def __init__(self):
        super().__init__()

        # input pre-processing: 28x28 shape 2d arrays -> 784 shape 1d arrays
        self.flatten = nn.Flatten()

        # define layers
        # Sequential = ordered container of modules (data passed through each module in order)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512), # input/first layer, w*x + b
            nn.ReLU(), # non-linear activation
            nn.Linear(512, 512), # hidden layer, fully connected
            nn.ReLU(),
            nn.Linear(512, 10), # output layer
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
