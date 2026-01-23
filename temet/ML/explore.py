"""
* Misc ML exploration.
"""

from os.path import isfile

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision.transforms import Lambda, ToTensor
from torchvision.utils import make_grid

from ..ML.common import test_model, train_model


path = "/u/dnelson/data/torch/"


class mnist_network(nn.Module):
    """Simple NN to play with the MNIST Fashion dataset."""

    def __init__(self):  # noqa: D107
        super().__init__()

        # input pre-processing: 28x28 shape 2d arrays -> 784 shape 1d arrays
        self.flatten = nn.Flatten()

        # define layers
        # Sequential = ordered container of modules (data passed through each module in order)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),  # input/first layer, w*x + b
            nn.ReLU(),  # non-linear activation
            nn.Linear(512, 512),  # hidden layer, fully connected
            nn.ReLU(),
            nn.Linear(512, 10),  # output layer
        )

    def forward(self, x):
        """Forward pass."""
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def mnist_tutorial():
    """Playing with the MNIST Fashion dataset."""
    torch.manual_seed(424242)

    # check gpu
    print(f"GPU is available: {torch.cuda.is_available()} [# devices = {torch.cuda.device_count()}]")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    labels = {
        0: "T-Shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot",
    }

    # one-hot encoding transformation for labels
    target_trans = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))

    # download/load data
    training_data = datasets.FashionMNIST(
        root=path, train=True, download=True, transform=ToTensor(), target_transform=target_trans
    )
    test_data = datasets.FashionMNIST(
        root=path, train=False, download=True, transform=ToTensor(), target_transform=target_trans
    )

    # plot some images
    if 1:
        fig = plt.figure(figsize=(8, 8))
        cols, rows = 3, 3

        for i in range(1, cols * rows + 1):
            sample_ind = torch.randint(len(training_data), size=(1,)).item()
            img, label_vec = training_data[sample_ind]
            label = labels[np.where(label_vec == 1)[0][0]]

            fig.add_subplot(rows, cols, i)
            plt.title(label)
            plt.axis("off")
            plt.imshow(img.squeeze(), origin="upper", cmap="gray")

        fig.savefig("mnist_tutorial.pdf")

    # create data loaders
    def collate_fn(x):
        """Helper to automatically transfer vectors to the device, when loaded in a DataLoader."""
        return tuple(x_.to(device) for x_ in default_collate(x))

    batch_size = 64
    train_dataloader = DataLoader(
        training_data, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=False, collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        test_data, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=False, collate_fn=collate_fn
    )

    print(f"Total training samples [{len(training_data)}]. ", end="")
    print(f"For [{batch_size = }], number of training batches [{len(train_dataloader)}].")

    # tensorboard
    writer = None

    writer = SummaryWriter(path + "runs/fashion_mnist")

    if 1:
        # tensorboard: add some images
        imgs = [training_data[i][0] for i in range(20)]
        img_grid = make_grid(imgs)

        writer.add_image("fashion_mnist_images", img_grid)
        writer.flush()

    if 1:
        # tensorboard: visualize image embeddings
        images = training_data.data[0:10]
        labels = training_data.data[0:10]
        features = images.view(-1, 28 * 28)

        writer.add_embedding(features, metadata=labels, label_img=images.unsqueeze(1))
        writer.flush()

    # load?
    if 0 and isfile("mnist_model.pth"):
        # load pickled model class, as well as the model weights
        model = torch.load("mnist_model.pth")
        print("Loaded: [mnist_model.pth].")
        # make sure to call model.eval() before inferencing to set the dropout and batch
        # normalization laers to evaluate mode. otherwise, inconsistent inference results.
        # model.eval() # not for additional training?
    else:
        # instantiate i.e. initialize a new model
        model = mnist_network()

        if isfile("mnist_weights.pth"):
            model.load_state_dict(torch.load("mnist_weights.pth"))  # load weights only
            print("Loaded: [mnist_weights.pth].")

    # move model to available device
    model.to(device)

    # inspect model structure
    print(model)

    # for name, param in model.named_parameters(): # also have .parameters()
    #    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

    # evaluate a forward pass (on a random input)
    if 0:
        X = torch.rand(1, 28, 28, device=device)
        logits = model(X)
        pred_probab = nn.Softmax(dim=1)(logits)  # transform from [-inf,inf] to [0,1]
        y_pred = pred_probab.argmax(1)  # choose largest probability as the predicted class
        print(f"Predicted class: {y_pred} [name = {labels[y_pred.item()]}]")

    # training hyperparameters
    learning_rate = 1e-3  # i.e. prefactor on grads for gradient descent
    epochs = 25  # number of times to iterate over the dataset

    # loss function
    loss_f = nn.CrossEntropyLoss()  # cross entropy = LogSoftMax and NLLLoss (negative log likelihood)

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # training loop
    test_loss_best = np.inf

    for i in range(epochs):
        # train and report loss
        train_loss = train_model(train_dataloader, model, loss_f, optimizer, batch_size, i, writer=writer)

        print(f"\nEpoch: [{i}] | Train Loss: {train_loss:.4f}")

        # test
        test_loss = test_model(
            test_dataloader,
            model,
            loss_f,
            current_sample=(i + 1) * len(training_data),
            acc_tol="exact_onehot",
            writer=writer,
        )

        # periodically save trained model (should put epoch number into filename)
        if test_loss < test_loss_best:
            torch.save(model.state_dict(), "mnist_weights.pth")
            torch.save(model, "mnist_model.pth")
            print(" saved: [mnist_model.pth] and [mnist_weights.pth].")
            test_loss_best = test_loss

    if writer is not None:
        writer.close()
