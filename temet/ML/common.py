"""
* Common ML/pytorch functionality.
"""

import numpy as np
import torch


def train_model(dataloader, model, loss_fn, optimizer, batch_size, epoch_num, writer=None, verbose=True):
    """Train model for one epoch."""
    # Set the model to training mode - important for batch normalization and dropout layers
    model.train()

    for batch_num, data in enumerate(dataloader):
        inputs, labels = data

        # Zero (previous) gradients
        optimizer.zero_grad()

        # Compute prediction and loss
        pred = model(inputs)
        loss = loss_fn(pred, labels)

        # Backpropagation
        loss.backward()

        # Adjust weights
        optimizer.step()

        # print progress
        fac = np.max([1, int(len(dataloader) / 5)])
        if batch_num % fac == 0:
            loss = loss.item()
            current_sample = batch_num * batch_size + len(inputs)
            tot_samples = len(dataloader.dataset)
            s = f" loss: {loss:>7f}  [{current_sample:>5d}/{tot_samples:>5d}]"

            if writer is not None:
                current_sample += epoch_num * len(dataloader) * batch_size  # 'global [int] step value'
                s += f" global {current_sample = }"
                writer.add_scalars("loss", {"train": loss}, current_sample)

            if verbose:
                print(s)

    return loss


def test_model(dataloader, model, loss_fn, current_sample, acc_tol=None, writer=None, verbose=True):
    """Test model and compute statistics."""
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0.0
    correct = 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for inputs, labels in dataloader:
            pred = model(inputs)
            test_loss += loss_fn(pred, labels).item()

            # 'correct' number of predictions
            if str(acc_tol) == "exact":
                # exact match
                w_loc = pred == labels
            elif str(acc_tol) == "exact_onehot":
                # exact match for one-hot encoded labels
                w_loc = pred.argmax(dim=1) == labels.argmax(dim=1)
            else:
                # within |acc_tol| in the original (untransformed) space and units
                pred_untransformed = dataloader.dataset.target_invtransform(pred).squeeze()
                labels_untransformed = dataloader.dataset.target_invtransform(labels).squeeze()
                w_loc = torch.abs(pred_untransformed - labels_untransformed) <= acc_tol

            correct += w_loc.type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    s = f" test: accuracy [{(100 * correct):>0.1f}%], avg loss [{test_loss:>8f}]"

    if writer is not None:
        s += f" global {current_sample = }"
        writer.add_scalars("loss", {"test": test_loss}, current_sample)

    if verbose:
        print(s)

    return test_loss
