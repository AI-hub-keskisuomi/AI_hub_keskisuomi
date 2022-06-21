import torch
import numpy as np


def test_model(model, dataloader, device, criterion=None):
    """Test the model with data

    Args:
        model (nn model): pytorch model
        dataloader (dataloader): pytorch dataloader to test the model with
        device (string): the device to use (cpu or cuda)
        criterion (loss function, optional): Loss function to use. Defaults to None.

    Returns:
        dictionary: results for model predictions, true labels, corrects, misslabeled, loss if criterion is given
    """

    correct = 0
    total = 0

    y_pred = []
    y_true = []

    running_loss = 0.0
    dataset_size = 0

    misclassified = {"images": [], "labels": [], "preds": []}

    model.eval()

    # if loss function is provided calculate loss also
    if criterion != None:
        calc_loss = True
    else:
        calc_loss = False

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():

        for batch in dataloader:
            images = batch["image"]
            labels = batch["target"]

            dataset_size += len(images)

            labels = labels.reshape(
                -1,
            )  # reshape to a vector

            # sent to device
            images = images.to(device)
            labels = labels.to(device)

            # calculate outputs by running images through the network
            outputs = model.forward(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)

            if calc_loss:
                # calculate loss
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)

            # collect misclassified images
            misclassified_mask = np.not_equal(predicted.cpu(), labels.cpu()).bool()
            misclassified["images"].extend(images[misclassified_mask].cpu())
            misclassified["labels"].extend(labels[misclassified_mask].cpu().numpy())
            misclassified["preds"].extend(predicted[misclassified_mask].cpu().numpy())

            # collect predictions
            y_pred.extend(predicted.cpu())
            y_true.extend(labels.cpu())

            # total samples tested
            total += labels.size(0)
            # total samples predicted correctly
            correct += (predicted == labels).sum().item()

    # results
    res = {
        "total_samples": total,
        "correct_samples": correct,
        "true_labels": y_true,
        "pred_labels": y_pred,
        "misclassified": misclassified,
    }

    if calc_loss:
        res["loss"] = running_loss / dataset_size

    return res


def diagnose_model(model, dataloader, device):

    y_pred = []
    y_true = []

    correct = {"images": [], "labels": [], "preds": []}

    misclassified = {"images": [], "labels": [], "preds": []}

    model.eval()

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():

        for batch in dataloader:
            images = batch["image"]
            labels = batch["target"]

            labels = labels.reshape(
                -1,
            )  # reshape to a vector

            # sent to device
            images = images.to(device)
            labels = labels.to(device)

            # calculate outputs by running images through the network
            outputs = model.forward(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)

            # collect predictions
            y_pred.extend(predicted.cpu())
            y_true.extend(labels.cpu())

            # collect correctly classified images
            correct_mask = np.equal(predicted.cpu(), labels.cpu()).bool()
            correct["images"].extend(images[correct_mask].cpu())
            correct["labels"].extend(labels[correct_mask].cpu().numpy())
            correct["preds"].extend(predicted[correct_mask].cpu().numpy())

            # collect misclassified images
            misclassified_mask = np.not_equal(predicted.cpu(), labels.cpu()).bool()
            misclassified["images"].extend(images[misclassified_mask].cpu())
            misclassified["labels"].extend(labels[misclassified_mask].cpu().numpy())
            misclassified["preds"].extend(predicted[misclassified_mask].cpu().numpy())

    # results
    res = {
        "true_labels": y_true,
        "pred_labels": y_pred,
        "correct": correct,
        "misclassified": misclassified,
    }

    return res
