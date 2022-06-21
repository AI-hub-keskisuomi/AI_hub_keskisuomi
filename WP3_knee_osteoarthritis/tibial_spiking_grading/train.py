import copy
import time

import torch


def train_model(
    model,
    dataloaders,
    image_datasets,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs,
):
    """Train specified model with train data, evaluate with validation. Runs for specified epochs.

    Args:
        model (pytorch model): pytorch cnn model
        dataloaders (pytorch dataloaders): train and validation dataloaders
        image_datasets (pytorch dataset): train and validataion datasets
        criterion (loss function): specified loss function
        optimizer (optimizer): optimizer to minimize loss function
        scheduler (pytorch scheduler): scheduler adjusts the learning-rate during training
        device (string): device to use (cpu or cuda)
        num_epochs (integer): specifies how many iterations over the train data should the training run

    Returns:
        model weigths: model weights are returned for the top iteration (epoch) selected by validation accuracy
    """

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_train_epoch = 0

    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val", "test"]}

    for epoch in range(num_epochs):

        if epoch % 5 == 0:
            print("\nEpoch {}/{}".format(epoch, num_epochs - 1))
            print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data in batches
            for batch in dataloaders[phase]:
                inputs = batch["image"]
                labels = batch["target"]

                labels = labels.reshape(
                    -1,
                )  # reshape to a vector

                # sent to device
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    # outputs = model(inputs)
                    outputs = model.forward(inputs)

                    # get predictions
                    _, preds = torch.max(outputs, 1)

                    # calculate loss
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if epoch % 5 == 0:
                print(
                    "{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc)
                )

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_train_epoch = epoch + 1  # epochs start from 0

    time_elapsed = time.time() - since
    print(
        "\nTraining complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))
    print("Best train epoch:", best_train_epoch)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
