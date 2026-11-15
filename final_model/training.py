import torch


def train_one_epoch(model, loader, loss_function, optimizer, device):

    model.train()

    running_loss = 0
    correct_predictions = 0
    total_predictions = 0

    for tensors, metadata, labels in loader:
        tensors, metadata, labels = tensors.to(device), metadata.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(tensors, metadata) 
        loss = loss_function(outputs, labels.long())  

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * tensors.size(0)
        predicted = torch.argmax(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)

    epoch_loss = running_loss / total_predictions
    epoch_accuracy = correct_predictions / total_predictions

    return epoch_loss, epoch_accuracy


def validation(model, loader, loss_function, device):

    model.eval()

    running_loss = 0
    correct_predictions = 0
    total_predictions = 0
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for tensors, metadata, labels in loader:
            tensors, metadata, labels = tensors.to(device), metadata.to(device), labels.to(device)

            outputs = model(tensors, metadata)
            loss = loss_function(outputs, labels.long())

            running_loss += loss.item() * tensors.size(0)
            predicted = torch.argmax(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

    epoch_loss = running_loss / total_predictions
    epoch_accuracy = correct_predictions / total_predictions

    return epoch_loss, epoch_accuracy, true_labels, predicted_labels
