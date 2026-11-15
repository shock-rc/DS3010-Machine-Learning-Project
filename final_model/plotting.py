import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_confusion_matrix(true_labels, predicted_labels):
    confusion_mtx = confusion_matrix(true_labels, predicted_labels)
    return ConfusionMatrixDisplay(confusion_matrix=confusion_mtx, display_labels=[0, 1]).plot()


def plot_training_history(history):
    epochs = [entry["epoch"] for entry in history]
    train_loss = [entry["train_loss"] for entry in history]
    val_loss = [entry["val_loss"] for entry in history]
    train_accuracy = [entry["train_accuracy"] for entry in history]
    val_accuracy = [entry["val_accuracy"] for entry in history]

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_loss, label="Train Loss")
    axes[0].plot(epochs, val_loss, label="Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, train_accuracy, label="Train Accuracy")
    axes[1].plot(epochs, val_accuracy, label="Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    figure.tight_layout()
    return figure, axes
