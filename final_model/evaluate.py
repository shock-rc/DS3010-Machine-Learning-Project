import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from .data import metadata_preprocessing, triplicate_split, TensorDataset
from .model import load_best_model
from .training import validation


def evaluate_best_model(
    metadata_path: str = "raw/oasis_cross-sectional.csv",
    checkpoint_path: str = "best_model.pth",
    batch_size: int = 4,
):
    data = metadata_preprocessing(metadata_path)
    _, _, test_data = triplicate_split(
        data,
        test_size = 0.2,
        cross_val_size = 0.2,
        random_state = 42,
        verbose = False
    )

    test_dataset = TensorDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers = 0)

    model, device = load_best_model(checkpoint_path)
    loss_function = torch.nn.CrossEntropyLoss()
    test_loss, test_accuracy, true_labels, predicted_labels = validation(
        model,
        test_loader,
        loss_function,
        device
    )

    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=["Non-Demented", "Demented"]
    )

    return {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "true_labels": true_labels,
        "predicted_labels": predicted_labels,
        "classification_report": report,
    }
