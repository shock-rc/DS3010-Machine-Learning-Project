# Basic libraries for PyTorch deep learning 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import nibabel as nib #MRI Image processing library, reads .nii files

# sklearn train test split and evaluation metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    roc_auc_score
)

# These functions are used for preprocessing the data before model inferencing
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

# Generic libraries, pls dont question their existence
import pandas as pd
import numpy as np
import os


def metadata_preprocessing(file_path:str = "raw/oasis_cross-sectional.csv") -> pd.DataFrame:
    """
    This function takes in a file path to a CSV file containing metadata and performs preprocessing steps on it.

    The file is located in the 'raw' directory of the project. The preprocessing steps include:
        - One hot encoding categorical variables
        - Removal of columns with significant missing values
        - KNN Imputation for missing values
        - Combining rare classes in the target variable to avoid class imbalance issues
        - Adding .nii file paths to the metadata for easier access to MRI images
        - Splitting of ID column into subject_id and session columns for better organization

    Args:
        file_path (str): The path to the CSV file containing metadata.
    Returns:
        metadata (pd.DataFrame): A preprocessed DataFrame containing the metadata.
    """

    metadata = pd.read_csv(file_path)
    imputer = KNNImputer(n_neighbors=5)
    scaler = StandardScaler()

    # Delay column had 95.4% missing values and was irrelevant to study
    # Hand was dropped as everyone was right-handed, not useful for analysis
    metadata = metadata.drop(columns = ["Delay", "Hand"])

    # Created binary columns for gender and dropped original column (One Hot Encoding)
    metadata['male'] = (metadata['M/F'] == 'M').astype(int)
    metadata['female'] = (metadata['M/F'] == 'F').astype(int)
    metadata = metadata.drop(columns = "M/F")

    # CDR of 2 i.e moderate dementia is very rare in the dataset, so we will combine it with CDR of 1 i.e mild dementia
    # Avoids serious class imbalance issues and allows for better model training
    metadata.loc[metadata['CDR'] == 2.0, 'CDR'] = 1

    # Performing class remappint with the following mapping -> {non-demented:0, mild demented:1, demented:1}
    remapping = {0.0: 0, 0.5: 1, 1.0: 1}
    metadata["CDR"] = metadata["CDR"].map(lambda x : remapping[x])

    # Extraction of subject id and session id and storing separately from the metadata
    subject_id = metadata["ID"].map(lambda x : "_".join(x.split("_")[:2]))
    session = metadata["ID"].map(lambda x : x.split("_")[-1])

    # Extracting data path from labels to easily access the MRI images for training and testing
    mapping:dict[str, str] = {}
    files:list[str] = os.listdir("raw/oasis/OASIS")

    for file in files:
        uid = "_".join(file.split("_")[:3])
        mapping[uid] = f"raw/oasis/OASIS/{file}"

    paths = metadata["ID"].map(lambda x : mapping[x])

    # Adding extracted subject id, session id and paths to the metadata and dropping ID as column is redundant now
    metadata["subject_id"] = subject_id
    metadata["session"] = session
    metadata["path"] = paths
    metadata = metadata.drop(columns = "ID")

    # Impute missing values using KNN using default 5nn, high number is not recommended due to small dataset size
    # StandardScaler is used for avoiding range bias on KNN, reverse transformation is done to restore original scale
    exclude = ["CDR"]
    num_cols = [col for col in metadata.select_dtypes(include = ["int", "float"]).columns if col not in exclude]
    metadata[num_cols] = scaler.inverse_transform(imputer.fit_transform(scaler.fit_transform(metadata[num_cols])))

    return metadata


def triplicate_split(data:pd.DataFrame, test_size:float = 0.2, cross_val_size:float = 0.2, random_state:int = 42, verbose = False) -> tuple:
    """
    This function takes in a DataFrame containing the metadata and a numpy array containing the target variable (CDR) for classification.
    It performs a triplicate split of the data into training, validation, and testing sets.

    Args:
        data (pd.DataFrame): A DataFrame containing the metadata.
        test_size (float): The proportion of the dataset to include in the test split. Default is 0.2.
        cross_val_size (float): The proportion of the training set to include in the validation split. Default is 0.2.
        random_state (int): The seed used by the random number generator. Default is 42.
        verbose (bool): If True, prints the shapes of the resulting datasets. Default is False.

    Returns:
        train_data (pd.DataFrame): A DataFrame containing the training data.
        val_data (pd.DataFrame): A DataFrame containing the validation data.
        test_data (pd.DataFrame): A DataFrame containing the testing data.
    """

    # Splitting into training and testing data
    train_data, test_data = train_test_split(data, test_size = test_size, random_state = random_state, stratify = data['CDR'])

    # Splitting training data into training and cross-validation data
    train_data, val_data = train_test_split(train_data, test_size = cross_val_size, random_state = random_state, stratify = train_data['CDR'])

    if verbose:
        print(f"Training data shape: {train_data.shape}")
        print(f"Validation data shape: {val_data.shape}")
        print(f"Testing data shape: {test_data.shape}")

    return train_data, val_data, test_data


class TensorDataset(Dataset):
    def __init__(self, metadata, target_tensor_size = (128,128,128)):
        """
        Args:
            metadata (pd.DataFrame): A DataFrame containing the metadata that is to be fed into the DL model.
            target_tensor_size (tuple): The target size of the tensor. Default is (128, 128, 128).
        """
        self.data = metadata
        self.target_size = target_tensor_size

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        """
        Based on the index, method retrieves corresponding MRI image and performs the following preprocessing steps:
            - Loads the corresponding .nii file using nibabel
            - Converts the binary data into a parseable numpy array and clears all NaN values as per nan_to_num method
            - Perfom z-score standardisation on non-zero values to avoid injecting noise from background MRI noise
            - Convert the numpy array into tensor and add channel dimension for CNN input (assumed grayscale as MRI is single channel)
            - splits the label and returns it separately
            - Takes all numerical metadata from the row and gives that as a feature vector
        Args:
            idx (int): The index of the data point to retrieve.
        Returns:
            tensor_data (torch.Tensor): A tensor containing the preprocessed MRI image data.
            feature_vector (torch.Tensor): A tensor containing the numerical metadata for the MRI image.
            label (torch.Tensor): A tensor containing the corresponding label (CDR) for the MRI image.
        """
        # Retrieves row information and corresponding .nii file from the dataset
        row_data = self.data.iloc[idx]
        nii_file = nib.load(row_data['path'])

        # Conversion of binary data into a numpy array of floats
        tensor_data = nii_file.get_fdata().astype(np.float32)
        tensor_data = np.nan_to_num(tensor_data)

        # MRI data has backrgound noise, near zero information, standardisation is performed using non-zero values to avoid injecting noise
        # z-score standardisation is performed
        non_zero_values = tensor_data[tensor_data != 0]
        tensor_data = (tensor_data - np.mean(non_zero_values)) / (np.std(non_zero_values) + 1e-8)  # Adding a small value to avoid division by zero

        # Conversion of numpy array into PyTorch tensor and adding channel dimension for CNN input (assumed grayscale as MRI is single channel)
        # float32 is used to avoid memory issues as model has been trained on 16GB unified RAM machine
        tensor_data = torch.from_numpy(tensor_data).unsqueeze(0).unsqueeze(0)  # Adding batch dim for interpolate
        tensor_data = F.interpolate(tensor_data, size = self.target_size, mode = 'trilinear', align_corners = False).squeeze(0)  # Removing batch dim

        label = torch.tensor(row_data['CDR'], dtype = torch.float32)

        # Extract numerical metadata and convert to tensor
        metadata_cols = ["Age", "Educ", "SES", "MMSE", "eTIV", "nWBV", "ASF", "male"]
        feature_vector = torch.tensor(row_data[metadata_cols].values.astype(np.float32), dtype=torch.float32)

        return tensor_data, feature_vector, label
