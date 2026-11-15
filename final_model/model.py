import torch
import torch.nn as nn


class ConvolutionModel(nn.Module):
    """ 
    Main Deep Learning model for class prediction of CDR from MRI images
    
    This model uses a 3D CNN architecture with the following pipeline and hyperparameters:

        - Takes the MRI tensor and repeatedly passes it through convolution layers
            - Convolution Layer with out_channel = 2 * in_channel
            - Batch Normalization and ReLU activation
            - Max Pooling to reduce parameters and ease computation (2*2*2 kernel)
        - After 4 convolution layers, the features are pooled using Adaptive Average Pooling to reduce overfitting
        - The pooled features are passed through a linear layer to reduce the feature dimension to 64
        - The metadata is passed through a single linear layer to reduce the feature dimension to 32
        - The features from both MRI and metadata are concatenated and passed through a final linear layer
    """
    def __init__(self, metadata_dim: int, num_classes:int = 3):
        super().__init__()

        self.features = nn.Sequential(

            # Takes in a single channel image and processes it into 16 activation channels, normalizing and adding non-linearity usig ReLu
            # Max pooling is used to reduce parameters in each layer and to ease computation
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2),

            # Same convolution layer as above but takes out of up layer and returns twice the layers
            # Number of channels is doubled to compensate for loss via max pooling and to allow for more complex feature extraction
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
        )

        # Taking average due to small dataset size and to avoid overfitting
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Takes features learnt from 3D brain scans and compresses them
        self.mri_features = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.25)
        )

        # Takes features from the metadata and passes them through a single linear layer (basically log regression)
        self.metadata_features = nn.Sequential(
            nn.Linear(metadata_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.25)
        )

        # Final single layer classifier (again logistic regression) -> in dimension is (mri dimension + metadata dimension)
        self.classifier = nn.Sequential(
            nn.Linear(64 + 32, 32),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(32, num_classes)
        )

    def forward(self, tensors, metadata):

        # Passes the MRI tensor through the 3D CNN feature extractor
        x = self.features(tensors)
        x = self.pool(x)
        x = torch.flatten(x, 1)  # Flattening for linear layer input
        x = self.mri_features(x)

        # Passes the metadata through its own feature extractor
        metadata_features = self.metadata_features(metadata)

        # Concatenates the features from both MRI and metadata
        combined_features = torch.cat((x, metadata_features), dim=1)

        # Passes the combined features through the classifier to get final predictions
        output = self.classifier(combined_features)
        return output


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    return device


def load_best_model(checkpoint_path: str = "best_model.pth", device=None):
    if device is None:
        device = get_device()

    model = ConvolutionModel(metadata_dim = 8, num_classes = 2)
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model, device
