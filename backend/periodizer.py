import torch
import torch.nn as nn

class StarClassifier(nn.Module):
    def __init__(self, num_classes, input_length=512):
        super(StarClassifier, self).__init__()
        
        self.input_length = input_length
        
        # Shared Feature Extraction Backbone
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten()
        )
        
        # Calculate the number of features after convolution layers
        # After two MaxPool1d with kernel_size=2: input_length // 4
        # After final conv layer: 128 channels
        conv_output_size = (input_length // 4) * 128
        
        # Period Prediction Head (Regression)
        self.period_head = nn.Sequential(
            nn.Linear(in_features=conv_output_size, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=1)
        )
        
        # Star Type Classification Head (Classification)
        self.type_head = nn.Sequential(
            nn.Linear(in_features=conv_output_size, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=num_classes)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Pass the input through the feature extraction backbone
        x = self.features(x)
        
        # Separate the output into the two heads
        period_prediction = self.period_head(x)
        type_prediction = self.type_head(x)
        
        return period_prediction, type_prediction