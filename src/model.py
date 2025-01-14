# src/model.py
import torch
import torch.nn as nn
from preprocess import N_MELS  # Import the constant

class SpeakerSeparationModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Store n_mels as instance variable
        self.n_mels = N_MELS
        
        # Feature extraction layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # LSTM for temporal pattern analysis
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        
        # Final layers for speaker separation
        self.fc_layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Final projection to match input spectrogram dimensions
        self.output_projection = nn.Linear(128, self.n_mels)  # n_mels should match preprocessing
    
    def forward(self, x):
        # Input shape: [batch_size, 1, time, freq]
        batch_size = x.size(0)
        
        # Conv layers
        x = self.conv_layers(x)
        
        # Reshape for LSTM
        x = x.permute(0, 2, 1, 3)  # [batch, time, channels, freq]
        x = x.reshape(batch_size, -1, x.size(2) * x.size(3))
        
        # LSTM
        x, _ = self.lstm(x)
        
        # FC layers
        x = self.fc_layers(x)
        
        # Project to final dimensions
        x = self.output_projection(x)
        
        return x
    
    def loss_function(self, predicted, target):
        """
        Modified loss function to handle dimension mismatch
        """
        # Ensure same dimensions
        if predicted.size() != target.size():
            # Interpolate predicted to match target size
            predicted = torch.nn.functional.interpolate(
                predicted.unsqueeze(1),
                size=target.size()[2:],
                mode='bilinear',
                align_corners=False
            ).squeeze(1)
        
        l1_loss = nn.L1Loss()(predicted, target)
        return l1_loss