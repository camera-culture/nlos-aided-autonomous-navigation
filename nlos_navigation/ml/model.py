from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Model(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor, camera_pose: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def debug(self, x: torch.Tensor):
        """Called on each eval so that the model can print useful info."""
        pass

    @classmethod
    def load(cls, path: str, *, eval: bool = True, **kwargs):
        model = cls(**kwargs)
        model.load_state_dict(torch.load(path, weights_only=True))
        if eval:
            model.eval()
        return model


class CNN(Model):
    def __init__(self, *, num_bins: int, output_dim: int):
        super().__init__()
        self.num_bins = num_bins

        # Histogram convolution layers
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=9, out_channels=4, kernel_size=3, padding=1),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3, padding=1),
            nn.BatchNorm1d(8),
            nn.Dropout(0.5),
            nn.ReLU(),
        )

        # Simplify fully connected layers
        fc_layers = [
            nn.Linear(8 * num_bins + 3, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim),
        ]
        if output_dim == 1:
            fc_layers.append(nn.Sigmoid())
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor, camera_pose: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(x)

        B, N, C, T = x.shape
        x = x.view(B * N, C, T)  # Reshape to (B*N, C, T) for Conv1d
        x = self.conv(x)
        x = x.view(B * N, -1)

        # Expand camera_pose to match batch size
        camera_pose = camera_pose.view(B, -1)
        camera_pose = (
            camera_pose.unsqueeze(1).expand(-1, N, -1).contiguous().view(B * N, -1)
        )

        # Concatenate processed histogram with camera_pose
        x = torch.cat((x, camera_pose), dim=1)
        x = self.fc(x)
        return x.view(B, N, -1).mean(dim=1)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        elif len(x.shape) == 2:
            x = x.unsqueeze(0).unsqueeze(1)
        return x

class ModelFactory:
    @staticmethod
    def create(model_type: str, **kwargs) -> Model:
        model = globals().get(model_type)
        if model is None:
            raise ValueError(f"Model {model_type} not found.")
        return model(**kwargs)