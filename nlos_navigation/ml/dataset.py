from typing import List, Self
from pathlib import Path
import pickle
import random
from abc import ABC, abstractmethod

import torch
from torch.utils.data import Dataset as TorchDataset
import torch.nn.functional as F

from nlos_navigation.ml.device import device


class Dataset(TorchDataset, ABC):
    pkl_paths: List[Path]

    @abstractmethod
    def copy(self, **kwargs) -> Self:
        pass


class HistogramDataset(Dataset):
    def __init__(
        self,
        pkl_paths: List[Path | str],
        *,
        start_gate: int,
        end_gate: int,
        add_noise: bool = False,
        return_histograms: bool = True,
        return_camera_pose: bool = True,
        return_object_pose: bool = True,
    ):
        self.pkl_paths = [Path(pkl_paths) for pkl_paths in pkl_paths]
        self.start_gate, self.end_gate = start_gate, end_gate
        self.add_noise = add_noise

        self.return_histograms = return_histograms
        self.return_camera_pose = return_camera_pose
        self.return_object_pose = return_object_pose

        self._loaded = False

    def copy(self, **kwargs) -> Self:
        kwargs.setdefault("pkl_paths", self.pkl_paths)
        kwargs.setdefault("start_gate", self.start_gate)
        kwargs.setdefault("end_gate", self.end_gate)
        kwargs.setdefault("add_noise", self.add_noise)
        kwargs.setdefault("return_histograms", self.return_histograms)
        kwargs.setdefault("return_camera_pose", self.return_camera_pose)
        kwargs.setdefault("return_object_pose", self.return_object_pose)
        return HistogramDataset(**kwargs)

    def load_data(self) -> List[dict]:
        def load_pkl(path: Path) -> List[dict]:
            path = path.resolve()
            assert path.exists(), f"Path {path} does not exist."
            with open(path, "rb") as file:
                data = []
                try:
                    while True:
                        entry = pickle.load(file)
                        if any(
                            [
                                entry.get(key, None) is None
                                for key in ["histograms", "object_pose"]
                            ]
                        ):
                            continue
                        data.append(entry)
                except EOFError:
                    pass

            return data

        data = []
        for pkl_path in self.pkl_paths:
            data.extend(load_pkl(pkl_path))
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> List[torch.Tensor]:
        sample = self.data[idx]
        if sample.get("camera_pose", None) is None:
            sample["camera_pose"] = torch.zeros(3)
        if sample.get("object_pose", None) is None:
            sample["object_pose"] = torch.zeros(3)
            return None

        # if "histogram" in sample:
        #     sample["histograms"] = sample.pop("histogram").reshape(9, -1)
        # if len(sample["camera_pose"]) == 4:
        #     sample["camera_pose"] = sample["camera_pose"][[0,1,3]]
        # if len(sample["object_pose"]) == 4:
        #     sample["object_pose"] = sample["object_pose"][[0,1,3]]
        histograms = torch.tensor(sample["histograms"], dtype=torch.float32)
        camera_pose = torch.tensor(sample["camera_pose"], dtype=torch.float32)
        object_pose = torch.tensor(sample["object_pose"], dtype=torch.float32)

        # Pass to device
        histograms = histograms.to(device)
        camera_pose = camera_pose.to(device)
        object_pose = object_pose.to(device)

        # Process histograms
        histograms = histograms[:, self.start_gate : self.end_gate]
        if self.add_noise:
            histograms = self.add_poisson_noise(histograms)

        # Normalize histograms
        # histograms = F.normalize(histograms, p=1, dim=0)

        out = []
        if self.return_histograms:
            out.append(histograms)
        if self.return_camera_pose:
            out.append(camera_pose)
        if self.return_object_pose:
            out.append(object_pose)
        return tuple(out)

    def add_poisson_noise(self, histograms: torch.Tensor) -> torch.Tensor:
        noise = torch.poisson(histograms)
        # gaussian_noise = torch.randn_like(histograms) * 0.1
        # return histograms
        return histograms + noise

    @property
    def data(self) -> List[dict]:
        if not self._loaded:
            self._data = self.load_data()
            self._loaded = True
        return self._data
