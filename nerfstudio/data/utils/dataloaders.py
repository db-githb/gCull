# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Code for sampling images from a dataset of images.
"""

# for multithreading
from abc import abstractmethod
from typing import Dict, List, Optional, Tuple, Union, TypeVar

import torch
from torch.utils.data.dataloader import DataLoader

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datasets.base_dataset import InputDataset

T = TypeVar("T")

def get_dict_to_torch(stuff: T, device: Union[torch.device, str] = "cpu", exclude: Optional[List[str]] = None) -> T:
    """Set everything in the dict to the specified torch device.

    Args:
        stuff: things to convert to torch
        device: machine to put the "stuff" on
        exclude: list of keys to skip over transferring to device
    """
    if isinstance(stuff, dict):
        for k, v in stuff.items():
            if exclude and k in exclude:
                stuff[k] = v
            else:
                stuff[k] = get_dict_to_torch(v, device)
        return stuff
    if isinstance(stuff, torch.Tensor):
        return stuff.to(device)
    return stuff


class EvalDataloader(DataLoader):
    """Evaluation dataloader base class

    Args:
        input_dataset: InputDataset to load data from
        device: Device to load data to
    """

    def __init__(
        self,
        input_dataset: InputDataset,
        device: Union[torch.device, str] = "cpu",
        **kwargs,
    ):
        self.input_dataset = input_dataset
        self.cameras = input_dataset.cameras.to(device)
        self.device = device
        self.kwargs = kwargs
        super().__init__(dataset=input_dataset)

    @abstractmethod
    def __iter__(self):
        """Iterates over the dataset"""
        return self

    @abstractmethod
    def __next__(self) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data"""

    def get_camera(self, image_idx: int = 0) -> Tuple[Cameras, Dict]:
        """Get camera for the given image index

        Args:
            image_idx: Camera image index
        """
        camera = self.cameras[image_idx : image_idx + 1]
        batch = self.input_dataset[image_idx]
        batch = get_dict_to_torch(batch, device=self.device, exclude=["image"])
        assert isinstance(batch, dict)
        return camera, batch

class FixedIndicesEvalDataloader(EvalDataloader):
    """Dataloader that returns a fixed set of indices.

    Args:
        input_dataset: InputDataset to load data from
        image_indices: List of image indices to load data from. If None, then use all images.
        device: Device to load data to
    """

    def __init__(
        self,
        input_dataset: InputDataset,
        image_indices: Optional[Tuple[int]] = None,
        device: Union[torch.device, str] = "cpu",
        **kwargs,
    ):
        super().__init__(input_dataset, device, **kwargs)
        if image_indices is None:
            self.image_indices = list(range(len(input_dataset)))
        else:
            self.image_indices = image_indices
        self.count = 0

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count < len(self.image_indices):
            image_idx = self.image_indices[self.count]
            camera, batch = self.get_camera(image_idx)
            self.count += 1
            return camera, batch
        raise StopIteration