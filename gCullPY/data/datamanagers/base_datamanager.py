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
Datamanager.
"""

from __future__ import annotations

from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Type
)

import torch
from torch import nn
from torch.nn import Parameter
from torch.utils.data.distributed import DistributedSampler
from typing_extensions import TypeVar
from gCullPY.cameras.cameras import Cameras
from gCullPY.data.datasets.base_dataset import InputDataset
from gCullPY.data.utils.nerfstudio_collate import nerfstudio_collate
from gCullUTILS.rich_utils import CONSOLE

TDataset = TypeVar("TDataset", bound=InputDataset, default=InputDataset)

def variable_res_collate(batch: List[Dict]) -> Dict:
    """Default collate function for the cached dataloader.
    Args:
        batch: Batch of samples from the dataset.
    Returns:
        Collated batch.
    """
    images = []
    imgdata_lists = defaultdict(list)
    for data in batch:
        image = data.pop("image")
        images.append(image)
        topop = []
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                # if the value has same height and width as the image, assume that it should be collated accordingly.
                if len(val.shape) >= 2 and val.shape[:2] == image.shape[:2]:
                    imgdata_lists[key].append(val)
                    topop.append(key)
        # now that iteration is complete, the image data items can be removed from the batch
        for key in topop:
            del data[key]

    new_batch = nerfstudio_collate(batch)
    new_batch["image"] = images
    new_batch.update(imgdata_lists)

    return new_batch


@dataclass
class DataManagerConfig():
    """Configuration for data manager instantiation; DataManager is in charge of keeping the train/eval dataparsers;
    After instantiation, data manager holds both train/eval datasets and is in charge of returning unpacked
    train/eval data at each iteration
    """

    _target: Type = field(default_factory=lambda: DataManager)
    """Target class to instantiate."""
    data: Optional[Path] = None
    """Source of data, may not be used by all models."""
    images_on_gpu: bool = False
    """Process images on GPU for speed at the expense of memory, if True."""


class DataManager(nn.Module):
    """Generic data manager's abstract class

    This version of the data manager is designed be a monolithic way to load data and latents,
    especially since this may contain learnable parameters which need to be shared across the train
    and test data managers. The idea is that we have setup methods for train and eval separately and
    this can be a combined train/eval if you want.

    Usage:
    To get data, use the next_train and next_eval functions.
    This data manager's next_train and next_eval methods will return 2 things:

    1. 'rays': This will contain the rays or camera we are sampling, with latents and
        conditionals attached (everything needed at inference)
    2. A "batch" of auxiliary information: This will contain the mask, the ground truth
        pixels, etc needed to actually train, score, etc the model

    Rationale:
    Because of this abstraction we've added, we can support more NeRF paradigms beyond the
    vanilla nerf paradigm of single-scene, fixed-images, no-learnt-latents.
    We can now support variable scenes, variable number of images, and arbitrary latents.


    Train Methods:
        setup_train: sets up for being used as train
        iter_train: will be called on __iter__() for the train iterator
        next_train: will be called on __next__() for the training iterator
        get_train_iterable: utility that gets a clean pythonic iterator for your training data

    Eval Methods:
        setup_eval: sets up for being used as eval
        iter_eval: will be called on __iter__() for the eval iterator
        next_eval: will be called on __next__() for the eval iterator
        get_eval_iterable: utility that gets a clean pythonic iterator for your eval data


    Attributes:
        train_count (int): the step number of our train iteration, needs to be incremented manually
        eval_count (int): the step number of our eval iteration, needs to be incremented manually
        train_dataset (Dataset): the dataset for the train dataset
        eval_dataset (Dataset): the dataset for the eval dataset
        includes_time (bool): whether the dataset includes time information

        Additional attributes specific to each subclass are defined in the setup_train and setup_eval
        functions.

    """

    train_dataset: Optional[InputDataset] = None
    eval_dataset: Optional[InputDataset] = None
    train_sampler: Optional[DistributedSampler] = None
    eval_sampler: Optional[DistributedSampler] = None
    includes_time: bool = False

    def __init__(self):
        """Constructor for the DataManager class.

        Subclassed DataManagers will likely need to override this constructor.

        If you aren't manually calling the setup_train and setup_eval functions from an overriden
        constructor, that you call super().__init__() BEFORE you initialize any
        nn.Modules or nn.Parameters, but AFTER you've already set all the attributes you need
        for the setup functions."""
        super().__init__()
        self.train_count = 0
        self.eval_count = 0

    def forward(self):
        """Blank forward method

        This is an nn.Module, and so requires a forward() method normally, although in our case
        we do not need a forward() method"""
        raise NotImplementedError

    def iter_train(self):
        """The __iter__ function for the train iterator.

        This only exists to assist the get_train_iterable function, since we need to pass
        in an __iter__ function for our trivial iterable that we are making."""
        self.train_count = 0

    def iter_eval(self):
        """The __iter__ function for the eval iterator.

        This only exists to assist the get_eval_iterable function, since we need to pass
        in an __iter__ function for our trivial iterable that we are making."""
        self.eval_count = 0

    @abstractmethod
    def setup_train(self):
        """Sets up the data manager for training.

        Here you will define any subclass specific object attributes from the attribute"""

    @abstractmethod
    def setup_eval(self):
        """Sets up the data manager for evaluation"""

    @abstractmethod
    def next_train(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next batch of data from the train data manager.

        Args:
            step: the step number of the eval image to retrieve
        Returns:
            A tuple of the camera for the image, and a dictionary of additional batch information
            such as the groundtruth image.
        """
        raise NotImplementedError

    @abstractmethod
    def next_eval(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next batch of data from the eval data manager.

        Args:
            step: the step number of the eval image to retrieve
        Returns:
            A tuple of the camera for the image, and a dictionary of additional batch information
            such as the groundtruth image.
        """
        raise NotImplementedError

    @abstractmethod
    def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
        """Retrieve the next eval image.

        Args:
            step: the step number of the eval image to retrieve
        Returns:
            A tuple of the step number, the camera for the image, and a dictionary of
            additional batch information such as the groundtruth image.
        """
        raise NotImplementedError

    @abstractmethod
    def get_train_rays_per_batch(self) -> int:
        """Returns the number of rays per batch for training."""
        raise NotImplementedError

    @abstractmethod
    def get_eval_rays_per_batch(self) -> int:
        """Returns the number of rays per batch for evaluation."""
        raise NotImplementedError

    @abstractmethod
    def get_datapath(self) -> Path:
        """Returns the path to the data. This is used to determine where to save camera paths."""

    @abstractmethod
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the data manager.

        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        return {}
