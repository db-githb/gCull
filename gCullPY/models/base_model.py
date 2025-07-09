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
Base Model implementation which takes in Cameras
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Type, Union

import torch
from torch import nn
from torch.nn import Parameter

from gCullPY.cameras.cameras import Cameras
# Model related configs
@dataclass
class ModelConfig():
    """Configuration for model instantiation"""

    _target: Type = field(default_factory=lambda: Model)
    """target class to instantiate"""


class Model(nn.Module):
    """Model class
    Where everything (Fields, Optimizers, Samplers, Visualization, etc) is linked together. This should be
    subclassed for custom NeRF model.

    Args:
        config: configuration for instantiating model
    """

    config: ModelConfig

    def __init__(
        self,
        config: ModelConfig,
        num_train_data: int,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_train_data = num_train_data
        self.kwargs = kwargs
        self.collider = None

        self.populate_modules()  # populate the modules
        self.callbacks = None
        # to keep track of which device the nn.Module is on
        self.device_indicator_param = nn.Parameter(torch.empty(0))

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.device_indicator_param.device

    def populate_modules(self):
        """Set the necessary modules to get the network working."""
        # default instantiates optional modules that are common among many networks
        # NOTE: call `super().populate_modules()` in subclasses


    @abstractmethod
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """

    @abstractmethod
    def get_outputs(self, cam: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Camera and returns a dictionary of outputs.

        Args:
            ray_bundle: Input camera. This camera should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """

    def forward(self, cam: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Run forward starting with a camera. This outputs different things depending on the configuration
        of the model and whether or not the batch is provided (whether or not we are training basically)

        Args:
            ray_bundle: containing all the information needed to render that ray latents included
        """

        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)

        return self.get_outputs(ray_bundle)

    def load_model(self, loaded_state: Dict[str, Any]) -> None:
        """Load the checkpoint from the given path

        Args:
            loaded_state: dictionary of pre-trained model states
        """
        state = {key.replace("module.", ""): value for key, value in loaded_state["model"].items()}
        self.load_state_dict(state)  # type: ignore
