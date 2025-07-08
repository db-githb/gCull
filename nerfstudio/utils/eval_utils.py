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
Evaluation utils
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal, Tuple

import torch
import re, yaml

from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.rich_utils import CONSOLE

from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig
from nerfstudio.models.splatfacto import SplatfactoModelConfig
from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParserConfig
from nerfstudio.data.datasets.base_dataset import InputDataset

def eval_load_checkpoint(config, pipeline: Pipeline) -> Tuple[Path, int]:
    ## TODO: ideally eventually want to get this to be the same as whatever is used to load train checkpoint too
    """Helper function to load checkpointed pipeline

    Args:
        config (DictConfig): Configuration of pipeline to load
        pipeline (Pipeline): Pipeline instance of which to load weights
    Returns:
        A tuple of the path to the loaded checkpoint and the step at which it was saved.
    """
    assert config.load_dir is not None
    if config.load_step is None:
        CONSOLE.print("Loading latest checkpoint from load_dir")
        # NOTE: this is specific to the checkpoint name format
        if not os.path.exists(config.load_dir):
            CONSOLE.rule("Error", style="red")
            CONSOLE.print(f"No checkpoint directory found at {config.load_dir}, ", justify="center")
            CONSOLE.print(
                "Please make sure the checkpoint exists, they should be generated periodically during training",
                justify="center",
            )
            sys.exit(1)
        load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(config.load_dir))[-1]
    else:
        load_step = config.load_step
    load_path = config.load_dir / f"step-{load_step:09d}.ckpt"
    assert load_path.exists(), f"Checkpoint {load_path} does not exist"
    loaded_state = torch.load(load_path, map_location="cpu")
    pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
    CONSOLE.print(f":white_check_mark: Done loading checkpoint from {load_path}")
    return load_path, load_step

def to_path(val):
        if isinstance(val, list):
            return Path(*val)
        return Path(val)

def eval_setup(
    config_path: Path,
    test_mode: Literal["test", "val", "inference"] = "test",
) -> Tuple[VanillaPipelineConfig, Pipeline]:

    # load save config
    txt = config_path.read_text()
    cleaned = re.sub(r'!+python/[^\s]+', '', txt)
    data = yaml.safe_load(cleaned)
    # Extract data root
    dm_data = data.get('data')
    if isinstance(dm_data, list):
        # e.g. ["data", "discord_car"] → Path("data/discord_car")
        dm_root = Path(*dm_data)
    else:
        # e.g. "data/discord_car" → Path("data/discord_car")
        dm_root = Path(dm_data)
    
    colmap_parser = ColmapDataParserConfig(
        colmap_path=(dm_root / "colmap" / "sparse" / "0").resolve(),
        images_path=(dm_root / "images").resolve(),
        load_3D_points=True,
        assume_colmap_world_coordinate_convention=True,
        auto_scale_poses=True,
        center_method="poses",
        downscale_factor=1
    )

    # Build datamanager config
    dm_conf = FullImageDatamanagerConfig(
        data=dm_root,
        dataset=InputDataset,
        dataparser=colmap_parser,
        cache_images="cpu",
        cache_images_type="uint8",
        camera_res_scale_factor=1.0,
    )
    
    config = VanillaPipelineConfig(
        datamanager=dm_conf,
        model=SplatfactoModelConfig(),
    )

    CONSOLE.print("Loading latest checkpoint from load_dir")
    output_dir = to_path(data["output_dir"])
    experiment_name = data["experiment_name"]
    method_name = data["method_name"]
    timestamp = data["timestamp"]
    relative_model_dir = to_path(data.get("relative_model_dir", "."))
    config.load_dir  = output_dir / experiment_name / method_name / timestamp / relative_model_dir
    load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(config.load_dir))[-1]
    load_path = config.load_dir / f"step-{load_step:09d}.ckpt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = config.setup(device=device, test_mode=test_mode)
    assert load_path.exists(), f"Checkpoint {load_path} does not exist"
    loaded_state = torch.load(load_path, map_location="cpu")
    pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
    CONSOLE.print(f":white_check_mark: Done loading checkpoint from {load_path}")
    
    return config, pipeline
