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

#!/usr/bin/env python
"""
render.py
"""
from __future__ import annotations
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from gCullPY.pipelines.base_pipeline import VanillaPipelineConfig
from gCullPY.data.datamanagers.full_images_datamanager import FullImageDatamanager
from gCullPY.data.datasets.base_dataset import Dataset
from gCullPY.data.utils.dataloaders import FixedIndicesEvalDataloader
from gCullPY.utils.rich_utils import CONSOLE, ItersPerSecColumn
from gCullPY.main.utils_main import setup_write_ply, write_ply, load_config, load_ply

from gCullPY.main.utils_cull import get_mask, get_cull_list

from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TimeRemainingColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.panel import Panel
from rich.table import Table
import numpy as np
from rich import box, style

@dataclass
class BaseCull:
    """Base class for rendering."""
    load_model: str
    """Path to model file."""
    output_dir: Path = Path("culled_models/output.ply")
    """Path to output model file."""


@contextmanager
def _disable_datamanager_setup(cls):
    """
    Disables setup_train or setup_eval for faster initialization.
    """
    old_setup_train = getattr(cls, "setup_train")
    old_setup_eval = getattr(cls, "setup_eval")
    setattr(cls, "setup_train", lambda *args, **kwargs: None)
    setattr(cls, "setup_eval", lambda *args, **kwargs: None)
    yield cls
    setattr(cls, "setup_train", old_setup_train)
    setattr(cls, "setup_eval", old_setup_eval)


@dataclass
class DatasetCull(BaseCull):
    """Cull using all images in the dataset."""

    mask_dir: Optional[Path] = None
    #"""Override path to the dataset."""
    
    # main/driver function
    def run_cull(self):
        downscale_factor = 1
        
        model_path = Path(self.load_model)

        ext = model_path.suffix.lower()

        if ext == '.ply':
            config, pipeline = load_ply(self.load_model)
        else:
            config, pipeline = load_config(
                self.load_model,
                test_mode="inference",
            )
        assert isinstance(config, (VanillaPipelineConfig))

        if downscale_factor is not None:
            dataparser = config.datamanager.dataparser
            if hasattr(dataparser, "downscale_factor"):
                setattr(dataparser, "downscale_factor", downscale_factor)

        root_dir = ""
        model = pipeline.model
        total_gauss = model.means.shape[0]
        cull_lst_master = torch.zeros(total_gauss, dtype=torch.bool)

        for split in "train+test".split("+"):
            datamanager: FullImageDatamanager
            dataset: Dataset
            if split == "train":
                with _disable_datamanager_setup(config.datamanager._target):  # pylint: disable=protected-access
                    datamanager = config.datamanager.setup(test_mode="test", device=pipeline.device)

                dataset = datamanager.train_dataset
                dataparser_outputs = getattr(dataset, "_dataparser_outputs", datamanager.train_dataparser_outputs)
            else:
                with _disable_datamanager_setup(config.datamanager._target):  # pylint: disable=protected-access
                    datamanager = config.datamanager.setup(test_mode=split, device=pipeline.device)

                dataset = datamanager.eval_dataset
                dataparser_outputs = getattr(dataset, "_dataparser_outputs", None)
                if dataparser_outputs is None:
                    dataparser_outputs = datamanager.dataparser.get_dataparser_outputs(split=datamanager.test_split)
            dataloader = FixedIndicesEvalDataloader(
                input_dataset=dataset,
                device=datamanager.device,
                num_workers=datamanager.world_size * 4,
            )
            images_root = Path(os.path.commonpath(dataparser_outputs.image_filenames))
            root_dir =  images_root.parent
            mask_root = root_dir
            
            with Progress(
                TextColumn(f"\u2702\ufe0f\u00A0 Culling split {split} \u2702\ufe0f\u00A0"),
                BarColumn(),
                TaskProgressColumn(
                    text_format="[progress.percentage]{task.completed}/{task.total:>.0f}({task.percentage:>3.1f}%)",
                    show_speed=True,
                ),
                ItersPerSecColumn(suffix="fps"),
                TimeRemainingColumn(elapsed_when_finished=False, compact=False),
                TimeElapsedColumn(),
            ) as progress:
                for camera_idx, (camera, batch) in enumerate(progress.track(dataloader, total=len(dataset))):
                    with torch.no_grad():
                        camera.camera_to_worlds = camera.camera_to_worlds.squeeze() # splatoff rasterizer requires cam2world.shape = [3,4]
                        bool_mask = get_mask(camera_idx, mask_root).to(pipeline.model.device)
                        cull_lst = get_cull_list(model, camera, bool_mask)
                        cull_lst_master |= cull_lst.to("cpu")
                        #print(f"{camera_idx}: {cull_lst_master.sum().item()}")

        print(f"Total culled: {cull_lst_master.sum().item()}/{total_gauss}")
        keep = ~cull_lst_master
        with torch.no_grad():
            pipeline.model.means.data = model.means[keep].clone()
            pipeline.model.opacities.data = model.opacities[keep].clone()
            pipeline.model.scales.data = model.scales[keep].clone()
            pipeline.model.quats.data = model.quats[keep].clone()
            pipeline.model.features_dc.data = model.features_dc[keep].clone()
            pipeline.model.features_rest.data = model.features_rest[keep].clone()
        
        config_path = Path(self.load_model)
        model_name = config_path.parent.name
        experiment_name = config_path.parts[1]  # e.g., 'my-experiment'
        filename = config_path.parent / f"{experiment_name}_{model_name}_culled.ply"
        count, map_to_tensors = setup_write_ply(pipeline.model)
        write_ply(filename, count, map_to_tensors)
    
        path = Path(filename)
        dir = root_dir.parents[1] / path.parent
        linked_name = f"[link=file://{dir}/]{path.name}[/link]"
        table = Table(
            title=None,
            show_header=False,
            box=box.MINIMAL,
            title_style=style.Style(bold=True),
        )
        table.add_row(f"Final 3DGS model", linked_name)
        CONSOLE.print(Panel(table, title="[bold green]🎉 Cull Complete![/bold green] 🎉", expand=False))
