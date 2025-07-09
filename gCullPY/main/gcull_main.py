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
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

import torch
import tyro

from typing_extensions import Annotated
from gCullPY.pipelines.base_pipeline import VanillaPipelineConfig
from gCullPY.data.datamanagers.full_images_datamanager import FullImageDatamanager
from gCullPY.data.datasets.base_dataset import Dataset
from gCullPY.data.utils.dataloaders import FixedIndicesEvalDataloader
from gCullPY.utils.rich_utils import CONSOLE, ItersPerSecColumn
from gCullPY.main.utils_main import setup_write_ply, write_ply, eval_setup
from PIL import Image

import matplotlib.pyplot as plt
from gCullPY.main.utils_cull import get_cull_list

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

    load_config: Path
    """Path to config YAML file."""
    output_path: Path = Path("renders/output.mp4")
    """Path to output video file."""
    image_format: Literal["jpeg", "png"] = "jpeg"
    """Image format"""
    jpeg_quality: int = 100
    """JPEG quality"""
    downscale_factor: float = 1.0
    """Scaling factor to apply to the camera image resolution."""
    eval_num_rays_per_chunk: Optional[int] = None
    """Specifies number of rays per chunk during eval. If None, use the value in the config file."""
    rendered_output_names: List[str] = field(default_factory=lambda: ["rgb"])
    """Name of the renderer outputs to use. rgb, depth, etc. concatenates them along y axis"""
    depth_near_plane: Optional[float] = None
    """Closest depth to consider when using the colormap for depth. If None, use min value."""
    depth_far_plane: Optional[float] = None
    """Furthest depth to consider when using the colormap for depth. If None, use max value."""
    render_nearest_camera: bool = False
    """Whether to render the nearest training camera to the rendered camera."""
    check_occlusions: bool = False
    """If true, checks line-of-sight occlusions when computing camera distance and rejects cameras not visible to each other"""


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
    """Render all images in the dataset."""

    output_path: Path = Path("renders")
    """Path to output video file."""
    data: Optional[Path] = None
    """Override path to the dataset."""
    split: Literal["train", "val", "test", "train+test"] = "train" #hardcode both splits
    """Split to render."""
    
    def main(self):
        def show_mask(bool_mask):
            mask_np = bool_mask.cpu().numpy()   # convert to NumPy (H×W) array of True/False

            plt.figure(figsize=(6,6))
            plt.imshow(mask_np, cmap='gray', interpolation='nearest')
            plt.title("Boolean Mask")
            plt.axis('off') 
            plt.show()

        def get_mask(camera_idx, mask_root):
            filepath = mask_root+"/masks/mask_"+str(camera_idx+1).zfill(4)+".png"
            bool_mask = torch.tensor(np.array(Image.open(filepath))) == 0 # convert to bool tensor for ease of CUDA hand-off where black = True / non-black = False
            #show_mask(bool_mask)
            return bool_mask
        
        config, pipeline = eval_setup(
            self.load_config,
            test_mode="inference",
        )
        assert isinstance(config, (VanillaPipelineConfig))

        if self.downscale_factor is not None:
            dataparser = config.datamanager.dataparser
            if hasattr(dataparser, "downscale_factor"):
                setattr(dataparser, "downscale_factor", self.downscale_factor)

        root_dir = ""
        model = pipeline.model
        total_gauss = model.means.shape[0]
        cull_lst_master = torch.zeros(total_gauss, dtype=torch.bool)

        for split in self.split.split("+"):
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
            root_dir =  os.path.dirname(images_root)
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

        filename = root_dir+"splat_mod.ply"
        count, map_to_tensors = setup_write_ply(pipeline.model)
        write_ply(filename, count, map_to_tensors)

        table = Table(
            title=None,
            show_header=False,
            box=box.MINIMAL,
            title_style=style.Style(bold=True),
        )
        for split in self.split.split("+"):
            table.add_row(f"Outputs {split}", str(self.output_path / split))
        CONSOLE.print(Panel(table, title="[bold][green]:tada: Cull on split {} Complete :tada:[/bold]", expand=False))


Commands = tyro.conf.FlagConversionOff[
        Annotated[DatasetCull, tyro.conf.subcommand(name="dataset")]
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()


if __name__ == "__main__":
    entrypoint()


def get_parser_fn():
    """Get the parser function for the sphinx docs."""
    return tyro.extras.get_parser(Commands)  # noqa
