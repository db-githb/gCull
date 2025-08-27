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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import torch

from gCullPY.pipelines.base_pipeline import VanillaPipelineConfig

from gCullPY.main.utils_main import write_ply, load_config, render_loop
from gCullPY.main.utils_cull import (
    statcull, 
    modify_model,
    cull_loop, 
    visualize_mask_and_points,
    find_ground_plane,
    get_ground_gaussians,
    modify_ground_gaussians, 
    densify_ground_plane_jitter, 
    append_gaussians_to_model,
    get_all_ground_gaussians,
    fill_hole_with_known_plane,
    assign_attrs_for_new_gaussians,
    mask_by_plane_alignment
)
from gCullMASK.mask_main import MaskProcessor
from gCullUTILS.rich_utils import CONSOLE, TABLE
from rich.panel import Panel
from threading import Thread

@dataclass
class BaseCull:
    """Base class for rendering."""
    model_path: Path
    """Path to model file."""
    output_dir: Path = Path("culled_models/output.ply")
    """Path to output model file."""


@dataclass
class DatasetCull(BaseCull):
    """Cull using all images in the dataset."""

    mask_dir: Optional[Path] = None
    #"""Override path to the dataset."""
    
    # main/driver function
    def run_cull(self):

        # load model
        config, pipeline = load_config(
            self.model_path,
            test_mode="inference",
        )
        config.datamanager.dataparser.downscale_factor = 1
        # Phase 1 - run statCull
        starting_total = pipeline.model.means.shape[0]
        cull_mask = statcull(pipeline)
        keep = ~cull_mask
        pipeline.model = modify_model(pipeline.model, keep)
        statcull_total = pipeline.model.means.shape[0]
        CONSOLE.log(f"Phase 1 culled: {cull_mask.sum().item()}/{starting_total} ➜ New Total = {statcull_total}")
       

        # render images from modified model
        CONSOLE.print("[bold][yellow]Rendering frames for mask extraction...[/bold]")
        #render_dir = render_loop(self.model_path, config, pipeline)
        CONSOLE.log("[bold][green]:tada: Render Complete :tada:[/bold]")

        # get masks from rendered images
        #mp = MaskProcessor(render_dir, "ground")
        #mp = MaskProcessor(Path("renders/IMG_4718/"), "car")
        #mp.run_mask_processing()

        # Phase 2 - run gCull
        keep = cull_loop(config, pipeline)
        pipeline.model = modify_model(pipeline.model, keep)
        gCull_total = pipeline.model.means.shape[0]
        CONSOLE.log(f"Total culled: {(statcull_total-keep.sum().item())}/{statcull_total} ➜ New Total = {gCull_total}")

        # Phase 3 - find ground
        CONSOLE.log(f":running: Running RANSAC and finding ground plane...")
        keep, is_ground, norm, offset = find_ground_plane(pipeline.model)
        ground_gaussians = get_ground_gaussians(pipeline.model, is_ground)
        CONSOLE.log(f"Culling noisy ground gaussians")
        pipeline.model = modify_model(pipeline.model, keep)
        CONSOLE.log(f"Total culled: {(gCull_total-keep.sum().item())}/{gCull_total} ➜ New Total = {pipeline.model.means.shape[0]}")

        # Phase 4 - cull gaussians with angular deviation from ground plane
        #keep = mask_by_plane_alignment(norm, ground_gaussians, tau_deg=20.0)
        #ground_gaussians = modify_ground_gaussians(ground_gaussians, keep)

        # Phase 5 - expand ground
        CONSOLE.log(f"Expanding ground")
        new_gaussians = densify_ground_plane_jitter(ground_gaussians, norm, offset)
        pipeline.model = append_gaussians_to_model(pipeline.model, new_gaussians)
        CONSOLE.log(f"New Total = {pipeline.model.means.shape[0]}")

        complete_ground_gaussians = get_all_ground_gaussians(ground_gaussians, new_gaussians)
        new_pts = fill_hole_with_known_plane(complete_ground_gaussians["means"].cpu(), norm.cpu(), offset.cpu(), keep_ratio=.1) # points to fill hole
        timbit =  assign_attrs_for_new_gaussians(complete_ground_gaussians, new_pts)
        timbit_dense = densify_ground_plane_jitter(timbit, norm, offset, samples_per_point=80, expand_scale=2)
        complete_ground_tile = get_all_ground_gaussians(ground_gaussians, timbit_dense)
        length = complete_ground_tile["means"][:,1].max() - complete_ground_tile["means"][:,1].min()
        complete_ground_tile["means"][:,1] += length
        pipeline.model = append_gaussians_to_model(pipeline.model, complete_ground_tile)
        CONSOLE.log(f"New Total = {pipeline.model.means.shape[0]}")

        # write modified model to file
        CONSOLE.log(f"Writing to ply...")
        filename = write_ply(self.model_path, pipeline.model)
        path = Path(filename)
        dir = config.datamanager.data.parents[1] / path.parent
        linked_name = f"[link=file://{dir}/]{path.name}[/link]"
        TABLE.add_row(f"Final 3DGS model", linked_name)
        CONSOLE.log(Panel(TABLE, title="[bold green]🎉 Cull Complete![/bold green] 🎉", expand=False))
