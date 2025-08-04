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

from gCullPY.pipelines.base_pipeline import VanillaPipelineConfig

from gCullPY.main.utils_main import write_ply, load_config, render_loop
from gCullPY.main.utils_cull import statcull, modify_model, cull_loop
from gCullMASK.mask_main import MaskProcessor
from gCullUTILS.rich_utils import CONSOLE, TABLE
from rich.panel import Panel

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

        # Phase 1 - run statCull
        #starting_total = pipeline.model.means.shape[0]
        #cull_mask = statcull(pipeline)
        #keep = ~cull_mask
        #pipeline.model = modify_model(pipeline.model, keep)
        #statcull_total = pipeline.model.means.shape[0]
        #CONSOLE.log(f"Phase 1 culled: {cull_mask.sum().item()}/{starting_total} ➜ New Total = {statcull_total}")
       

        # render images from modified model
        #CONSOLE.print("[bold][yellow]Rendering frames for mask extraction...[/bold]")
        #render_dir = render_loop(self.model_path, config, pipeline)
        #CONSOLE.log("[bold][green]:tada: Render Complete :tada:[/bold]")

        # get masks from rendered images
        #mp = MaskProcessor(render_dir, "car")
        #mp.run_mask_processing()

        # Phase 2 - run gCull
        rgb = cull_loop(config, pipeline)
        print("done")
        #keep = cull_lst == 0
        #pipeline.model = modify_model(pipeline.model, keep)
        #CONSOLE.log(f"Total culled: {cull_lst.sum().item()}/{statcull_total} ➜ New Total = {pipeline.model.means.shape[0]}, writing to ply...")

        # write modified model to file
        #filename = write_ply(self.model_path, pipeline.model)
        #path = Path(filename)
        #dir = config.datamanager.data.parents[1] / path.parent
        #linked_name = f"[link=file://{dir}/]{path.name}[/link]"
        #TABLE.add_row(f"Final 3DGS model", linked_name)
        #CONSOLE.log(Panel(TABLE, title="[bold green]🎉 Cull Complete![/bold green] 🎉", expand=False))
