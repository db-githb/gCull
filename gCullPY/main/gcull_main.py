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
from gCullPY.main.utils_main import (
    write_ply,
    load_config, 
    render_loop,
    run_mask_processing)

from gCullPY.main.utils_cull import (
    statcull, 
    modify_model,
)

from gCullPY.main.utils_ground import (
    find_ground_plane,
    get_ground_gaussians,
    ground_driver
)

from gCullUTILS.rich_utils import CONSOLE, TABLE
from rich.panel import Panel
from threading import Thread

import time
from contextlib import contextmanager

def _pct(n, d):  # safe percent string
    return f"{(n/d):.1%}" if d else "n/a"

@contextmanager
def step(console, title, emoji=":arrow_forward:"):
    t0 = time.perf_counter()
    console.log(f"{emoji} [bold]{title}[/bold]")
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        console.log(f":white_check_mark: Done [dim]({dt:.2f}s)[/dim]")

def log_totals(console, before, after, noun="splats"):
    culled = before - after
    console.log(
        f"• Removed {culled}/{before} {noun} "
        f"([bold]{_pct(culled, before)}[/bold]) → total: [bold]{after}[/bold]"
    )


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
        # Phase 1 — statistical cull
        starting_total = pipeline.model.means.shape[0]
        with step(CONSOLE, "Phase 1 — Statistical cull", emoji=":broom:"):
            cull_mask = statcull(pipeline)
            keep = ~cull_mask
            pipeline.model = modify_model(pipeline.model, keep)
            statcull_total = pipeline.model.means.shape[0]
            log_totals(CONSOLE, starting_total, statcull_total)

        # Phase 2 — render frames
        with step(CONSOLE, "Phase 2 — Rendering frames for mask extraction", emoji=":film_frames:"):
            render_dir = render_loop(self.model_path, config, pipeline)
            CONSOLE.log(":tada: Render complete")

        # Phase 3a — car mask (keep car)
        with step(CONSOLE, "Phase 3a — Apply car mask (keep car)", emoji=":car:"):
            pipeline.model = run_mask_processing("car", 0.9, 0.25, False, render_dir, config, pipeline)

        # Phase 3b — ground plane via RANSAC
        with step(CONSOLE, "Phase 3b — Find ground plane (RANSAC)", emoji=":triangular_ruler:"):
            keep, is_ground, norm, offset = find_ground_plane(pipeline.model)
            ground_gaussians = get_ground_gaussians(pipeline.model, is_ground)
            before = pipeline.model.means.shape[0]
            pipeline.model = modify_model(pipeline.model, keep)
            after = pipeline.model.means.shape[0]
            nvals = norm.detach().cpu().tolist()
            CONSOLE.log(f"• Plane normal: {tuple(round(float(x), 3) for x in nvals)}  offset: {float(offset):.3f}")
            log_totals(CONSOLE, before, after)

        # Phase 3c — ground mask (remove ground pixels)
        with step(CONSOLE, "Phase 3c — Apply ground mask (remove ground)", emoji=":herb:"):
            # NOTE: set invert=True if your run_mask_processing keeps matches by default.
            pipeline.model = run_mask_processing("ground", 0.5, 0.25, True, render_dir, config, pipeline)

        # Phase 4 — synthesize/restore ground
        with step(CONSOLE, "Phase 4 — Synthesize ground plane geometry", emoji=":seedling:"):
            before = pipeline.model.means.shape[0]
            pipeline.model = ground_driver(norm, ground_gaussians, pipeline)
            after = pipeline.model.means.shape[0]
            added = after - before
            CONSOLE.log(f"• Added {added} ground splats")

        # Phase 5 — write PLY
        with step(CONSOLE, "Phase 5 — Write PLY", emoji=":floppy_disk:"):
            filename = write_ply(self.model_path, pipeline.model)
            path = Path(filename)
            dir = config.datamanager.data.parents[1] / path.parent
            linked = f"[link=file://{dir}/]{path.name}[/link]"
            TABLE.add_row("Final 3DGS model", linked)
            CONSOLE.log(Panel(TABLE, title="[bold green]🎉 Cull Complete![/bold green] 🎉", expand=False))
