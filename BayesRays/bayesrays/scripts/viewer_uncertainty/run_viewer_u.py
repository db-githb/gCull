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
# for older versions of nerfstudio filtering slider is currently not supported.
"""
Starts viewer in eval mode.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, fields
from pathlib import Path

import tyro
import torch
import torch.nn.functional as func
import numpy as np
import types

import pkg_resources

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import writer, colors
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.models.nerfacto import NerfactoModel
from nerfstudio.models.splatfacto import SplatfactoModel
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components import renderers
from nerfstudio.field_components.spatial_distortions import SceneContraction

import torch.nn.functional as F
from bayesrays.scripts.output_uncertainty import get_uncertainty

if pkg_resources.get_distribution("nerfstudio").version >= "0.3.1":
    from nerfstudio.viewer_legacy.server.viewer_state import ViewerLegacyState
    from nerfstudio.viewer_legacy.server.viewer_elements import  ViewerSlider, ViewerCheckbox
else:
    from nerfstudio.viewer_legacy.server import viewer_utils
    from nerfstudio.utils.writer import EventName, TimeWriter

import os
from plyfile import PlyData, PlyElement

from bayesrays.utils.utils import find_grid_indices, get_rasterizer_output, num_iPts, get_opacity, sort_package
import numpy as np
from tqdm import tqdm
from datetime import datetime
import json

def construct_list_of_attributes(model, exclude_filter=False):
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(model.features_dc.shape[1]):
        l.append('f_dc_{}'.format(i))
    for i in range(model.features_rest.shape[1]*model.features_rest.shape[2]):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(model.scales.shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(model.quats.shape[1]):
        l.append('rot_{}'.format(i))
    if not exclude_filter:
        l.append('filter_3D')
    return l

def save_ply(model, path):
    print("**SAVING PLY**")

    xyz = model.means[:4000,...].detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = model.features_dc[:4000,...].detach().contiguous().cpu().numpy()
    f_rest = model.features_rest[:4000,...].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = model.opacities[:4000,...].detach().cpu().numpy()
    scale = model.scales[:4000,...].detach().cpu().numpy()
    rotation = model.quats[:4000,...].detach().cpu().numpy()
        
    filter_3D = model.filter_3D[:4000,...].detach().cpu().numpy()
    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(model)]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, filter_3D), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el], text=True).write(path)
    print("**END SAVE: ", path)

def get_output_nerfacto_new(self, ray_bundle):
    ''' reimplementation of get_output function from models because of lack of proper interface to outputs dict'''

    N = self.N 
    reg_lambda = 1e-4 /( (2**self.lod)**3)
    H = self.hessian/N + reg_lambda
    self.un = 1/H
    
    max_uncertainty = 6 #approximate upper bound of the function log10(1/(x+lambda)) when lambda=1e-4/(256^3) and x is the hessian
    min_uncertainty = -3 #approximate lower bound of that function (cutting off at hessian = 1000)
    density_fns_new = []
    num_fns = len(self.density_fns) 
    for i in self.density_fns:
        density_fns_new.append(lambda x, i=i: i(x) * (self.get_uncertainty(x)<= self.filter_thresh*max_uncertainty))
        
    if pkg_resources.get_distribution("nerfstudio").version >= "0.3.1":
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=density_fns_new)
    else:
        ray_samples,_, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=density_fns_new)
    field_outputs = self.field(ray_samples, compute_normals=self.config.predict_normals)
    points = ray_samples.frustums.get_positions()
    torch.save(points, "../visualize_points/ray_bundle.pt")
    un_points = self.get_uncertainty(points)

    #get weights
    density = field_outputs[FieldHeadNames.DENSITY] * (un_points <= self.filter_thresh*max_uncertainty)
    weights = ray_samples.get_weights(density)

    uncertainty = torch.sum(weights * un_points, dim=-2) 
    uncertainty += (1-torch.sum(weights,dim=-2)) * min_uncertainty #alpha blending
    
    #normalize into acceptable range for rendering
    uncertainty = torch.clip(uncertainty, min_uncertainty, max_uncertainty)
    uncertainty = (uncertainty-min_uncertainty)/(max_uncertainty-min_uncertainty)
    
    if self.white_bg:
        self.renderer_rgb.background_color=colors.WHITE    
    elif self.black_bg:
        self.renderer_rgb.background_color=colors.BLACK        
    rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
    depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
    accumulation = self.renderer_accumulation(weights=weights)
    
    # this is based on https://arxiv.org/pdf/2211.12656.pdf and the summation is not normalized. Check normalized in the viewer.
    # Uniform sampler (weights_list[0]) is used for getting uniform samples on each ray frustrum and find sum of entropy
    # of ray termination probabilities. Change sum to average for a more correct form. 
    ww = torch.clamp(weights_list[0], 1e-10 ,1.)
    entropy = -torch.sum(ww* torch.log2(ww) + (1-ww) * torch.log2(1-ww), dim=1)
   
    original_outputs = {
        "rgb": rgb,
        "accumulation": accumulation,
        "depth": depth,
    }
    
    original_outputs['uncertainty'] = uncertainty 
    original_outputs['entropy'] = entropy

    if self.config.predict_normals:
        normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
        pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
        original_outputs["normals"] = self.normals_shader(normals)
        original_outputs["pred_normals"] = self.normals_shader(pred_normals)    

    if self.training and self.config.predict_normals:
        original_outputs["rendered_orientation_loss"] = orientation_loss(
            weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
        )

        original_outputs["rendered_pred_normal_loss"] = pred_normal_loss(
            weights.detach(),
            field_outputs[FieldHeadNames.NORMALS].detach(),
            field_outputs[FieldHeadNames.PRED_NORMALS],
        )

    for i in range(self.config.num_proposal_iterations):
        original_outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

    return original_outputs

import time
class Temp:
    pipeline: Pipeline
    count: int = 0
    camera = None
    prev = -4.0
    def plus_one(self):
        Temp.count += 1
        return
    @staticmethod
    def next_camera():
        current = time.perf_counter()
        if current > (Temp.prev+3.0):
            Temp.camera, batch = Temp.pipeline.datamanager.next_train(0)
            Temp.prev = current
        return

def get_unc(self, outputs):

    points = outputs["xw"]
    weights = outputs["cDiff"]
    colors = outputs["rgb"]
    rows = colors.shape[0]
    cols = colors.shape[1]

    aabb = self.scene_box.aabb.to(points.device)
    inds, coeffs = find_grid_indices(points, aabb, None ,self.lod, points.device, zero_out=False) # distortion = 

    cfs_2 = (coeffs**2)/torch.sum((coeffs**2),dim=0, keepdim=True)
    uns = self.un[inds.long()] #[8,N]
    un_points = torch.sqrt(torch.sum((uns*cfs_2),dim=0)).view(num_iPts,rows,cols).permute(1,2,0)

    #for stability in volume rendering we use log uncertainty
    un_points = torch.log10(un_points+1e-12)
    max_uncertainty = 6 # approximate upper bound of the function log10(1/(x+lambda)) when lambda=1e-4/(256^3) and x is the hessian
    min_uncertainty = -1 # approximate lower bound of that function (cutting off at hessian = 1000)

    # un_points_normalized = (un_points-un_points.min())/(un_points.max()-un_points.min())
    # un_points_weighted = un_points.sum(dim=2)/NUM_iPOINTS
    # un_points_weighted = (weights * un_points).sum(dim=2)
    #un_points_weighted = ((1-weights) * un_points + weights*min_uncertainty).sum(dim=2)
    #uncertainty = un_points_weighted.view((points.shape[0], points.shape[1],1))
    mask = outputs["gIndex"] > -1
    uncertainty = (mask[:,:,0] == False).float()*6
    for i in range(0,num_iPts):
        prod = 1
        for j in range(0,i):
            prod *= (1-weights[:,:,j])
        uncertainty[:,:] += weights[:,:,i]*un_points[:,:,i]*prod*mask[:,:,i]
    uncertainty = uncertainty.unsqueeze(-1)
    uncertainty = un_points
    #uncertainty = (weights[:,:,0]*un_points[:,:,0]
    #               +weights[:,:,1]*un_points[:,:,1]*(1-weights[:,:,0])#).unsqueeze(-1)
    #               +weights[:,:,2]*un_points[:,:,2]*(1-weights[:,:,1])*(1-weights[:,:,0])).unsqueeze(-1)
    
    #print(f"Max: {uncertainty.max()}, Min: {uncertainty.min()}, Weight max: {weights.max()}, Weight min: {weights.min()}")
    #normalize into acceptable range for rendering
    uncertainty = torch.clip(uncertainty, min_uncertainty, max_uncertainty)
    uncertainty = (uncertainty-min_uncertainty)/(max_uncertainty-min_uncertainty)
    #uncertainty = (uncertainty-uncertainty.min())/(uncertainty.max()-uncertainty.min())

    # TODO: partial alpha blending.  Before summing color across rays, weight the color
    filter = 1 #weights * (uncertainty < self.filter_thresh)
    #colors[:,:,:,0] = colors[:,:,:,0] *  filter
    #colors[:,:,:,1] = colors[:,:,:,1] *  filter
    #colors[:,:,:,2] = colors[:,:,:,2] *  filter

    return uncertainty, colors#.sum(dim=2).clamp(max=1)   

def get_weights(xw, contrib, num_iPts):

    #return contrib

    diff = torch.diff(xw, dim=2)
    half_diff = torch.norm(diff, dim=3)/2
    dist1 = torch.cat([torch.zeros(half_diff.shape[0], half_diff.shape[1], 1).to("cuda"), half_diff[:,:,:]], dim=-1)
    dist2 = torch.cat([half_diff[:,:,:], torch.zeros(half_diff.shape[0], half_diff.shape[1], 1).to("cuda")], dim=-1)
    delta_density = (dist1+dist2) * contrib
    alphas = 1 - torch.exp(-delta_density)

    # TODO: investigate transmittance
    #transmittance = torch.cumsum(delta_density[..., :-1, :], dim=-1)
    ## start cumulutive sum lists at zero
    #transmittance = torch.cat(
    #    [torch.zeros((*transmittance.shape[:1], 1, num_iPts), device=delta_density.device), transmittance], dim=-2
    #)
#
    ## transmittance = torch.exp(-transmittance)  # [..., "num_samples"]
    #weights = alphas * transmittance  # [..., "num_samples"]
    #weights = torch.nan_to_num(weights)
    #weights = (weights-weights.min())/(weights.max()-weights.min())
    return alphas

def get_output_splatfacto_new(self, in_camera):
    ''' reimplementation of get_output function from models because of lack of proper interface to outputs dict'''
    
    TEMPCAM = False

    if TEMPCAM:
        if Temp.camera == None:
            Temp.camera, batch = Temp.pipeline.datamanager.next_train(0)
        #Temp.next_camera()
        camera = Temp.camera
    else:
        camera = in_camera

    # test = Test()

    TEST = False

    # TEST VALUES
    # means3D =  torch.zeros_like(self.means) # * torch.tensor([0, -500, 0], device="cuda")
    # opacity = torch.ones_like(self.opacities) 
    # scales =  torch.ones_like(self.scales) * .5
    # rotation = torch.tensor([0.0,0.0,0.0,1.0], device="cuda").repeat(self.means.shape[0], 1, 1)
    # means2D = torch.zeros_like(self.means)

    if self.splatFlag:
        outputs = self.renderer_rgb(camera)
    else:
        outputs ={}
        raster_pkg = get_rasterizer_output(self, camera)
        pckg = sort_package(raster_pkg, camera)
        #get_weights(outputs["xw"], weights, num_iPts)
        outputs["rgb"] = pckg["rgb"]
        outputs["depth"] = pckg["depth"]
        temp = pckg["uncertainty"]
        mean = temp.mean()
        std = temp.std()
        temp = torch.clamp(temp, min=(mean-std), max=(mean+std))
        outputs["uncertainty"] = (temp - temp.min())/(temp.max() - temp.min())
        outputs["contr_count"] = pckg["alpha"]

        if self.renderImg:
            from PIL import Image
            output_image = outputs["rgb"].detach().cpu().numpy()
            output_root = "/home/damian/projects/nerfstudio/renders/"
            output_dir = output_root
            file_name = "without_filter"+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            if self.xg_thresh != 0.0:
                file_name = "with_filter"+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_image = (output_image * 255).astype(np.uint8)  # Rescale to 0-255 for saving
            output_image_pil = Image.fromarray(output_image)
            output_image_path = os.path.join(output_dir, file_name+'.png')
            output_image_pil.save(output_image_path)

            cam_data ={
                "camera_path": [
                    {
                        "camera_to_world": camera.camera_to_worlds.tolist(),
                    "fov": 50,
                    "aspect": 1
                    }
                ]
            }
            output_json_path = os.path.join(output_dir, file_name+'.json')
            with open(output_json_path, "w") as f:
                json.dump(cam_data, f, indent=1)


        #outputs["uncertainty"], outputs["filtered_rgb"] = get_unc(self, pckg) #[:H*W,0].reshape(H,W,1)

    return outputs


def get_output_fn(model):

    if isinstance(model, NerfactoModel):
        return get_output_nerfacto_new
    elif isinstance(model, SplatfactoModel):
        return get_output_splatfacto_new
    else:
        raise Exception("Sorry, this model is not currently supported.")

        
@dataclass
class ViewerConfigWithoutNumRays(ViewerConfig):
    """Configuration for viewer instantiation"""

    num_rays_per_chunk: tyro.conf.Suppress[int] = -1

    def as_viewer_config(self):
        """Converts the instance to ViewerConfig"""
        return ViewerConfig(**{x.name: getattr(self, x.name) for x in fields(self)})


@dataclass
class RunViewerU:
    """Load a checkpoint and start the viewer."""
    load_config: Path
    """Path to config YAML file."""
    viewer: ViewerConfigWithoutNumRays = field(default_factory=ViewerConfigWithoutNumRays)
    """Viewer configuration"""
    white_bg: bool = True
    """ Render empty space as white when filtering""" 
    black_bg: bool = False
    """ Render empty space as black when filtering""" 
    use_splat: bool = False
    """ Render splatfacto with splats or rays"""

    def main(self) -> None:
        """Main function."""
        if pkg_resources.get_distribution("nerfstudio").version >= "0.3.1":
            config, pipeline, _, step = eval_setup(
                self.load_config,
                eval_num_rays_per_chunk=None,
                test_mode="test",
            )
        else:
            config, pipeline, _ = eval_setup(
                self.load_config,
                eval_num_rays_per_chunk=None,
                test_mode="test",
            )
            step = 0
        
        self.device = pipeline.device
        pipeline.model.white_bg = self.white_bg
        pipeline.model.black_bg = self.black_bg

        if isinstance(pipeline.model, SplatfactoModel):
            Temp.pipeline = pipeline
            #pipeline.model.compute_3D_filter(pipeline)
        # save_cameras(pipeline)
        # save_ply(pipeline.model, "outputs/splatfacto_train_points3d.ply")
        new_method = get_output_fn(pipeline.model)
        pipeline.model.get_outputs = types.MethodType(new_method, pipeline.model)
          
        num_rays_per_chunk = config.viewer.num_rays_per_chunk
        assert self.viewer.num_rays_per_chunk == -1
        config.vis = "viewer"
        config.viewer = self.viewer.as_viewer_config()
        config.viewer.num_rays_per_chunk = num_rays_per_chunk
    
        self._start_viewer(config, pipeline, step)

    def save_checkpoint(self, *args, **kwargs):
        """
        Mock method because we pass this instance to viewer_state.update_scene
        """
    def _update_viewer_state(self, viewer_state: viewer_utils.ViewerState, pipeline: Pipeline):
        """Updates the viewer state by rendering out scene with current pipeline
        Returns the time taken to render scene.

        """
        # NOTE: step must be > 0 otherwise the rendering would not happen
        step = 1
        num_rays_per_batch = pipeline.datamanager.get_train_rays_per_batch()
        with TimeWriter(writer, EventName.ITER_VIS_TIME) as _:
            viewer_state.update_scene = types.MethodType(update_scene_new, viewer_state)
            try:
                viewer_state.update_scene(self, step, pipeline.model, num_rays_per_batch)
            except RuntimeError:
                time.sleep(0.03)  # sleep to allow buffer to reset
                assert viewer_state.vis is not None
                viewer_state.vis["renderingState/log_errors"].write(
                    "Error: GPU out of memory. Reduce resolution to prevent viewer from crashing."
                )

    def _start_viewer(self, config: TrainerConfig, pipeline: Pipeline, step: int):
        """Starts the viewer

        Args:
            config: Configuration of pipeline to load
            pipeline: Pipeline instance of which to load weights
            step: Step at which the pipeline was saved
        """
        base_dir = config.get_base_dir()
        viewer_log_path = base_dir / config.viewer.relative_log_filename

        if pkg_resources.get_distribution("nerfstudio").version >= "0.3.1":
            viewer_state = ViewerLegacyState(
            config.viewer,
            log_filename=viewer_log_path,
            datapath=pipeline.datamanager.get_datapath(),
            pipeline=pipeline,
            )
            viewer_state.control_panel._filter3D = ViewerCheckbox(
                    name="3D Filter for Opacity",
                    default_value=False,
                )
            viewer_state.control_panel._renderImg = ViewerCheckbox(
                    name="Render Image",
                    default_value=False,
                )

            viewer_state.control_panel._two_pass = ViewerCheckbox(
                    name="Two-Pass Filter",
                    default_value=False,
                )
            viewer_state.control_panel._filt3D_scale = ViewerSlider(
                   "filter 3D scale",
                    default_value=0.2,
                    min_value=0.0,
                    max_value=1.0,
                    step=0.01,
                    hint="Filtering threshold for uncertain areas.",
                )
            viewer_state.control_panel.d_thresh = ViewerSlider(
                   "depth thresh",
                    default_value=0.0,
                    min_value=0.0,
                    max_value=1.0,
                    step=0.01,
                    hint="Filtering threshold for uncertain areas.",
                )
            viewer_state.control_panel._filter = ViewerSlider(
                    "Filter Threshold",
                    default_value=1.,
                    min_value=0.0,
                    max_value=1,
                    step=0.05,
                    hint="Filtering threshold for uncertain areas.",
                )
            viewer_state.control_panel.xg_filter = ViewerSlider(
                   "xg threshold",
                    default_value=0.0,
                    min_value=0.0,
                    max_value=1.0,
                    step=0.01,
                    hint="Filtering threshold for uncertain areas.",
                )
            viewer_state.control_panel.m_filter = ViewerSlider(
                    "mean threshold",
                    default_value=0.0,
                    min_value=0.0,
                    max_value=9999.0,
                    step=1.0,
                    hint="Filtering threshold for visible Gaussian depth.",
                )
            viewer_state.control_panel.s_filter = ViewerSlider(
                    "scale threshold",
                    default_value=0.0,
                    min_value=0.0,
                    max_value=1.0,
                    step=0.01,
                    hint="Filtering threshold for number of Guassian contributing to a pixel ray.",
                )
            viewer_state.control_panel.cc_filter = ViewerSlider(
                    "gaussian/pixel ratio",
                    default_value=0.5,
                    min_value=0.0,
                    max_value=1.0,
                    step=0.01,
                    hint="Filtering threshold for number of Guassian contributing to a pixel ray.",
                )
            viewer_state.control_panel.add_element(viewer_state.control_panel._filter3D)
            viewer_state.control_panel.add_element(viewer_state.control_panel._renderImg)
            viewer_state.control_panel.add_element(viewer_state.control_panel._two_pass)
            banner_messages = [f"Viewer at: {viewer_state.viewer_url}"]
        else:
            viewer_state, banner_messages = viewer_utils.setup_viewer(
                config.viewer, log_filename=viewer_log_path, datapath=pipeline.datamanager.get_datapath()
            )

        # We don't need logging, but writer.GLOBAL_BUFFER needs to be populated
        config.logging.local_writer.enable = False
        writer.setup_local_writer(config.logging, max_iter=config.max_num_iterations, banner_messages=banner_messages)

        assert viewer_state and pipeline.datamanager.train_dataset

        nerfstudio_version = pkg_resources.get_distribution("nerfstudio").version

        if nerfstudio_version >= "0.3.1":
            if nerfstudio_version >=  "0.3.3":
                viewer_state.init_scene(
                    train_dataset=pipeline.datamanager.train_dataset,
                    train_state="completed",
                )
            else:
                viewer_state.init_scene(
                    dataset=pipeline.datamanager.train_dataset,
                    train_state="completed",
                )
            viewer_state.viser_server.set_training_state("completed")
            viewer_state.update_scene(step=step)
            while True:
                time.sleep(0.01)
                pipeline.model.filter3D = viewer_state.control_panel._filter3D.value
                pipeline.model.renderImg = viewer_state.control_panel._renderImg.value
                pipeline.model.two_pass = viewer_state.control_panel._two_pass.value
                pipeline.model.filter3D_scale = viewer_state.control_panel._filt3D_scale.value
                pipeline.model.d_thresh = viewer_state.control_panel.d_thresh.value
                pipeline.model.xg_thresh = viewer_state.control_panel.xg_filter.value
                pipeline.model.m_thresh = viewer_state.control_panel.m_filter.value
                pipeline.model.s_thresh = viewer_state.control_panel.s_filter.value
                pipeline.model.cc_thresh = viewer_state.control_panel.cc_filter.value
        else:
            viewer_state.init_scene(
                dataset=pipeline.datamanager.train_dataset,
                start_train=False,
            )   
            while True:
                viewer_state.vis["renderingState/isTraining"].write(False)
                self._update_viewer_state(viewer_state, pipeline)

                
# this function is redefined to allow support of filter threshold slider in the viewer for old nerfstudio versions.
# The "Train Util." slider in eval time will control the filter threshold instead. 
def update_scene_new(self, trainer, step: int, graph: Model, num_rays_per_batch: int) -> None:
    """updates the scene based on the graph weights

    Args:
        step: iteration step of training
        graph: the current checkpoint of the model
    """
    has_temporal_distortion = getattr(graph, "temporal_distortion", None) is not None
    self.vis["model/has_temporal_distortion"].write(str(has_temporal_distortion).lower())

    is_training = self.vis["renderingState/isTraining"].read()
    self.step = step

    self._check_camera_path_payload(trainer, step)
    self._check_populate_paths_payload(trainer, step)

    camera_object = self._get_camera_object()
    if camera_object is None:
        return

    if is_training is None or is_training:
        # in training mode

        if self.camera_moving:
            # if the camera is moving, then we pause training and update camera continuously

            while self.camera_moving:
                self._render_image_in_viewer(camera_object, graph, is_training)
                camera_object = self._get_camera_object()
        else:
            # if the camera is not moving, then we approximate how many training steps need to be taken
            # to render at a FPS defined by self.static_fps.

            if EventName.TRAIN_RAYS_PER_SEC.value in GLOBAL_BUFFER["events"]:
                train_rays_per_sec = GLOBAL_BUFFER["events"][EventName.TRAIN_RAYS_PER_SEC.value]["avg"]
                target_train_util = self.vis["renderingState/targetTrainUtil"].read()
                if target_train_util is None:
                    target_train_util = 0.9

                batches_per_sec = train_rays_per_sec / num_rays_per_batch

                num_steps = max(int(1 / self.static_fps * batches_per_sec), 1)
            else:
                num_steps = 1

            if step % num_steps == 0:
                self._render_image_in_viewer(camera_object, graph, is_training)

    else:
        # in pause training mode, enter render loop with set graph
        local_step = step
        run_loop = not is_training
        while run_loop:
            # if self._is_render_step(local_step) and step > 0:
            if step > 0:
                self._render_image_in_viewer(camera_object, graph, is_training)
                camera_object = self._get_camera_object()
            th =  self.vis["renderingState/targetTrainUtil"].read() 
            graph.filter_thresh = th if th is not None else 1.    
            is_training = self.vis["renderingState/isTraining"].read()
            self._check_populate_paths_payload(trainer, step)
            self._check_camera_path_payload(trainer, step)
            run_loop = not is_training
            local_step += 1
                

def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(RunViewerU).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(RunViewerU)  # noqa