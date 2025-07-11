import torch
import numpy as np
import re, yaml
import os
from typing_extensions import Literal, Tuple
from collections import OrderedDict
from pathlib import Path

from gCullPY.utils.rich_utils import CONSOLE
from gCullPY.pipelines.base_pipeline import Pipeline
from gCullPY.pipelines.base_pipeline import VanillaPipelineConfig
from gCullPY.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig
from gCullPY.models.splatfacto import SplatfactoModelConfig
from gCullPY.data.dataparsers.colmap_dataparser import ColmapDataParserConfig
from gCullPY.data.datasets.base_dataset import InputDataset

def to_path(val):
        if isinstance(val, list):
            return Path(*val)
        return Path(val)

def eval_setup(
    config_path: Path,
    test_mode: Literal["test", "val", "inference"] = "test",
) -> Tuple[VanillaPipelineConfig, Pipeline]:

    # load save config
    txt = Path(config_path).read_text()
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
    loaded_state = torch.load(load_path, map_location="cpu",  weights_only=False)
    pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
    CONSOLE.print(f":white_check_mark: Done loading checkpoint from {load_path}")
    
    return config, pipeline

def setup_write_ply(inModel):
    model = inModel
    count = 0
    map_to_tensors = OrderedDict()

    with torch.no_grad():
        positions = model.means.cpu().numpy()
        count = positions.shape[0]
        n = count

        map_to_tensors["x"] = positions[:, 0]
        map_to_tensors["y"] = positions[:, 1]
        map_to_tensors["z"] = positions[:, 2]
        map_to_tensors["nx"] = np.zeros(n, dtype=np.float32)
        map_to_tensors["ny"] = np.zeros(n, dtype=np.float32)
        map_to_tensors["nz"] = np.zeros(n, dtype=np.float32)

        if model.config.sh_degree > 0:
            shs_0 = model.shs_0.contiguous().cpu().numpy()
            for i in range(shs_0.shape[1]):
                map_to_tensors[f"f_dc_{i}"] = shs_0[:, i, None]
            # transpose(1, 2) was needed to match the sh order in Inria version
            shs_rest = model.shs_rest.transpose(1, 2).contiguous().cpu().numpy()
            shs_rest = shs_rest.reshape((n, -1))
            for i in range(shs_rest.shape[-1]):
                map_to_tensors[f"f_rest_{i}"] = shs_rest[:, i, None]
        else:
            colors = torch.clamp(model.colors.clone(), 0.0, 1.0).data.cpu().numpy()
            map_to_tensors["colors"] = (colors * 255).astype(np.uint8)

        map_to_tensors["opacity"] = model.opacities.data.cpu().numpy()
        scales = model.scales.data.cpu().numpy()
        for i in range(3):
            map_to_tensors[f"scale_{i}"] = scales[:, i, None]
        quats = model.quats.data.cpu().numpy()
        for i in range(4):
            map_to_tensors[f"rot_{i}"] = quats[:, i, None]

    # post optimization, it is possible have NaN/Inf values in some attributes
    # to ensure the exported ply file has finite values, we enforce finite filters.
    select = np.ones(n, dtype=bool)
    for k, t in map_to_tensors.items():
        n_before = np.sum(select)
        select = np.logical_and(select, np.isfinite(t).all(axis=-1))
        n_after = np.sum(select)
        if n_after < n_before:
            CONSOLE.print(f"{n_before - n_after} NaN/Inf elements in {k}")
    if np.sum(select) < n:
        CONSOLE.print(f"values have NaN/Inf in map_to_tensors, only export {np.sum(select)}/{n}")
        for k, t in map_to_tensors.items():
            map_to_tensors[k] = map_to_tensors[k][select]
        count = np.sum(select)
    return count, map_to_tensors

def write_ply(filename, count, map_to_tensors):
        
    # Ensure count matches the length of all tensors
    if not all(len(tensor) == count for tensor in map_to_tensors.values()):
        raise ValueError("Count does not match the length of all tensors")
    
    # Type check for numpy arrays of type float or uint8 and non-empty
    if not all(
        isinstance(tensor, np.ndarray)
        and (tensor.dtype.kind == "f" or tensor.dtype == np.uint8)
        and tensor.size > 0
        for tensor in map_to_tensors.values()
    ):
        raise ValueError("All tensors must be numpy arrays of float or uint8 type and not empty")
    
    with open(filename, "wb") as ply_file:
        # Write PLY header
        ply_file.write(b"ply\n")
        ply_file.write(b"format binary_little_endian 1.0\n")
        ply_file.write(f"element vertex {count}\n".encode())

        # Write properties, in order due to OrderedDict
        for key, tensor in map_to_tensors.items():
            data_type = "float" if tensor.dtype.kind == "f" else "uchar"
            ply_file.write(f"property {data_type} {key}\n".encode())
        ply_file.write(b"end_header\n")

        # Write binary data
        # Note: If this is a performance bottleneck consider using numpy.hstack for efficiency improvement
        for i in range(count):
            for tensor in map_to_tensors.values():
                value = tensor[i]
                if tensor.dtype.kind == "f":
                    ply_file.write(np.float32(value).tobytes())
                elif tensor.dtype == np.uint8:
                    ply_file.write(value.tobytes())

