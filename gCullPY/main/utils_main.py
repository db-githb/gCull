import torch
import numpy as np
import re, yaml
import os
from PIL import Image
from typing_extensions import Literal, Tuple
from collections import OrderedDict
from pathlib import Path
from plyfile import PlyData

from gCullUTILS.rich_utils import CONSOLE, get_progress
from gCullPY.pipelines.base_pipeline import Pipeline
from gCullPY.pipelines.base_pipeline import VanillaPipelineConfig
from gCullPY.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig
from gCullPY.models.splatfacto import SplatfactoModelConfig
from gCullPY.data.dataparsers.colmap_dataparser import ColmapDataParserConfig
from gCullPY.data.datasets.base_dataset import InputDataset
from gCullPY.data.utils.dataloaders import FixedIndicesEvalDataloader
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from contextlib import contextmanager

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

def to_path(val):
        if isinstance(val, list):
            return Path(*val)
        return Path(val)

def load_ply(
        ply_path: str
):
    data_path = Path(ply_path).parent
    # 1) Define where your COLMAP data and images live.
    colmap_parser = ColmapDataParserConfig(
        colmap_path=(data_path / "colmap" / "sparse" / "0").resolve(),
        images_path=(data_path / "images").resolve(),
        load_3D_points=True,
        assume_colmap_world_coordinate_convention=True,
        auto_scale_poses=True,
        center_method="poses",
        downscale_factor=1 # simple_trainer from gsplat sets data_factor to 4
    )

    # 2) Build the datamanager config.
    dm_conf = FullImageDatamanagerConfig(
        data=data_path,
        dataset=InputDataset,
        dataparser=colmap_parser,
        cache_images="cpu",
        cache_images_type="uint8",
        camera_res_scale_factor=1.0,
    )

    # 3) Define a SplatFacto model config.
    model_conf = SplatfactoModelConfig(
        random_init=False,
        sh_degree=3,
    )

    # 4) Bundle into a vanilla pipeline config.
    config = VanillaPipelineConfig(
        datamanager=dm_conf,
        model=model_conf,
    )

    # 5) Instantiate the pipeline.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = config.setup(device=device, test_mode="inference")

    # 6) Read the PLY
    ply = PlyData.read(str(ply_path))
    verts = ply['vertex'].data

    # 7) Extract arrays
    means = torch.stack([
        torch.tensor(verts[name])
        for name in ('x','y','z')
    ], dim=1)
    scales = torch.stack([
        torch.tensor(verts[f"scale_{i}"])
        for i in range(3)
    ], dim=1)
    quats = torch.stack([
        torch.tensor(verts[f"rot_{i}"])
        for i in range(4)
    ], dim=1)
    opacities = torch.tensor(verts['opacity']).unsqueeze(-1) 
    f_dc = torch.stack([
        torch.tensor(verts[f"f_dc_{i}"]) 
        for i in range(pipeline.model.config.sh_degree)
    ],
    dim=1)
    f_rest = torch.stack([
        torch.tensor(verts[f"f_rest_{i}"]) 
        for i in range(45)
    ],
    dim=1).reshape(-1, 15, 3)

    # 8) Override model parameters
    pipeline.model.means.data = means.to(device)
    pipeline.model.scales.data = scales.to(device)
    pipeline.model.quats.data = quats.to(device)
    pipeline.model.opacities.data = opacities.to(device)
    pipeline.model.features_dc.data = f_dc.to(device)
    pipeline.model.features_rest.data = f_rest.to(device)
    
    return config, pipeline

def load_config(
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
        downscale_factor=data["pipeline"]["datamanager"]["dataparser"]["downscale_factor"]
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

    CONSOLE.log("Loading latest checkpoint from load_dir")
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

def write_ply(model_path, model):
    
    model_name = model_path.parent.name
    experiment_name = model_path.parts[1]  # e.g., 'my-experiment'
    filename = model_path.parent / f"{experiment_name}_{model_name}_culled.ply"
    count, map_to_tensors = setup_write_ply(model)

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
    
    return filename

def build_loader(config, split, device):
    test_mode = "train" if split == "train" else "test"

    with _disable_datamanager_setup(config.datamanager._target):  # pylint: disable=protected-access
        datamanager = config.datamanager.setup(
            test_mode=test_mode,
            device=device
        )
        
    dataset = getattr(datamanager, f"{split}_dataset", datamanager.eval_dataset)
    
    dataloader = FixedIndicesEvalDataloader(
        input_dataset=dataset,
        device=datamanager.device,
        num_workers=datamanager.world_size * 4,
    )
    
    return dataset, dataloader

def render_loop(model_path, config, pipeline):
        #df = config.datamanager.dataparser.downscale_factor 
        #pipeline.model.downscale_factor = df
        render_dir = model_path.parent.parts[1]
        output_dir = Path("renders") / f"{render_dir}"
        output_dir.mkdir(parents=True, exist_ok=True)
        idx = 1
        
        for split in "train+test".split("+"):
            dataset, dataloader = build_loader(config, split, pipeline.device,)
            desc = f":movie_camera: Rendering split {split} :movie_camera:"
            
            with get_progress(desc) as progress:
                for camera, _ in progress.track(dataloader, total=len(dataset)):
                    with torch.no_grad():
                        rgb_tensor = pipeline.model.get_outputs(camera)["rgb"]

                    # convert [C,H,W] float tensor in [0,1] to a H×W×3 uint8 image
                    img_np = (
                        rgb_tensor
                        .clamp(0.0, 1.0)
                        .mul(255)
                        .byte()
                        .cpu()
                        .numpy()
                    )
                    Image.fromarray(img_np).save(output_dir / f"frame_{idx:05d}.png")
                    idx+=1
        return output_dir