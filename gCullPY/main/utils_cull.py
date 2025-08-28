import torch
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from gCullUTILS.rich_utils import get_progress
from gCullPY.data.utils.dataloaders import FixedIndicesEvalDataloader
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

# ---- small utilities ----
def normalize(v, eps=1e-12):
    return v / (torch.linalg.norm(v, dim=-1, keepdim=True) + eps)

def gather_rows(mat, idx):  # mat: (N, C...), idx: (M,) or (M,k)
    # returns (M, C...) or (M,k,C...)
    if idx.ndim == 1:
        return mat[idx]
    # idx is (M,k)
    flat = mat[idx.reshape(-1)]
    return flat.view(idx.shape + mat.shape[1:])

def weighted_mean(nei_vals, w):  # nei_vals: (M,k,...) ; w: (M,k)
    while w.ndim < nei_vals.ndim:
        w = w.unsqueeze(-1)
    return (w * nei_vals).sum(dim=1)

def quat_to_mat(q):
            # q assumed (w,x,y,z) or (x,y,z,w)? If your repo uses (x,y,z,w), swap below accordingly.
            # Here we assume (w,x,y,z). Adjust if needed.
            w,x,y,z = q.unbind(-1)
            xx,yy,zz = x*x, y*y, z*z
            wx,wy,wz = w*x, w*y, w*z
            xy,xz,yz = x*y, x*z, y*z
            R = torch.stack([
                1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy),
                2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx),
                2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)
            ], dim=-1).reshape(-1,3,3)
            return R

def mat_to_quat(R):
    # Returns (w,x,y,z) - splatfacto/nerfstudio convention
    m00,m01,m02 = R[...,0,0], R[...,0,1], R[...,0,2]
    m10,m11,m12 = R[...,1,0], R[...,1,1], R[...,1,2]
    m20,m21,m22 = R[...,2,0], R[...,2,1], R[...,2,2]
    t = 1 + m00 + m11 + m22
    w = torch.sqrt(torch.clamp(t, 1e-12)) / 2
    x = (m21 - m12) / (4*w + 1e-12)
    y = (m02 - m20) / (4*w + 1e-12)
    z = (m10 - m01) / (4*w + 1e-12)
    q = torch.stack([w,x,y,z], dim=-1)
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-12)
    return q
 # ---- -------------- ----

def get_tanHalfFov(camera):
     # calculate the FOV of the camera given fx and fy, width and height
    px = camera.image_width.item() # pixel width
    py = camera.image_height.item() # pixel height
    fx = camera.fx.item() # focal width
    fy = camera.fy.item() # focal height
    tanHalfFovX = 0.5*px/fx # # comma makes output of equation a tuple that must be indexed
    tanHalfFovY = 0.5*py/fy # comma makes output of equation a tuple that must be indexed

    return tanHalfFovX, tanHalfFovY

def get_Rt_inv(camera):
    camera_to_world = camera.camera_to_worlds

    # shift the camera to center of scene looking at center
    R =  camera_to_world[:3, :3] #torch.eye(3, device="cuda") # 3 x 3
    T =  camera_to_world[:3, 3:4] #torch.tensor([[0.0],[0.0],[0.0]], device="cuda")  # 3 x 1
    
    # flip the z and y axes to align with gsplat conventions
    R_edit = torch.diag(torch.tensor([1, -1, -1], device=R.device, dtype=R.dtype))
    R = R @ R_edit
    # analytic matrix inverse to get world2camera matrix
    R_inv = R.T

    T_inv = -R_inv @ T
    return R_inv, T_inv
    
def get_viewmat(camera):
    R_inv, T_inv = get_Rt_inv(camera)
    viewmat = torch.eye(4, device=R_inv.device, dtype=R_inv.dtype) # viewmat = world to camera -> https://docs.gsplat.studio/main/conventions/data_conventions.html
    viewmat[:3, :3] = R_inv
    viewmat[:3, 3:4] = T_inv
    #viewmat = viewmat.T # transpose for GOF code

    #not sure if I need to do this
    # viewmat[:, 1:3] *= -1.0 # switch between openCV and openGL conventions?

    return viewmat

# taken from gaussian-opacity-fields
def get_full_proj_transform(tanHalfFovX, tanHalfFovY):
    zfar = 100.0
    znear = 0.01
    z_sign = 1.0

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    projMat = torch.zeros(4, 4, device="cuda")
    projMat[0, 0] = 2.0 * znear / (right - left)
    projMat[1, 1] = 2.0 * znear / (top - bottom)
    projMat[0, 2] = (right + left) / (right - left)
    projMat[1, 2] = (top + bottom) / (top - bottom)
    projMat[3, 2] = z_sign
    projMat[2, 2] = z_sign * zfar / (zfar - znear)
    projMat[2, 3] = -(zfar * znear) / (zfar - znear)
    return projMat #(viewMat.unsqueeze(0).bmm(projMat.transpose(0,1).unsqueeze(0))).squeeze(0)

def get_cull_list(model, camera, bool_mask):
    viewmat = get_viewmat(camera)
    tanHalfFovX, tanHalfFovY = get_tanHalfFov(camera)
    
    # use this for post-rendering image filters
    height =int(camera.height.item())
    width = int(camera.width.item())

    means3D =  model.means
    device     = means3D.device
    dtype      = means3D.dtype

    N = means3D.shape[0]
    X_h = torch.cat([means3D, torch.ones(N, 1, device=device, dtype=dtype)], dim=1)  # (N,4) homogenous coordinates
    
    projmatrix= get_full_proj_transform(tanHalfFovX, tanHalfFovY)
    FULL = projmatrix @ viewmat
    clip = (FULL @ X_h.t()).t()  
    w = clip[:, 3]
    eps = torch.finfo(dtype).eps
    x_ndc = clip[:, 0] / (w + eps)
    y_ndc = clip[:, 1] / (w + eps)
    in_ndc = (w > 0) & (x_ndc >= -1) & (x_ndc <= 1) & (y_ndc >= -1) & (y_ndc <= 1)
    
    u = ((x_ndc + 1) * 0.5) * (width  - 1)
    v = ((y_ndc + 1) * 0.5) * (height - 1)

    inside = in_ndc & (u >= 0) & (u <= width-1) & (v >= 0) & (v <= height-1)

    u_i = torch.clamp(u.round().long(), 0, width  - 1)
    v_i = torch.clamp(v.round().long(), 0, height - 1)

    m = bool_mask.to(means3D.device, dtype=torch.bool)
    on_black = torch.zeros(means3D.shape[0], dtype=torch.bool, device=means3D.device)
    on_black[inside] = ~m[v_i[inside], u_i[inside]]
    black_indices = torch.nonzero(on_black, as_tuple=False).squeeze(1)

    return u_i, v_i, on_black, black_indices

def get_mask(batch, mask_dir):
    img_idx = int(batch["image_idx"])+1 #add one for alignement
    mask_name = f"mask_{img_idx:05d}.png"
    mask_path = Path(mask_dir) / mask_name
    binary_mask = torch.tensor(np.array(Image.open(mask_path)))# convert to bool tensor for ease of CUDA hand-off where black = True / non-black = False
    #show_mask(bool_mask)
    return binary_mask

def modify_model(og_model, keep):
    with torch.no_grad():
        og_model.means.data = og_model.means[keep].clone()
        og_model.opacities.data = og_model.opacities[keep].clone()
        og_model.scales.data = og_model.scales[keep].clone()
        og_model.quats.data = og_model.quats[keep].clone()
        og_model.features_dc.data = og_model.features_dc[keep].clone()
        og_model.features_rest.data = og_model.features_rest[keep].clone()
    return og_model
    
def get_mask_dir(config):
    root = config.datamanager.data
    downscale_factor = config.datamanager.dataparser.downscale_factor 
    if downscale_factor > 1:
        mask_dir = root / f"masks_{downscale_factor}"
    else:
        mask_dir = root / "masks"
    return mask_dir

def statcull(pipeline):
    means3D     = pipeline.model.means.to("cpu") # cpu is faster for these operations
    center      = means3D.median(dim=0)
    std_dev     = means3D.std(dim=0)
    z_scores    = torch.abs((means3D - center.values) / std_dev)
    thr         = torch.tensor([.2, .2, 1.0])
    cull_mask   = (z_scores > thr).any(dim=1)
    return cull_mask

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

def cull_loop(config, pipeline, debug=False):

    render_dir = config.datamanager.data / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = config.datamanager.data / "masks" #get_mask_dir(config)
    keep_lst_master = torch.zeros(pipeline.model.means.shape[0], dtype=torch.bool)
    config.datamanager.dataparser.downscale_factor = 1
    for split in "train+test".split("+"):
        # FIX THIS
        if split != "train":
            break
        dataset, dataloader = build_loader(config, split, pipeline.device)
        desc = f"\u2702\ufe0f\u00A0 Culling split {split} \u2702\ufe0f\u00A0"

        with get_progress(desc) as progress:
            for idx, (camera, batch) in enumerate(progress.track(dataloader, total=len(dataset))):
                with torch.no_grad():
                    #frame_idx = batch["image_idx"]
                    camera.camera_to_worlds = camera.camera_to_worlds.squeeze() # splatoff rasterizer requires cam2world.shape = [3,4]
                    bool_mask = get_mask(batch, mask_dir)
                    u_i, v_i, keep_lst, _ = get_cull_list(pipeline.model, camera, bool_mask)
                    keep_lst_master |= keep_lst.to("cpu")
                    if debug:
                        visualize_mask_and_points(u_i[keep_lst], v_i[keep_lst], bool_mask)
                        print(f"{idx}: {keep_lst.sum().item()}/{keep_lst_master.sum().item()}")

    return keep_lst_master

def visualize_mask_and_points(u_i, v_i, bool_mask):
    mask_np = bool_mask.cpu().numpy()
    u_np = u_i.cpu().numpy()
    v_np = v_i.cpu().numpy()
    
    plt.figure(figsize=(12, 6))

    # Plot 1: Original mask
    plt.subplot(1, 2, 1)
    plt.imshow(mask_np, cmap='gray')
    plt.title("Original Car Mask")
    plt.colorbar()

    # Plot 2: Mask with projected points overlaid
    plt.subplot(1, 2, 2)
    plt.imshow(mask_np, cmap='gray', alpha=0.7)
    plt.scatter(u_np, v_np, c='red', s=1, alpha=0.8)
    plt.title("Mask + Projected Points")
    plt.xlabel("u (width)")
    plt.ylabel("v (height)")
    plt.show()
    return

