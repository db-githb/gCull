import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from gCullPY.main.utils_main import build_loader
from gCullUTILS.rich_utils import get_progress


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
    model = og_model
    with torch.no_grad():
        og_model.means.data = model.means[keep].clone()
        og_model.opacities.data = model.opacities[keep].clone()
        og_model.scales.data = model.scales[keep].clone()
        og_model.quats.data = model.quats[keep].clone()
        og_model.features_dc.data = model.features_dc[keep].clone()
        og_model.features_rest.data = model.features_rest[keep].clone()
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

def find_ground_plane(model, plane_eps=0.02, n_ransac=1024):  # plane_eps = thickness threshold for ground plane (m)
    
    # ---------- World points ----------
    P_world = model.means                                        # (N,3)
    dev, dt = P_world.device, P_world.dtype
    N = P_world.shape[0]
    
    # ---------- Ground plane (RANSAC) ----------
    # Optional: world 'up' hint if you have it (e.g., torch.tensor([0,1,0], device=dev))
    up_hint = torch.tensor([0,0,1], device=dev, dtype=dt) #None
    n, d = fit_plane_ransac(P_world, n_iters=n_ransac, eps=plane_eps, up_hint=up_hint)

    if n is not None:
        #dist = torch.abs(P_world @ n + d)                        # (N,)
        #is_ground = dist <= plane_eps
        signed = P_world @ n + d # signed distance (upward positive)
        # delete anything strictly below the plane (optionally with a margin in meters)
        margin = 0.01                      # e.g., 0.01 to keep a 1 cm buffer above the plane
        below = signed < -margin          # True = below plane
        is_ground = ~below
    else:
        # fall back: no plane found -> don't remove by plane
        is_ground = torch.zeros(N, dtype=torch.bool, device=dev)
    
    return is_ground

@torch.no_grad()
def fit_plane_ransac(points, n_iters=1024, eps=0.02, up_hint=None, up_align=0.7):
    """
    points: (N,3) world coords
    Returns (n, d): unit normal and offset so that plane is {x | n·x + d = 0}
    eps: inlier distance (meters)
    up_hint: optional world 'up' vector to prefer a horizontal plane
    """
    device = points.device
    N = points.shape[0]
    best_inliers = -1
    best_n = None
    best_d = None

    for _ in range(n_iters):
        idx = torch.randint(0, N, (3,), device=device)
        p1, p2, p3 = points[idx]             # (3,3)
        v1, v2 = p2 - p1, p3 - p1
        n = torch.linalg.cross(v1, v2)
        norm = torch.linalg.norm(n) + 1e-12
        if norm < 1e-8:
            continue
        n = n / norm
        # make normal point roughly upward if we have a hint
        if up_hint is not None:
            if torch.dot(n, up_hint) < 0:
                n = -n
            if torch.abs(torch.dot(n, up_hint)) < up_align:
                continue
        d = -torch.dot(n, p1)
        dist = torch.abs(points @ n + d)
        inliers = (dist <= eps).sum().item()
        if inliers > best_inliers:
            best_inliers = inliers
            best_n, best_d = n, d

    return best_n, best_d