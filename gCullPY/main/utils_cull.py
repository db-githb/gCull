import torch
import numpy as np
from scipy.spatial import KDTree
from scipy.ndimage import binary_fill_holes

import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from gCullPY.main.utils_main import build_loader
from gCullUTILS.rich_utils import get_progress

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

def get_ground_gaussians(model, is_ground):
    ground_gaussians = {}
    ground_gaussians["means"] = model.means[is_ground]
    ground_gaussians["opacities"] = model.opacities[is_ground]
    ground_gaussians["scales"] = model.scales[is_ground]
    ground_gaussians["quats"] = model.quats[is_ground]
    ground_gaussians["features_dc"] = model.features_dc[is_ground]
    ground_gaussians["features_rest"] = model.features_rest[is_ground]
    return ground_gaussians

def modify_ground_gaussians(ground_gaussians, keep):
    ground_gaussians["means"]         = ground_gaussians["means"][keep]
    ground_gaussians["opacities"]     = ground_gaussians["opacities"][keep]
    ground_gaussians["scales"]        = ground_gaussians["scales"][keep]
    ground_gaussians["quats"]         = ground_gaussians["quats"][keep]
    ground_gaussians["features_dc"]   = ground_gaussians["features_dc"][keep]
    ground_gaussians["features_rest"] = ground_gaussians["features_rest"][keep]
    return ground_gaussians

def modify_model(og_model, keep):
    with torch.no_grad():
        og_model.means.data = og_model.means[keep].clone()
        og_model.opacities.data = og_model.opacities[keep].clone()
        og_model.scales.data = og_model.scales[keep].clone()
        og_model.quats.data = og_model.quats[keep].clone()
        og_model.features_dc.data = og_model.features_dc[keep].clone()
        og_model.features_rest.data = og_model.features_rest[keep].clone()
    return og_model

def append_gaussians_to_model(og_model, new_gaussians):
    with torch.no_grad():
        og_model.means.data = torch.cat((og_model.means.data, new_gaussians["means"]))
        og_model.opacities.data = torch.cat((og_model.opacities.data, new_gaussians["opacities"]))
        og_model.scales.data = torch.cat((og_model.scales.data, new_gaussians["scales"]))
        og_model.quats.data = torch.cat((og_model.quats.data, new_gaussians["quats"]))
        og_model.features_dc.data = torch.cat((og_model.features_dc.data, new_gaussians["features_dc"]))
        og_model.features_rest.data = torch.cat((og_model.features_rest.data, new_gaussians["features_rest"]))
    return og_model

def get_all_ground_gaussians(og_ground, new_ground):
    with torch.no_grad():
        og_ground["means"] = torch.cat((og_ground["means"], new_ground["means"]))
        og_ground["opacities"] = torch.cat((og_ground["opacities"], new_ground["opacities"]))
        og_ground["scales"] = torch.cat((og_ground["scales"], new_ground["scales"])) 
        og_ground["quats"] = torch.cat((og_ground["quats"], new_ground["quats"]))
        og_ground["features_dc"] = torch.cat((og_ground["features_dc"], new_ground["features_dc"]))
        og_ground["features_rest"] = torch.cat((og_ground["features_rest"], new_ground["features_rest"]))
    return og_ground
    

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
        dist = torch.abs(P_world @ n + d)                        # (N,)
        is_ground = dist <= plane_eps
        signed = P_world @ n + d # signed distance (upward positive)
        # delete anything strictly below the plane (optionally with a margin in meters)
        margin = 0.01                      # e.g., 0.01 to keep a 1 cm buffer above the plane
        below = signed < -margin          # True = below plane
        keep = ~below
    else:
        # fall back: no plane found -> don't remove by plane
        keep = torch.ones(N, dtype=torch.bool, device=dev)
        is_ground = torch.zeros(N, dtype=torch.bool, device=dev)
    
    return keep, is_ground, n, d

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

@torch.no_grad()
def densify_ground_plane_jitter(ground_gaussians, n, d,
                                samples_per_point=2,
                                spacing_percentile=0.5,
                                expand_scale=1.2,       # >1 grows outward more
                                normal_jitter=0.0,      # e.g. 0.003 (meters)
                                subsample_cap=20000):
    """
    points        : (N,3) world coords (float, CUDA ok)
    is_ground     : (N,) bool mask (True = ground)
    n, d          : plane {x | n·x + d = 0}, with ||n||=1
    Returns:
        new_pts   : (M,3) sampled points on/near the plane
    """
    Pg = ground_gaussians["means"]
    dev, dt = Pg.device, Pg.dtype              
    if Pg.numel() == 0:
        return torch.empty(0, 3, device=dev, dtype=dt)

    # --- plane basis (u, v) ---
    # pick a non-parallel vector to n
    a = torch.tensor([1., 0., 0.], device=dev, dtype=dt)
    if torch.abs((a @ n).abs() - 1.0) < 1e-6:
        a = torch.tensor([0., 1., 0.], device=dev, dtype=dt)
    u = torch.linalg.cross(n, a); u = u / (u.norm() + 1e-12)
    v = torch.linalg.cross(n, u)

    # --- estimate typical spacing r0 (median N.N. distance on a subsample) ---
    G = Pg.shape[0]
    if G > subsample_cap:
        idx = torch.randperm(G, device=dev)[:subsample_cap]
        Q = Pg[idx]
    else:
        Q = Pg
    # pairwise distance to get nearest neighbor (~O(S^2); OK for capped S)
    with torch.amp.autocast('cuda', enabled=False):
        D = torch.cdist(Q, Q, p=2)
    D[D == 0] = float('inf')
    nn = D.min(dim=1).values
    r0 = torch.quantile(nn, spacing_percentile, interpolation='nearest')  # e.g., median
    r_max = expand_scale * r0

    # --- sample K points per ground mean, uniform in disk radius r∈[0,r_max] ---
    G = Pg.shape[0]
    K = samples_per_point
    if K <= 0:
        return torch.empty(0, 3, device=dev, dtype=dt)
    # polar sampling: radius ~ sqrt(U) for uniform disk, theta ~ U[0, 2π)
    U = torch.rand(G, K, device=dev, dtype=dt)
    R = r_max * torch.sqrt(U)
    TH = 2.0 * torch.pi * torch.rand(G, K, device=dev, dtype=dt)
    # offsets in plane: r*cosθ u + r*sinθ v
    off = (R * torch.cos(TH)).unsqueeze(-1) * u + (R * torch.sin(TH)).unsqueeze(-1) * v  # (G,K,3)

    if normal_jitter > 0:
        off = off + normal_jitter * torch.randn_like(off) * n

    new_pts = (Pg.unsqueeze(1) + off).reshape(-1, 3)

    # Prepare output dict
    out = {k: v for k, v in ground_gaussians.items()}

    # align new gaussians with ground plane
    R = basis_from_normal(n)
    #_, quats = ensure_minor_axis_is_normal(R, ground_gaussians["scales"])
    quats = mat_to_quat(R)
    N = new_pts.shape[0]
    
    # Append attributes
    out["means"] = new_pts
    out["features_dc"] = ground_gaussians["features_dc"].repeat_interleave(K, dim=0)
    out["features_rest"] = ground_gaussians["features_rest"].repeat_interleave(K, dim=0)
    out["opacities"] = ground_gaussians["opacities"].repeat_interleave(K, dim=0)
    out["quats"] = quats.unsqueeze(0).expand(N, -1).clone()
    out["scales"] = move_minor_scale_to_axis(ground_gaussians["scales"].repeat_interleave(K,0), normal_axis=2)

    return out

def move_minor_scale_to_axis(scales, normal_axis=2):
    """
    scales: (N,3)   (log or linear; this only permutes slots)
    normal_axis: 0=x,1=y,2=z -> slot that should hold the *smallest* value
    """
    N = scales.shape[0]
    min_idx = scales.argmin(dim=1)                       # (N,)
    order = torch.arange(3, device=scales.device).expand(N,3).clone()
    rows = (min_idx != normal_axis).nonzero(as_tuple=False).squeeze(1)
    if rows.numel():
        i = normal_axis
        j = min_idx[rows]
        tmp = order[rows, i].clone()
        order[rows, i] = order[rows, j]
        order[rows, j] = tmp
    return scales.gather(1, order)

def ensure_minor_axis_is_normal(R, scales, normal_axis=1):
    assert scales.ndim == 2 and scales.shape[1] == 3
    N = scales.shape[0]

    # Broadcast a single 3x3 to all rows if needed
    if R.ndim == 2:
        assert R.shape == (3,3)
        R = R.unsqueeze(0).expand(N, -1, -1).contiguous()
    else:
        assert R.shape == (N,3,3)

    # For each row, find which local axis is currently the smallest
    min_idx = scales.argmin(dim=1)  # (N,)

    # Build per-row permutation that swaps min_idx with normal_axis when needed
    order = torch.arange(3, device=scales.device).expand(N, 3).clone()
    rows = (min_idx != normal_axis).nonzero(as_tuple=False).squeeze(1)
    if rows.numel() > 0:
        i = normal_axis
        j = min_idx[rows]
        tmp = order[rows, i].clone()
        order[rows, i] = order[rows, j]
        order[rows, j] = tmp

    # Permute columns of R and entries of scales consistently
    Rp      = R.gather(2, order.unsqueeze(1).expand(-1, 3, -1))   # (N,3,3)
     # if you store quats, recompute them from the permuted R
    quats_p = mat_to_quat(Rp)
    return Rp, quats_p

def basis_from_normal(n, roll=0.0, axis='y'):
    """
    n: (3,) plane normal (not necessarily unit)
    roll: in-plane rotation (radians) around n
    axis: which local axis should align with n: 'x' | 'y' | 'z'
    Returns: R (3,3) with columns = world directions of local axes
    """
    n = normalize(n)
    # choose a reference not parallel to n
    ref = torch.tensor([0.0, 0.0, 1.0], dtype=n.dtype, device=n.device)
    if torch.abs((n * ref).sum()) > 0.999:  # nearly parallel
        ref = torch.tensor([0.0, 1.0, 0.0], dtype=n.dtype, device=n.device)

    # tangent basis (x_tan, y_tan)
    x_tan = normalize(torch.linalg.cross(ref, n))
    y_tan = torch.linalg.cross(n, x_tan)

    # apply in-plane roll
    c, s = torch.cos(torch.tensor(roll, dtype=n.dtype, device=n.device)), torch.sin(torch.tensor(roll, dtype=n.dtype, device=n.device))
    u = x_tan * c + y_tan * s
    v = -x_tan * s + y_tan * c

    if axis == 'z':      # local z -> n
        R = torch.stack([u, v, n], dim=1)
    elif axis == 'y':    # local y -> n
        R = torch.stack([u, n, v], dim=1)
    elif axis == 'x':    # local x -> n
        R = torch.stack([n, u, v], dim=1)
    else:
        raise ValueError("axis must be 'x','y','z'")
    return R

@torch.no_grad()
def fill_hole_with_known_plane(points_xyz,
                               plane_n,
                               plane_d=None,
                               plane_point=None,
                               keep_ratio=1.0,
                               k_nn=100,
                               seed=0):
    """
    points_xyz: (N,3) np.array of points on (or near) the plane, with a hole
    plane_n:    (3,) plane normal (unit or not; will be normalized)
    plane_d:    scalar d for plane n·x + d = 0  (optional if plane_point given)
    plane_point:(3,) a known point on the plane (optional if plane_d given)
    keep_ratio: 1.0 ~ match original density; <1.0 = denser, >1.0 = sparser
    k_nn:       k for local spacing estimate (use 6–8)
    returns:
      filled_pts: (N+M,3) original + newly filled points
      new_pts:    (M,3) just the added points
    """
    P = np.asarray(points_xyz, float)
    n = np.asarray(plane_n, float)
    n = n / (np.linalg.norm(n) + 1e-12)

    if plane_point is None:
        if plane_d is None:
            raise ValueError("Provide either plane_d or plane_point.")
        plane_d = np.asarray(plane_d, float)
        p0 = -plane_d * n  # a point on the plane
    else:
        p0 = np.asarray(plane_point, float)

    # Orthonormal basis (u, v) spanning the plane
    a = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(a, n)) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    u = np.cross(n, a); u /= (np.linalg.norm(u) + 1e-12)
    v = np.cross(n, u)  # already unit

    # 1) Project points to 2D coords on the plane (origin p0)
    X = P - p0
    XY = np.stack([X @ u, X @ v], axis=1)  # (N,2)

    # 2) Estimate typical spacing from k-NN
    kdt = KDTree(XY)
    dists, _ = kdt.query(XY, k=k_nn+1)     # includes self at index 0
    spacing = np.median(dists[:, -1])
    target_r = 0.9 * keep_ratio * spacing  # Poisson-disk radius

    # 3) Rasterize occupancy and find interior “hole” cells
    pad = 3 * spacing
    minxy = XY.min(axis=0) - pad
    maxxy = XY.max(axis=0) + pad
    cell = 0.6 * spacing                   # grid resolution
    H = int(np.ceil((maxxy[1]-minxy[1]) / cell))
    W = int(np.ceil((maxxy[0]-minxy[0]) / cell))
    idx = np.floor((XY - minxy) / cell).astype(int)
    idx[:,0] = np.clip(idx[:,0], 0, W-1)
    idx[:,1] = np.clip(idx[:,1], 0, H-1)
    occ = np.zeros((H, W), dtype=bool)
    occ[idx[:,1], idx[:,0]] = True

    filled = binary_fill_holes(occ)
    hole_mask = filled & ~occ

    ys, xs = np.nonzero(hole_mask)
    if len(xs) == 0:
        return P, np.empty((0,3))
    centers = np.stack([xs + 0.5, ys + 0.5], axis=1) * cell + minxy  # (M,2)

    # 4) Blue-noise (dart-throwing) inside the hole at ~original density
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(centers))
    accepted = []
    tree = KDTree(XY)
    for i in order:
        p = centers[i]
        if tree.query(p, k=1)[0] < target_r:   # too close to existing points
            continue
        if accepted:
            A = np.vstack(accepted)
            if np.min(np.linalg.norm(A - p, axis=1)) < target_r:
                continue
        accepted.append(p)

    if not accepted:
        return P, np.empty((0,3))

    A = np.vstack(accepted)  # (M,2)

    # 5) Lift new samples back to 3D on the plane
    new_pts = p0 + A[:,0:1]*u + A[:,1:2]*v
    return torch.from_numpy(new_pts).to(torch.float32)

@torch.no_grad()
def assign_attrs_for_new_gaussians(
    og_attrs,
    new_pts,                      # (M,3) float tensor, same device/dtype as model.means
):
    """
    Returns a dict with tensors for new gaussians:
      features_dc, features_rest, opacities, quats, scales
    """
    dev=og_attrs["means"].device
    N = new_pts.shape[0]
    opacities = og_attrs["opacities"].to(dev).mean()
    scales  = og_attrs["scales"].to(dev).mean(dim=0)
    quats  = og_attrs["quats"].to(dev).mean(dim=0)
    features_rest = og_attrs["features_rest"].to(dev).mean(dim=0)
    features_dc = og_attrs["features_dc"].to(dev).mean(dim=0)

    return {
        "means": new_pts.to(dev),
        "opacities": opacities.unsqueeze(0).expand(N, -1),
        "scales": scales.unsqueeze(0).expand(N, -1),
        "quats": quats.unsqueeze(0).expand(N, -1),
        "features_rest": features_rest.unsqueeze(0).expand(N, -1, -1),
        "features_dc": features_dc.unsqueeze(0).expand(N, -1)
    }

@torch.no_grad()
def mask_by_plane_alignment(
    n,                      # (3,) plane normal (need not be unit)
    gaussians,             
    tau_deg=20.0,           # threshold in degrees
    align='normal',         # 'normal' or 'tangent'
    scales_log=True,        # exp() to get axis lengths if using 3DGS-style log-scales
    isotropy_eps=1e-3       # if axes are ~equal, orientation is ambiguous
):
    """
    Returns boolean keep_mask of shape (N,) where True means "keep this Gaussian".
    Strategy: pick the axis with the smallest scale, compare to plane normal.
    """
    quats  = gaussians["quats"]     # (N,4)
    scales = gaussians["scales"]    # (N,3) - log-scales in 3DGS; set scales_log=True if so

    device = quats.device
    n = n.to(device, dtype=quats.dtype)
    n = n / (torch.linalg.norm(n) + 1e-12)

    # Effective axis lengths
    axis_len = torch.exp(scales) if scales_log else scales
    # Smallest axis index (N,)
    idx_min = torch.argmin(axis_len, dim=1)  # the axis we expect to align with the plane normal

    # Convert quats to rotation matrices
    R = quat_to_mat(quats)  # (N,3,3)

    # Gather the world-space direction of that axis: columns of R are rotated basis axes
    # Build an index tensor to pick column idx_min for each row
    N = quats.shape[0]
    col_idx = idx_min.view(N, 1, 1).expand(N, 3, 1)
    axis_world = torch.gather(R, dim=2, index=col_idx).squeeze(2)  # (N,3)
    axis_world = axis_world / (torch.linalg.norm(axis_world, dim=1, keepdim=True) + 1e-12)

    # Angle to plane normal
    cosang = torch.sum(axis_world * n.unsqueeze(0), dim=1).abs()     # |cos(theta)|
    cosang = torch.clamp(cosang, -1.0, 1.0)
    theta = torch.arccos(cosang)                                     # radians
    tau = torch.tensor(tau_deg, device=device, dtype=theta.dtype) * (torch.pi/180.0)

    # Handle near-isotropic Gaussians where orientation isn't meaningful
    span = axis_len.max(dim=1).values - axis_len.min(dim=1).values   # (N,)
    not_isotropic = span > isotropy_eps

    if align == 'normal':
        # aligned if theta <= tau
        ok = theta <= tau
    elif align == 'tangent':
        # tangent means axis ⟂ normal => theta ~ 90°. Keep if |90° - theta| <= tau
        ok = (torch.abs((torch.pi/2) - theta) <= tau)
    else:
        raise ValueError("align must be 'normal' or 'tangent'")

    # Only enforce orientation when it's meaningful; otherwise keep
    keep_mask = torch.where(not_isotropic, ok, torch.ones_like(ok, dtype=torch.bool))
    return keep_mask



