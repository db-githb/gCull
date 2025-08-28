import torch
from gCullPY.main.utils_cull import normalize, mat_to_quat

def append_gaussians_to_model(og_model, new_gaussians):
    with torch.no_grad():
        og_model.means.data = torch.cat((og_model.means.data, new_gaussians["means"]))
        og_model.opacities.data = torch.cat((og_model.opacities.data, new_gaussians["opacities"]))
        og_model.scales.data = torch.cat((og_model.scales.data, new_gaussians["scales"]))
        og_model.quats.data = torch.cat((og_model.quats.data, new_gaussians["quats"]))
        og_model.features_dc.data = torch.cat((og_model.features_dc.data, new_gaussians["features_dc"]))
        og_model.features_rest.data = torch.cat((og_model.features_rest.data, new_gaussians["features_rest"]))
    return og_model

def ground_driver(norm, ground_gaussians, pipeline):
    p0 = p0_from_inliers_centroid(ground_gaussians["means"], norm)
    ground_tile = fill_plane_with_gaussians(norm, p0, 100, 100, .1, jitter=.1) #only has means, quats, scales
    
    N = ground_tile["means"].shape[0]
    dev = ground_tile["means"].device
    dt = ground_tile["quats"].dtype
    opacity = ground_gaussians["opacities"].median()
    ground_tile["opacities"] = torch.full((ground_tile["means"].shape[0],1), opacity.item(), device=dev, dtype=dt)
    ground_tile["features_dc"], ground_tile["features_rest"] = sample_sh_by_index(ground_gaussians["features_dc"], ground_gaussians["features_rest"], N)
    pipeline.model = append_gaussians_to_model(pipeline.model, ground_tile)
    return pipeline.model

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

def get_all_ground_gaussians(og_ground, new_ground):
    with torch.no_grad():
        og_ground["means"] = torch.cat((og_ground["means"], new_ground["means"]))
        og_ground["opacities"] = torch.cat((og_ground["opacities"], new_ground["opacities"]))
        og_ground["scales"] = torch.cat((og_ground["scales"], new_ground["scales"])) 
        og_ground["quats"] = torch.cat((og_ground["quats"], new_ground["quats"]))
        og_ground["features_dc"] = torch.cat((og_ground["features_dc"], new_ground["features_dc"]))
        og_ground["features_rest"] = torch.cat((og_ground["features_rest"], new_ground["features_rest"]))
    return og_ground

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
def fill_plane_with_gaussians(
    n,                         # (3,) plane normal
    p0,                        # (3,) a point on the plane (plane origin)
    nx, ny,                    # grid size (cols, rows)
    pitch,                     # spacing between neighbors (world units)
    log_scales=True,           # True for 3DGS-style log-scales
    scale_tan=(0.05, 0.05),    # in-plane radii (x/z on the plane)
    scale_norm=0.01,           # thickness (along the normal) -> keep small so splats lie flat
    quat_normal_axis='z',      # which local axis is aligned to plane normal: 'x'|'y'|'z'
    roll=0.0,                  # in-plane rotation (radians)
    jitter=0.0,                # 0..1 random jitter fraction of pitch
    device='cuda',
    dtype=torch.float32
):
    n = torch.as_tensor(n, device=device, dtype=dtype)
    p0 = torch.as_tensor(p0, device=device, dtype=dtype)

    # 1) tangent basis (u,v) and normal n̂
    u, v, _ = basis_from_normal(n, roll=roll)
    n_hat = normalize(n)

    # 2) grid in (u,v) centered on p0
    xs = torch.linspace(-(nx-1)/2, (nx-1)/2, steps=nx, device=device, dtype=dtype) * pitch
    ys = torch.linspace(-(ny-1)/2, (ny-1)/2, steps=ny, device=device, dtype=dtype) * pitch
    X, Y = torch.meshgrid(xs, ys, indexing='xy')                # (nx,ny)
    if jitter > 0:
        X = X + (torch.rand_like(X)-0.5) * (jitter * pitch)
        Y = Y + (torch.rand_like(Y)-0.5) * (jitter * pitch)
    # means on plane: p = p0 + x*u + y*v
    means = (p0[None,None,:]
             + X[...,None]*u[None,None,:]
             + Y[...,None]*v[None,None,:]).reshape(-1, 3)       # (N,3), N=nx*ny

    # 3) rotation: choose which local axis maps to n̂
    if   quat_normal_axis == 'z':
        R = torch.stack([u, v, n_hat], dim=-1)                  # (3,3)
    elif quat_normal_axis == 'y':
        R = torch.stack([u, n_hat, v], dim=-1)
    elif quat_normal_axis == 'x':
        R = torch.stack([n_hat, u, v], dim=-1)
    else:
        raise ValueError("quat_normal_axis must be 'x','y','z'")
    
    R = R.unsqueeze(0).expand(means.shape[0], -1, -1).contiguous()
    quats = mat_to_quat(R)                                 # (N,4)

    # 4) scales: two in-plane radii (larger), one small along normal axis
    sx, sy = scale_tan
    sn = scale_norm
    if   quat_normal_axis == 'z': base = [sx, sy, sn]
    elif quat_normal_axis == 'y': base = [sx, sn, sy]
    else:                         base = [sn, sx, sy]           # 'x'
    base = torch.tensor(base, device=device, dtype=dtype)
    if log_scales:
        base = torch.log(base)
    scales = base.unsqueeze(0).expand(means.shape[0], -1).contiguous()

    return {
        "means":     means,         # (N,3)
        "quats":     quats,         # (N,4)
        "scales":    scales,        # (N,3) (log if log_scales=True)
        # Fill opacities/features as you like, e.g.:
        # "opacities": torch.full((means.shape[0],1), 0.8, device=device, dtype=dtype)
    }

def p0_from_inliers_centroid(P_inliers, n):
    """
    P_inliers: (M,3) inlier points from RANSAC
    n: (3,) plane normal (need not be unit)
    Returns p0 on the plane near the middle of the inliers.
    Plane is assumed to be n·x = c (c estimated from inliers).
    """
    n_hat = normalize(n)
    c = torch.median(P_inliers @ n_hat)           # robust plane offset along n (use mean if you prefer)
    centroid = P_inliers.mean(dim=0)
    # project centroid onto plane: subtract normal component difference
    p0 = centroid - ((centroid @ n_hat) - c) * n_hat
    return p0


def eligible(mask_mode, clip, z):
    if mask_mode == "all":
        return (z.abs() <= clip).all(dim=1)                # [N]
    elif mask_mode == "any":
        return (z.abs() <= clip).any(dim=1)
    elif mask_mode == "mahal":
        return (z.pow(2).sum(dim=1).sqrt() <= clip)
    else:
        raise ValueError("mode must be 'all' | 'any' | 'mahal'")

@torch.no_grad()
def sample_sh_by_index(
    template_dc,            # [N,3]
    template_rest,          # [N,15,3]
    M,                      # number of new Gaussians (int or scalar tensor)
    sigma_clip=1.0,         # “within ±1σ”
    mode="mahal",           # "all" | "any" | "mahal"
    with_replacement=True,  # allow drawing the same index multiple times
    relax_if_needed=True,   # widen the band if none eligible
    generator=None          # torch.Generator for reproducibility
):
    assert template_dc.ndim == 2 and template_dc.shape[1] == 3
    assert template_rest.shape[0] == template_dc.shape[0] and template_rest.shape[1:] == (15,3)
    dev, dt = template_dc.device, template_dc.dtype
    N = template_dc.shape[0]
    M = int(torch.as_tensor(M).item())  # ensure plain int

    # Per-coefficient stats across Gaussians
    mu = template_dc.mean(dim=0)                                # [3]
    sd = template_dc.std(dim=0, unbiased=False) + 1e-8          # [3]
    z  = (template_dc - mu) / sd                                # [N,3]
    
    clip = float(sigma_clip)
    mask = eligible(mode, clip, z)

    # If none eligible, optionally relax
    if relax_if_needed and mask.sum().item() == 0:
        for grow in (1.25, 1.5, 2.0, 3.0):
            mask = eligible(mode, clip * grow, z)
            if mask.sum().item() > 0:
                break

    idx_pool = mask.nonzero(as_tuple=False).squeeze(1)          # [K]
    if idx_pool.numel() == 0:
        # final fallback: use all N (we’ll sample from them)
        idx_pool = torch.arange(N, device=dev)

    # Sample M indices from the pool
    if with_replacement:
        idx = idx_pool[torch.randint(idx_pool.numel(), (M,), device=dev, generator=generator)]
    else:
        K = idx_pool.numel()
        if K >= M:
            perm = torch.randperm(K, device=dev, generator=generator)
            idx = idx_pool[perm[:M]]
        else:
            # repeat/shuffle to reach M unique-ish picks
            reps = (M + K - 1) // K
            idx = idx_pool.repeat(reps)[:M]
            shuf = torch.randperm(M, device=dev, generator=generator)
            idx = idx[shuf]

    # Gather paired SH coefficients
    new_dc   = template_dc.index_select(0, idx).contiguous()    # [M,3]
    new_rest = template_rest.index_select(0, idx).contiguous()  # [M,15,3]
    return new_dc, new_rest

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
    return u, v, R