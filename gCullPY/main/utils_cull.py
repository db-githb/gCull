import torch
import numpy as np
from PIL import Image
from pathlib import Path
from gCullCUDA import GaussianCullSettings, GaussianCuller
from gCullPY.main.utils_main import build_loader
from gCullUTILS.rich_utils import get_progress

# taken from gaussian-opacity-fields
def compute_3D_filter(model, camera, s):
    #TODO consider focal length and image width
    xyz = model.means
    distance = torch.ones((xyz.shape[0]), device=xyz.device) * 100000.0
    #valid_points = torch.zeros((xyz.shape[0]), device=xyz.device, dtype=torch.bool)
    
    # we should use the focal length of the highest resolution camera
    focal_length = camera.fx.item()

    # transform points to camera space - GOF grabs rotation and translation from input file which is just colmap data
    camera_to_world = camera.camera_to_worlds

    device = camera_to_world.device.type
    R = camera_to_world[:3, :3].T # 3 x 3
    T = camera_to_world[:3, 3:4] * -1 # 3 x 1
    # R is stored transposed due to 'glm' in CUDA code so we don't neet transopse here - NOTE: ??? this comment is from GOF
    xyz_cam = xyz @ R + T.T # I don't know why GOF uses [None, :] but this messes up the matrix/vector dimensions
    
    # project to screen space - renamed valid_depth to valid
    valid_depth = xyz_cam[:, 2] > 0.2 # TODO remove hard coded value - depth is camera center to gaussian mean
    
    x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
    z = torch.clamp(z, min=0.001)
    
    x = x / z * camera.fx.item() + camera.image_width.item() / 2.0
    y = y / z * camera.fy.item() + camera.image_height.item() / 2.0
    
    in_screen = torch.logical_and(torch.logical_and(x >= -0.15 * camera.image_width.item(), x <= camera.image_width.item() * 1.15), torch.logical_and(y >= -0.15 * camera.image_height.item(), y <= 1.15 * camera.image_height.item()))

    in_screen = z > 0.0 # filter out all Gaussians behind camera for efficiency
    valid = torch.logical_and(valid_depth, in_screen)
    if torch.all(valid == False):
        return torch.zeros(xyz.shape, device="cuda")
    
    distance[valid] = torch.min(distance[valid], z[valid])
    
    distance[~valid] = distance[valid].max()
    
    #TODO remove hard coded value
    #TODO box to gaussian transform
    filter_3D = distance / focal_length * (s ** 0.5)
    filter_3D = filter_3D.unsqueeze(1) # mine - needs second dimension to allow for operations with opacity and scales
    return filter_3D

# taken from gaussian-opacity-fields
def get_opacity_with_3D_filter(model, filter_3D):
    opacities = model.opacities
    opacity_activation = torch.sigmoid(opacities)
    # apply 3D filter
    scale_activation = get_scaling(model)
    
    scales_square = torch.square(scale_activation)
    det1 = scales_square.prod(dim=1)
    
    scales_after_square = scales_square + torch.square(filter_3D) 
    det2 = scales_after_square.prod(dim=1) 
    coef = torch.sqrt(det1 / det2)
    return opacity_activation * coef[..., None]

# opacity with activation 
def get_opacity(model, filter_3D = None):

    if filter_3D is not None:
         opacity = get_opacity_with_3D_filter(model, filter_3D) 
    else:
        opacity = torch.sigmoid(model.opacities) # activation without 3D filter
    
    return opacity

# scales with activation etc. but no 3D filter
def get_scaling(model, filter_3D = None):
        
    scales = torch.exp(model.scales)

    if filter_3D is not None:
        # taken from gaussian-opacity-fields
        scales = torch.square(scales) + torch.square(filter_3D)
        scales = torch.sqrt(scales)

    return scales

def get_rot_with_act_func(model):
    return torch.nn.functional.normalize(model.quats)

def get_tanHalfFov(camera):
     # calculate the FOV of the camera given fx and fy, width and height
    px = camera.image_width.item() # pixel width
    py = camera.image_height.item() # pixel height
    fx = camera.fx.item() # focal width
    fy = camera.fy.item() # focal height
    tanHalfFovX = 0.5*px/fx # # comma makes output of equation a tuple that must be indexed
    tanHalfFovY = 0.5*py/fy # comma makes output of equation a tuple that must be indexed

    return tanHalfFovX, tanHalfFovY

def get_Rt_inv(model, camera):
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
    
def get_viewmat(model, camera):
    R_inv, T_inv = get_Rt_inv(model, camera)
    viewmat = torch.eye(4, device=R_inv.device, dtype=R_inv.dtype) # viewmat = world to camera -> https://docs.gsplat.studio/main/conventions/data_conventions.html
    viewmat[:3, :3] = R_inv
    viewmat[:3, 3:4] = T_inv
    viewmat = viewmat.T # transpose for GOF code

    #not sure if I need to do this
    # viewmat[:, 1:3] *= -1.0 # switch between openCV and openGL conventions?

    return viewmat

# taken from gaussian-opacity-fields
def get_full_proj_transform(tanHalfFovX, tanHalfFovY, viewMat):
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
    return (viewMat.unsqueeze(0).bmm(projMat.transpose(0,1).unsqueeze(0))).squeeze(0)

def get_cull_list(model, camera, binary_mask):
    background = torch.ones(3, device=model.device) # get_background(model)
    _ , T_inv = get_Rt_inv(model, camera)
    viewmat = get_viewmat(model, camera)
    tanHalfFovX, tanHalfFovY = get_tanHalfFov(camera)
    
    # use this for post-rendering image filters
    height =int(camera.height.item())
    width = int(camera.width.item())

    cull_settings = GaussianCullSettings(
        image_height= height,
        image_width= width,
        tanfovx=  tanHalfFovX, # actually using the half angle i.e. tan half fov
        tanfovy= tanHalfFovY,
        kernel_size= 0.0, #  low-pass filter ensures every Gaussian should be at least one pixel wide/high. Discard 3rd row and column. (3DGS = 0.3f, gauss-op-field = 0.0)
        subpixel_offset=torch.zeros((int(camera.height.item()), int(camera.width.item()), 2), device="cuda"), #torch.zeros((int(py), int(px), 2), dtype=torch.float32, device="cuda"), # from gauss-opac-field
        bg=background,
        scale_modifier=1.0, # used to tweak covariance matrix leave as 1.0 (from gauss-op-field)
        viewmatrix=viewmat, # world2cam from nerfstudio
        projmatrix= get_full_proj_transform(tanHalfFovX, tanHalfFovY, viewmat), #from gauss-op-field
        sh_degree= 0, # GOF has this set to 3 (I think this is 3DGS default)
        campos= T_inv, # from nerfstudio 
        prefiltered=False, # keep prefiltered as false (from gauss-op-field)
        debug=True)
    
    gCuller = GaussianCuller(cull_settings=cull_settings)

    means3D =  model.means

    filter_3D = compute_3D_filter(model, camera,.01) #model.filter3D_scale, renderer) # compute_3D_filter per camera (GOF does it for each camera at the beginning of training)
    opacity = get_opacity(model, filter_3D) 
    
    scales =   get_scaling(model, filter_3D) # self.get_scaling_with_3D_filter(self) # mip-splatting 3D filter from GOF 
    rotation = get_rot_with_act_func(model) # self.quats 

    cull_lst = gCuller(
            binary_mask = binary_mask,
            means3D = means3D,
            shs = torch.cat((model.features_dc.unsqueeze(1), model.features_rest), dim=1),
            colors_precomp = None,
            opacities = opacity,
            scales = scales,
            rotations = rotation, #self.quats,
            cov3D_precomp = None,
            view2gaussian_precomp=None)
    
    return cull_lst

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

def cull_loop(config, pipeline):

    mask_dir = config.datamanager.data / "masks" #get_mask_dir(config)
    cull_lst_master = torch.zeros(pipeline.model.means.shape[0], dtype=torch.int)

    for split in "train+test".split("+"):

        dataset, dataloader = build_loader(config, split, pipeline.device)
        desc = f"\u2702\ufe0f\u00A0 Culling split {split} \u2702\ufe0f\u00A0"
        if split == "test":
            break

        with get_progress(desc) as progress:
            for camera_idx, (camera, batch) in enumerate(progress.track(dataloader, total=len(dataset))):
                with torch.no_grad():
                    #frame_idx = batch["image_idx"]
                    camera.camera_to_worlds = camera.camera_to_worlds.squeeze() # splatoff rasterizer requires cam2world.shape = [3,4]
                    binary_mask = get_mask(batch, mask_dir).int()
                    #bool_mask = get_mask(batch, config.datamanager.data / "masks_4")
                    #bool_mask.data = bool_mask_1.data
                    cull_lst = get_cull_list(pipeline.model, camera, binary_mask)
                    cull_lst_master |= cull_lst.to("cpu")
                    print(f"{camera_idx}: {cull_lst.sum().item()}/{cull_lst_master.sum().item()}")
                    if camera_idx == 100:
                        break

    return cull_lst_master

