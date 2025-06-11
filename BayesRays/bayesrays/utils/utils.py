import torch
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.math import Gaussians, conical_frustum_to_gaussian

from nerfstudio.model_components import renderers
from ray_splat_renderer import GaussianRasterizationSettings, GaussianRasterizer

from scipy import ndimage as img
num_iPts = 1

def sort_package(raster_pkg, camera):
    cols, rows = camera.width.item(), camera.height.item()
    num_pixels = cols*rows

       #   xw      xg         T-test_t       alpha*T      gColor       gIndex        gradients
    myCh = [9, 9+3*num_iPts, 9+6*num_iPts, 9+7*num_iPts, 9+8*num_iPts, 9+11*num_iPts, 9+12*num_iPts] # My Channels: xw, ow, rw, colors, contrib, gIdx
    myCh = [x * num_pixels for x in myCh]

      #            rgb       normal       depth       alpha         uncertainty (formerly distortion)
    origCh = [3*num_pixels,6*num_pixels,7*num_pixels,8*num_pixels,9*num_pixels]
        
    outputs = {}
    outputs["rgb"] = raster_pkg[:origCh[0]].view(3,rows,cols).permute(1,2,0).clamp(max=1) # clamp to prevent color overflow - also causes hard ring with single gaussian
    outputs["depth"] = raster_pkg[origCh[1]:origCh[2]].view(rows,cols,1)
    outputs["alpha"] = raster_pkg[origCh[2]:origCh[3]].view(rows,cols,1) # contr_count
    outputs["xw"] = raster_pkg[myCh[0]:myCh[1]].view(num_iPts,rows*cols, 3) # view(num_iPts,rows*cols, 3)
    outputs["xg"] = raster_pkg[myCh[1]:myCh[2]].view(num_iPts,rows*cols, 3)
    outputs["cDiff"] = raster_pkg[myCh[2]:myCh[3]].view(num_iPts, rows, cols).permute(1,2,0)
    outputs["cProd"] = raster_pkg[myCh[3]:myCh[4]].view(num_iPts, rows, cols).permute(1,2,0)
    outputs["gColor"] = raster_pkg[myCh[4]:myCh[5]].view(num_iPts, rows,cols, 3)
    outputs["gIndex"] = raster_pkg[myCh[5]:myCh[6]].view(num_iPts, rows, cols).permute(1,2,0).int()
    outputs["gradients"] = raster_pkg[myCh[6]:myCh[6]+cols*rows*num_iPts*3*3].reshape(num_iPts, cols*rows, 3, 3)
    outputs["uncertainty"] = raster_pkg[origCh[3]:origCh[4]].view(rows,cols,1)
    
    write = False

    if write:
        with open("rgb.txt", "w") as file:
            for row in outputs["rgb"].reshape(rows,cols,3):
                for col in row:
                    for el in col:
                        file.write(f"{el.item():7.4f} ")
                    file.write(", ")
                file.write("\n")

    return outputs

def normalize_points(points):
    device = points.device
    max = torch.tensor((points[:,:,0].max(), points[:,:,1].max(), points[:,:,2].max()), device=device)
    min = torch.tensor((points[:,:,0].min(), points[:,:,1].min(), points[:,:,2].min()), device=device)
    delta = max - min
    points =(points - min)/(delta+1e-7)
    #points = (points*2)-1 # aabb range is from -1 to 1 so set points in same range
    selector = ((points >= 0.0) & (points <= 1.0)).all(dim=-1) #points outside aabb are filtered out
    return points, selector

def normalize_point_coords(points, aabb, distortion):
    ''' coordinate normalization process according to density_feild.py in nerfstudio'''
    if distortion is not None:
        pos = distortion(points)
        pos = (pos + 2.0) / 4.0
    else:        
        pos = SceneBox.get_normalized_positions(points, aabb)
    selector = ((pos > 0.0) & (pos < 1.0)).all(dim=-1) #points outside aabb are filtered out
    pos = pos * selector[..., None]
    return pos, selector

def find_grid_indices(points, aabb, distortion, lod, device, zero_out=True):
    
    # assume max value is the max bound of world space and min value=min bound.
    if distortion is None:
        pos, selector = normalize_points(points)
    else: # OG Bayes Rayes for nerfacto uses distortion
        pos, selector = normalize_point_coords(points.to(device), aabb, distortion)

    pos, selector = pos.reshape(-1, 3), selector[..., None].view(-1, 1)
    uncertainty_lod = 2 ** lod
    coords = (pos * uncertainty_lod).unsqueeze(0)
    inds = torch.zeros((8, pos.shape[0]), dtype=torch.int32, device=device)
    coefs = torch.zeros((8, pos.shape[0]), device=device)
    corners = torch.tensor(
        [[0, 0, 0, 0], [1, 0, 0, 1], [2, 0, 1, 0], [3, 0, 1, 1], [4, 1, 0, 0], [5, 1, 0, 1], [6, 1, 1, 0],
         [7, 1, 1, 1]], device=device)
    corners = corners.unsqueeze(1)
    inds[corners[:, :, 0].squeeze(1)] = (
            (torch.floor(coords[..., 0]) + corners[:, :, 1]) * uncertainty_lod * uncertainty_lod +
            (torch.floor(coords[..., 1]) + corners[:, :, 2]) * uncertainty_lod +
            (torch.floor(coords[..., 2]) + corners[:, :, 3])).int()
    coefs[corners[:, :, 0].squeeze(1)] = torch.abs(
        coords[..., 0] - (torch.floor(coords[..., 0]) + (1 - corners[:, :, 1]))) * torch.abs(
        coords[..., 1] - (torch.floor(coords[..., 1]) + (1 - corners[:, :, 2]))) * torch.abs(
        coords[..., 2] - (torch.floor(coords[..., 2]) + (1 - corners[:, :, 3])))
    if zero_out:
        coefs[corners[:, :, 0].squeeze(1)] *= selector[..., 0].unsqueeze(0)  # zero out the contribution of points outside aabb box

    return inds, coefs

def get_gaussian_blob_new(self) -> Gaussians: #for mipnerf
    """Calculates guassian approximation of conical frustum.

    Returns:
        Conical frustums approximated by gaussian distribution.
    """
    # Cone radius is set such that the square pixel_area matches the cone area.
    cone_radius = torch.sqrt(self.pixel_area) / 1.7724538509055159  # r = sqrt(pixel_area / pi)
    
    return conical_frustum_to_gaussian(
        origins=self.origins + self.offsets, #deforms Gaussian mean
        directions=self.directions,
        starts=self.starts,
        ends=self.ends,
        radius=cone_radius,
    )

# taken from gaussian-opacity-fields
def compute_3D_filter(model, camera, s, renderer=False):
    #TODO consider focal length and image width
    xyz = model.means
    distance = torch.ones((xyz.shape[0]), device=xyz.device) * 100000.0
    #valid_points = torch.zeros((xyz.shape[0]), device=xyz.device, dtype=torch.bool)
    
    # we should use the focal length of the highest resolution camera
    focal_length = camera.fx.item()

    # transform points to camera space - GOF grabs rotation and translation from input file which is just colmap data
    if renderer:
        camera_to_world = camera.camera_to_worlds
    else:
        camera_to_world = model.camera_optimizer.apply_to_camera(camera)[0, ...]

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
    
    # in_screen = torch.logical_and(torch.logical_and(x >= 0, x < camera.image_width.item()), torch.logical_and(y >= 0, y < camera.image_height.item()))
    
    #BEGIN - COMMENTING OUT BECAUSE DOESN'T SEEM WORTH IT
    #use similar tangent space filtering as in the paper
    in_screen = torch.logical_and(torch.logical_and(x >= -0.15 * camera.image_width.item(), x <= camera.image_width.item() * 1.15), torch.logical_and(y >= -0.15 * camera.image_height.item(), y <= 1.15 * camera.image_height.item()))

    in_screen = z > 0.0 # filter out all Gaussians behind camera for efficiency
    valid = torch.logical_and(valid_depth, in_screen)
    if torch.all(valid == False):
        return torch.zeros(xyz.shape, device="cuda")
    #END
    
    # distance[valid] = torch.min(distance[valid], xyz_to_cam[valid])
    distance[valid] = torch.min(distance[valid], z[valid])
    #valid_points = torch.logical_or(valid_points, valid)
    
    #distance[~valid_points] = distance[valid_points].max()
    distance[~valid] = distance[valid].max()
    
    #TODO remove hard coded value
    #TODO box to gaussian transform
    filter_3D = distance / focal_length * (s ** 0.5)
    filter_3D = filter_3D.unsqueeze(1) # mine - needs second dimension to allow for operations with opacity and scales
    return filter_3D

def get_background(model):
    # get the background color
    if model.training:
        if model.config.background_color == "random":
            background = torch.rand(3, device=model.device)
        elif model.config.background_color == "white":
            background = torch.ones(3, device=model.device)
        elif model.config.background_color == "black":
            background = torch.zeros(3, device=model.device)
        else:
            background = model.background_color.to(model.device)
    else:
        if renderers.BACKGROUND_COLOR_OVERRIDE is not None:
            background = renderers.BACKGROUND_COLOR_OVERRIDE.to(model.device)
        else:
            background = model.background_color.to(model.device)

    return background

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

def get_Rt_inv(model, camera, renderer):
    if renderer:
        camera_to_world = camera.camera_to_worlds
    else:
        camera_to_world = model.camera_optimizer.apply_to_camera(camera)[0, ...]


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
    
def get_viewmat(model, camera, renderer):
    R_inv, T_inv = get_Rt_inv(model, camera, renderer)
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

def get_rasterizer_output(model, camera, renderer=False):
    background = torch.ones(3, device=model.device) # get_background(model) - use white background to compare with bayes rays
    _ , T_inv = get_Rt_inv(model, camera, renderer)
    viewmat = get_viewmat(model, camera, renderer)
    tanHalfFovX, tanHalfFovY = get_tanHalfFov(camera)
    
    # use this for post-rendering image filters
    scale_img = 1
    height =int(camera.height.item())*scale_img
    width = int(camera.width.item())*scale_img

    raster_settings = GaussianRasterizationSettings(
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
    
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D =  model.means

    filter_3D = compute_3D_filter(model, camera,.01, renderer) #model.filter3D_scale, renderer) # compute_3D_filter per camera (GOF does it for each camera at the beginning of training)

    if model.filter3D:
        opacity =  get_opacity(model, filter_3D) # self.get_opacity_with_3D_filter(self) # mip-splatting 3D filter from GOF self.opacities
    else:
        opacity =  get_opacity(model, None) # self.get_opacity_with_3D_filter(self) # mip-splatting 3D filter from GOF self.opacities

    
    scales =   get_scaling(model, filter_3D) # self.get_scaling_with_3D_filter(self) # mip-splatting 3D filter from GOF 
    rotation = get_rot_with_act_func(model) # self.quats 
    means2D = torch.zeros_like(opacity)
    gaussian_index = torch.arange(model.means.shape[0], dtype=torch.int32, device="cuda")

    raster_pkg, model.radii = rasterizer(
            means3D = means3D,
            means2D = means2D, # not used in gof (legacy param from 3DGS)
            shs = torch.cat((model.features_dc.unsqueeze(1), model.features_rest), dim=1),
            colors_precomp = None,
            opacities = opacity,
            scales = scales,
            rotations = rotation, #self.quats,
            gaussian_index = gaussian_index,
            cov3D_precomp = None,
            view2gaussian_precomp=None)
    
    return raster_pkg



