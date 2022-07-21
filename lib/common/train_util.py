from skimage import measure
from ..data.mesh_util import *
from ..model.geometry import *
from ..common.sdf import create_grid, eval_grid_octree, eval_grid


def reshape_multiview_tensors(tensor):
    if isinstance(tensor, list):
        return [t.view(t.shape[0] * t.shape[1], *t.shape[2:]) for t in tensor]
    elif torch.is_tensor(tensor):
        return tensor.view(tensor.shape[0] * tensor.shape[1], *tensor.shape[2:]).contiguous()
    else:
        raise TypeError('tensor must be a list or torch.Tensor')


def reshape_sample_tensor(sample_tensor, num_views):
    if num_views == 1 or sample_tensor is None:
        return sample_tensor
    if torch.is_tensor(sample_tensor):
        # Need to repeat sample_tensor along the batch dim num_views times
        sample_tensor = sample_tensor.unsqueeze(dim=1).repeat(1, num_views, 1, 1).view(
            sample_tensor.shape[0] * num_views, *sample_tensor.shape[1:]
        ).contiguous()
        return sample_tensor
    elif isinstance(sample_tensor, list):
        return [reshape_sample_tensor(t, num_views) for t in sample_tensor]
    else:
        raise TypeError('tensor must be a list or torch.Tensor')


def warp_and_project_points(points, skin_weights, joint_transform, calib=None):
    """
    Warp a canonical point cloud to multiple posed spaces and project to image space
    Args:
        points: [N, 3] Tensor of 3D points
        skin_weights: [N, J]  corresponding skinning weights of points
        joint_transform: [B, J, 4, 4] joint transform matrix of a batch of poses
        calib: [B, 24, 4, 4] calibration matrix
    Returns:
        posed_points [B, N, 3] warpped points in posed space
        xyz: [B, N, 3] projected points in image space
    """
    if not torch.is_tensor(points):
        points = torch.as_tensor(points).float()
    if not torch.is_tensor(joint_transform):
        joint_transform = torch.as_tensor(joint_transform).float()
    if not torch.is_tensor(skin_weights):
        skin_weights = torch.as_tensor(skin_weights).float()

    # warping
    batch = joint_transform.shape[0]
    points_posed = linear_blend_skinning(points.expand(batch, -1, -1),
                                         skin_weights.expand(batch, -1, -1),
                                         joint_transform)
    if calib is None:
        return points_posed
    # projection
    xyz = orthogonal(points_posed.transpose(1, 2), calib).transpose(1, 2)

    return points_posed, xyz


def reconstruction(net, cuda, calib_tensor,
                   joint_transform,
                   smpl_lbs_weights,
                   smpl_dict,
                   resolution,
                   b_min, b_max,
                   use_octree=False, num_samples=100000, thresh=0.5):
    '''
    Reconstruct meshes from sdf predicted by the network.
    :param net: a BasePixImpNet object. call image filter beforehead.
    :param cuda: cuda device
    :param calib_tensor: calibration tensor
    :param resolution: resolution of the grid cell
    :param b_min: bounding box corner [x_min, y_min, z_min]
    :param b_max: bounding box corner [x_max, y_max, z_max]
    :param joint_transform
    :param use_octree: whether to use octree acceleration
    :param num_samples: how many points to query each gpu iteration
    :param smpl_lbs_weights [N, 24]
    :thresh
    :return: marching cubes results.
    :thresh:
    Args:
    '''

    smpl_v = smpl_dict['canon_smpl_vert'][0]
    b_max = smpl_v.cpu().numpy().max(0) + 0.2
    b_min = smpl_v.cpu().numpy().min(0) - 0.2

    # First we create a grid by resolution
    # and transforming matrix for grid coordinates to real world xyz
    coords, mat = create_grid(resolution, resolution, resolution, b_min, b_max)
    num_views = net.num_views

    # Then we define the lambda function for cell evaluation
    def eval_func(points):
        with torch.no_grad():
            canon_points = torch.from_numpy(points).t().to(device=cuda).float()
            nn_lbs_weights = query_lbs_weight(canon_points, smpl_v, smpl_lbs_weights)
            posed_points, projected_points = warp_and_project_points(canon_points,
                                                                     skin_weights=nn_lbs_weights,
                                                                     joint_transform=joint_transform,
                                                                     calib=calib_tensor)

            canon_points = canon_points.t().unsqueeze(0).expand(num_views, -1, -1)
            posed_points = posed_points.transpose(1, 2)
            projected_points = projected_points.transpose(1, 2)

            geo_feat = net.get_geo_feat(
                space='canon',
                canon_points=canon_points,
                posed_points=posed_points,
                projected_points=projected_points, **smpl_dict)

            net.query(projected_points, geo_feat)

            pred = net.get_preds()

            if not net.sdf:
                pred = 1 - pred
            return pred[0][0].detach().cpu().numpy()

    # Then we evaluate the grid
    if use_octree:
        sdf = eval_grid_octree(coords, eval_func, num_samples=num_samples)
    else:
        sdf = eval_grid(coords, eval_func, num_samples=num_samples)

    # Finally we do marching cubes
    try:
        verts, faces, normals, values = measure.marching_cubes_lewiner(sdf, thresh)
        verts = np.matmul(mat[:3, :3], verts.T) + mat[:3, 3:4]
        verts = verts.T
        normals = normals * 0.5 + 0.5
        return verts, faces, normals, values

    except Exception as e:
        print(e)
        print('error cannot marching cubes')
        # return -1
        exit()


def gen_mesh(opt, netG, cuda, data, save_path, use_octree=True):
    image_tensor = data['image'].to(device=cuda)
    calib_tensor = data['calib'].to(device=cuda)
    joint_transform = data['joint_transform'].to(device=cuda)
    smpl_lbs_weights = data['smpl_lbs_weights'].to(device=cuda)
    b_min = data['b_min']
    b_max = data['b_max']

    smpl_dicts = {k: v.to(cuda) if 'posed' in k else v.unsqueeze(0).expand(netG.num_views, -1, -1).to(cuda)
                  for k, v in data.items() if 'smpl' in k and 'lbs' not in k}

    netG.filter(image_tensor)

    verts, faces, normals, values = reconstruction(
        netG, cuda, calib_tensor,
        joint_transform,
        smpl_lbs_weights,
        smpl_dicts,
        opt.mcube_res, b_min, b_max,
        use_octree=use_octree,
        thresh=opt.thresh
    )

    mesh = trimesh.Trimesh(verts, faces, normals, vertex_colors=values)

    if opt.clean_mesh:
        mesh = mesh_clean(mesh)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    mesh.export(save_path)

    return mesh


def adjust_learning_rate(optimizer, epoch, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= gamma


def move_dict_to_device(dict, device, to_batch=False):
    for k, v in dict.items():
        if torch.is_tensor(v):
            if to_batch:
                dict[k] = v.unsqueeze(0).float().to(device)
            else:
                dict[k] = v.float().to(device)


def move_dict_to_batch(dict):
    for k, v in dict.items():
        if torch.is_tensor(v):
            dict[k] = v.unsqueeze(0).float()


def concat_dict_tensor(dict, data):
    if dict == {}:
        return data
    if data == {}:
        return dict
    return {k: torch.cat([dict[k], data[k]]) for k in dict.keys()}


def convert_dict_to_str(data_dict):
    string = ''
    for k, v in data_dict.items():
        string += '| %s:%.4f ' % (k, v)
    return string


def save_samples_truncted_prob(fname, points, prob):
    '''
    Save the visualization of sampling to a ply file.
    Red points represent positive predictions.
    Green points represent negative predictions.
    :param fname: File name to save
    :param points: [N, 3] array of points
    :param prob: [N, 1] array of predictions in the range [0~1]
    :return:
    '''
    r = (prob > 0.5).reshape([-1, 1]) * 255
    g = (prob < 0.5).reshape([-1, 1]) * 255
    b = np.zeros(r.shape)
    to_save = np.concatenate([points, r, g, b], axis=-1)
    return np.savetxt(fname,
                      to_save,
                      fmt='%.6f %.6f %.6f %d %d %d',
                      comments='',
                      header=(
                          'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
                          points.shape[0])
                      )


def save_samples_rgb(fname, points, rgb):
    '''
    Save the visualization of sampling to a ply file.
    Red points represent positive predictions.
    Green points represent negative predictions.
    :param fname: File name to save
    :param points: [N, 3] array of points
    :param rgb: [N, 3] array of rgb values in the range [0~1]
    :return:
    '''
    to_save = np.concatenate([points, rgb * 255], axis=-1)
    return np.savetxt(fname,
                      to_save,
                      fmt='%.6f %.6f %.6f %d %d %d',
                      comments='',
                      header=(
                          'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
                          points.shape[0])
                      )


def scatter_points_to_image(points, image):
    """
    Args:
        points: [3, N]
        image: [3, H, W]
    Returns:
        image [image_size, resolution, 3]
    """
    im_size = image.shape[-1]
    image_np = image.permute(1, 2, 0).numpy() * 0.5 + 0.5
    xys = ((points[:2].T + 1.0) * 0.5 * im_size)

    for xy in xys:
        if 0 <= int(xy[0]) < im_size and 0 <= int(xy[1]) < im_size:
            image_np[int(xy[1]), int(xy[0]), :] = (1, 1, 1)
    return image * 255



