from skimage import measure
from scipy.ndimage import filters
from ..data.mesh_util import *
from lib.common.geometry import *
from ..common.sdf import create_grid, eval_grid_octree, eval_grid
from ..common.lbs_util import query_lbs_weight, warp_and_project_points


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

    # pr_smpl_v = orthogonal(smpl_dict["posed_smpl_vert"].transpose(1, 2), calib_tensor).transpose(1, 2)
    # b_max = np.clip(pr_smpl_v[0].cpu().numpy().max(0) + 0.05, -1., 1.)
    # b_min = np.clip(pr_smpl_v[0].cpu().numpy().min(0) - 0.05, -1., 1.)
    # b_max = np.array([1., 1., 1.])
    # b_min = np.array([-1., -1., -1.])
    # First we create a grid by resolution
    # and transforming matrix for grid coordinates to real world xyz
    coords, mat = create_grid(resolution, resolution, resolution, b_min, b_max)
    num_views = net.num_views

    # Then we define the lambda function for cell evaluation
    def eval_func(points):
        with torch.no_grad():
            canon_points = torch.from_numpy(points).t().to(device=cuda).float()
            nn_lbs_weights = query_lbs_weight(canon_points, smpl_v, smpl_lbs_weights)
            posed_points, projected_points, vT = warp_and_project_points(canon_points,
                                                                         skin_weights=nn_lbs_weights,
                                                                         joint_transform=joint_transform,
                                                                         calib=calib_tensor, return_vT=True)

            canon_points = canon_points.t().unsqueeze(0).expand(num_views, -1, -1)
            posed_points = posed_points.transpose(1, 2)
            projected_points = projected_points.transpose(1, 2)

            geo_feat = net.get_geo_feat(
                space='canon',
                canon_points=canon_points,
                posed_points=posed_points,
                projected_points=projected_points, **smpl_dict)

            net.query(projected_points, geo_feat, vT, calib_tensor)

            pred = net.get_preds()

            if not net.sdf:
                pred = 1 - pred
            return pred[0][0].detach().cpu().numpy()

    def eval_func2(points):
        with torch.no_grad():
            p_points = torch.from_numpy(points).t().to(device=cuda).float()
            p_points = p_points.t().unsqueeze(0).expand(num_views, -1, -1)
            pr_points = p_points * torch.as_tensor([1, -1, -1]).view(1, 3, 1).to(device=cuda).float()

            net.query(pr_points, pr_points, space=net.space_list[0])

            pred = net.get_preds(space=net.space_list[0])

            if not net.sdf:
                pred = 1 - pred
            return pred[0][0].detach().cpu().numpy()

    # Then we evaluate the grid
    if use_octree:
        sdf = eval_grid_octree(coords, eval_func, num_samples=num_samples)
    else:
        sdf = eval_grid(coords, eval_func, num_samples=num_samples)

    sdf = gaussian_blur(sdf, 0.5)

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
        return -1


def gen_mesh(opt, netG, cuda, data, save_path, use_octree=True):
    calib_tensor = data['calib'].to(device=cuda)
    joint_transform = data['joint_transform'].to(device=cuda)
    smpl_lbs_weights = data['smpl_lbs_weights'].to(device=cuda)
    b_min = data['b_min']
    b_max = data['b_max']

    smpl_dicts = {
        k: v.to(cuda) if 'posed' in k
        else v.unsqueeze(0).expand(netG.num_views, -1, -1).to(cuda)
        for k, v in data.items() if 'smpl' in k and 'lbs' not in k
    }

    netG.filter({k: v.to(cuda) for k, v in data.items() if k in ['rgb', 'normal', 'depth']})

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

    # if opt.clean_mesh:
    mesh = mesh_clean(mesh)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    mesh.export(save_path)

    return mesh


def pifu_gen_mesh(opt, netG, cuda, data, save_path, use_octree=True):
    smpl_dicts = {
        k: v.to(cuda) if 'canon' not in k and not k == 'smpl_faces'
        else v.unsqueeze(0).expand(netG.num_views, -1, -1).to(cuda)
        for k, v in data.items() if ('smpl' in k or k == 'sdf') and torch.is_tensor(v)
    }

    netG.filter({k: v.to(cuda) for k, v in data.items() if k in ['rgb', 'normal', 'depth']})

    verts, faces, normals, values = pifu_reconstruction(
        netG, cuda, smpl_dicts, opt.mcube_res, use_octree=use_octree, thresh=opt.thresh)

    mesh = trimesh.Trimesh(verts, faces, normals, vertex_colors=values)

    # if opt.clean_mesh:
    mesh = mesh_clean(mesh)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    mesh.export(save_path)

    return mesh


def pifu_reconstruction(net, cuda, smpl_dict, resolution,
                        b_min=np.array([-1., -1., -1.]) + 0.01, b_max=np.array([1., 1., 1.]),
                        use_octree=False, num_samples=100000, thresh=0.5):
    '''
    Reconstruct meshes from sdf predicted by the network.
    :param net: a BasePixImpNet object. call image filter beforehead.
    :param cuda: cuda device
    :param calib_tensor: calibration tensor
    :param resolution: resolution of the grid cell
    :param b_min: bounding box corner [x_min, y_min, z_min]
    :param b_max: bounding box corner [x_max, y_max, z_max]
    :param use_octree: whether to use octree acceleration
    :param num_samples: how many points to query each gpu iteration
    :thresh
    :return: marching cubes results.
    :thresh:
    Args:
    '''

    # First we create a grid by resolution
    # and transforming matrix for grid coordinates to real world xyz
    coords, mat = create_grid(resolution, resolution, resolution, b_min, b_max)

    # Then we define the lambda function for cell evaluation
    def eval_func(points):
        projected_points = torch.from_numpy(points).unsqueeze(0).to(device=cuda).float()
        projected_points[:, 1, :] *= -1
        geo_feat = net.get_geo_feat(space='projected', projected_points=projected_points, **smpl_dict)
        net.query(projected_points, geo_feat, space='projected')
        pred = net.get_preds(space='projected')

        return pred[0][0].detach().cpu().numpy()

        # if net.sdf:
        #     return pred[0][0].detach().cpu().numpy()
        # else:
        #     return 1 - pred[0][0].detach().cpu().numpy()

    # Then we evaluate the grid
    if use_octree:
        sdf = eval_grid_octree(coords, eval_func, num_samples=num_samples)
    else:
        sdf = eval_grid(coords, eval_func, num_samples=num_samples)

    # Finally we do marching cubes
    try:
        verts, faces, normals, values = measure.marching_cubes(sdf, thresh)
        verts = np.matmul(mat[:3, :3], verts.T) + mat[:3, 3:4]
        verts = verts.T
        normals = normals * 0.5 + 0.5
        return verts, faces, normals, values

    except Exception as e:
        print(e)
        print('error cannot marching cubes')
        # return -1


def gaussian_blur(sdf, sigma=1.):
    for i in range(sdf.shape[2]):
        sdf[:, :, i] = filters.gaussian_filter(sdf[:, :, i], sigma=sigma)
    for i in range(sdf.shape[1]):
        sdf[:, i, :] = filters.gaussian_filter(sdf[:, i, :], sigma=sigma)
    # for i in range(sdf.shape[0]):
    #     sdf[i, :, :] = filters.gaussian_filter(sdf[i, :, :], sigma=2)
    return sdf


def adjust_learning_rate(optimizer, epoch, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= gamma
    return optimizer.param_groups[0]['lr']


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


def save_samples_truncted_prob(fname, points, prob, thresh=0.5):
    '''
    Save the visualization of sampling to a ply file.
    Red points represent positive predictions.
    Green points represent negative predictions.
    :param fname: File name to save
    :param points: [N, 3] array of points
    :param prob: [N, 1] array of predictions in the range [0~1]
    :param thresh number in the range [0~1]
    :return:
    '''
    r = (prob > thresh).reshape([-1, 1]) * 255
    g = (prob < thresh).reshape([-1, 1]) * 255
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
    image_np = image.permute(1, 2, 0).numpy()[..., ::-1] * 0.5 + 0.5
    xys = ((points[:2].T + 1.0) * 0.5 * im_size)

    for xy in xys:
        if 0 <= int(xy[0]) < im_size and 0 <= int(xy[1]) < im_size:
            image_np[int(xy[1]), int(xy[0]), :] = (1, 1, 1)
    return image_np * 255


def get_mgrid(res):
    # Generate 2D pixel coordinates from an image of res x res
    pixel_coords = np.stack(np.mgrid[:res, :res])
    pixel_coords = pixel_coords / res * 2 - 1.
    pixel_coords = pixel_coords[::-1, ::-1, :].copy()
    pixel_coords = torch.Tensor(pixel_coords).float()
    return pixel_coords
