import torch
from pytorch3d.ops.knn import knn_points, knn_gather
from lib.common.geometry import orthogonal


def linear_blend_skinning(points, weight, joint_transform, return_vT=False, inverse=False):
    """
    Args:
         points: FloatTensor [batch, N, 3]
         weight: FloatTensor [batch, N, K]
         joint_transform: FloatTensor [batch, K, 4, 4]
         return_vT: return vertex transform matrix if true
         inverse: bool inverse LBS if true
    Return:
        points_deformed: FloatTensor [batch, N, 3]
    """
    if not weight.shape[0] == joint_transform.shape[0]:
        raise AssertionError('batch should be same,', weight.shape, joint_transform.shape)

    if not torch.is_tensor(points):
        points = torch.as_tensor(points).float()
    if not torch.is_tensor(weight):
        weight = torch.as_tensor(weight).float()
    if not torch.is_tensor(joint_transform):
        joint_transform = torch.as_tensor(joint_transform).float()

    batch = joint_transform.size(0)
    vT = torch.bmm(weight, joint_transform.contiguous().view(batch, -1, 16)).view(batch, -1, 4, 4)
    if inverse:
        vT = torch.inverse(vT.view(-1, 4, 4)).view(batch, -1, 4, 4)

    R, T = vT[:, :, :3, :3], vT[:, :, :3, 3]
    deformed_points = torch.matmul(R, points.unsqueeze(-1)).squeeze(-1) + T

    if return_vT:
        return deformed_points, vT
    return deformed_points


def warp_and_project_points(points, skin_weights, joint_transform, calib=None, return_vT=False, inverse=False):
    """
    Warp a canonical point cloud to multiple posed spaces and project to image space
    Args:
        points: [N, 3] Tensor of 3D points
        skin_weights: [N, J]  corresponding skinning weights of points
        joint_transform: [B, J, 4, 4] joint transform matrix of a batch of poses
        calib: [B, 24, 4, 4] calibration matrix
        return_vT: bool return vert  transform matrix if true
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
    points_posed, vT = linear_blend_skinning(points.expand(batch, -1, -1),
                                             skin_weights.expand(batch, -1, -1),
                                             joint_transform, return_vT=True, inverse=inverse)
    if calib is None:
        return points_posed

    # projection
    xyz = orthogonal(points_posed.transpose(1, 2) if not inverse else points.t().expand(batch, -1, -1),
                     calib).transpose(1, 2)

    if return_vT:
        return points_posed, xyz, vT
    return points_posed, xyz


def query_lbs_weight(points, surface_points, weights, device=None):
    """
    todo replace nn vertex to nn-face
        compute per vert-to-bone weights
    Args:
        points: FloatTensor [N, 3]
        surface_points: FloatTensor [M, 3]
        weights: FloatTensor [M, J]
        device: torch.device
    return:
        weights: FloatTensor [N, J]
    """
    if not torch.is_tensor(points):
        points = torch.as_tensor(points).float()
    if not torch.is_tensor(surface_points):
        surface_points = torch.as_tensor(surface_points).float()
    if not torch.is_tensor(weights):
        weights = torch.as_tensor(weights).float()

    if not (points.dim() == 2):
        raise AssertionError(points.shape, surface_points.shape, weights.shape)

    if device is None:
        device = points.device

    points = points.unsqueeze(0).to(device)
    surface_points = surface_points.unsqueeze(0).to(device)
    weights = weights.unsqueeze(0).to(device)

    J = weights.shape[-1]
    _, idx, _ = knn_points(points, surface_points)
    # weights = torch.gather(weights, 1, idx.expand(-1, -1, J))
    # return weights.squeeze(0)
    weights = knn_gather(weights, idx)[0, :, 0, :]
    return weights
