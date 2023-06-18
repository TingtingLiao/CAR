import torch


def index(feat, uv):
    '''
    :param feat: [B, C, H, W] image features
    :param uv: [B, 2, N] uv coordinates in the image plane, range [-1, 1]
    :return: [B, C, N] image features at the uv coordinates
    '''
    uv = uv.transpose(1, 2)  # [B, N, 2]
    uv = uv.unsqueeze(2)  # [B, N, 1, 2]
    # NOTE: for newer PyTorch, it seems that training results are degraded due to implementation diff in F.grid_sample
    samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True)[:, :, :, 0]  # [B, C, N]
    # samples = grid_sample(feat, uv)
    return samples


def grid_sample(feature, uv):
    N, C, IH, IW = feature.shape
    _, H, W, _ = uv.shape

    ix = uv[..., 0]
    iy = uv[..., 1]

    ix = ((ix + 1) / 2) * (IW - 1)
    iy = ((iy + 1) / 2) * (IH - 1)
    with torch.no_grad():
        ix_nw = torch.floor(ix)
        iy_nw = torch.floor(iy)
        ix_ne = ix_nw + 1
        iy_ne = iy_nw
        ix_sw = ix_nw
        iy_sw = iy_nw + 1
        ix_se = ix_nw + 1
        iy_se = iy_nw + 1

    nw = (ix_se - ix) * (iy_se - iy)
    ne = (ix - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix) * (iy - iy_ne)
    se = (ix - ix_nw) * (iy - iy_nw)

    with torch.no_grad():
        torch.clamp(ix_nw, 0, IW - 1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH - 1, out=iy_nw)
        torch.clamp(ix_ne, 0, IW - 1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH - 1, out=iy_ne)
        torch.clamp(ix_sw, 0, IW - 1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH - 1, out=iy_sw)
        torch.clamp(ix_se, 0, IW - 1, out=ix_se)
        torch.clamp(iy_se, 0, IH - 1, out=iy_se)

    image = feature.view(N, C, IH * IW)

    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, -1).repeat(1, C, 1))
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, -1).repeat(1, C, 1))
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, -1).repeat(1, C, 1))
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, -1).repeat(1, C, 1))

    out_val = (nw_val.view(N, C, -1) * nw.view(N, 1, -1) +
               ne_val.view(N, C, -1) * ne.view(N, 1, -1) +
               sw_val.view(N, C, -1) * sw.view(N, 1, -1) +
               se_val.view(N, C, -1) * se.view(N, 1, -1))

    return out_val


def orthogonal(points, calibrations, transforms=None):
    '''
    Compute the orthogonal projections of 3D points into the image plane by given projection matrix
    :param points: [B, 3, N] Tensor of 3D points
    :param calibrations: [B, 4, 4] Tensor of projection matrix
    :param transforms: [B, 2, 3] Tensor of image transform matrix
    :return: xyz: [B, 3, N] Tensor of xyz coordinates in the image plane
    '''
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    pts = torch.baddbmm(trans, rot, points)  # [B, 3, N]
    if transforms is not None:
        scale = transforms[:2, :2]
        shift = transforms[:2, 2:3]
        pts[:, :2, :] = torch.baddbmm(shift, scale, pts[:, :2, :])
    return pts


def perspective(points, calibrations, transforms=None):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, 3, N): 3D points
        K (bs, 3, 3): Camera rotation
        transforms: [2x3] Tensor of image transform matrix
    """
    '''
        Compute the perspective projections of 3D points into the image plane by given projection matrix
        :param points: [Bx3xN] Tensor of 3D points
        :param calibrations: [Bx4x4] Tensor of projection matrix
        :param transforms: [Bx2x3] Tensor of image transform matrix
        :return: xy: [Bx2xN] Tensor of xy coordinates in the image plane
    '''
    # Apply camera intrinsics
    points = points + calibrations[:, :3, 3:4].expand_as(points)
    xyz = torch.einsum('bij,bjk->bik', calibrations[:, :3, :3], points)
    xy = xyz[:, :2] / xyz[:, 2:3]
    if transforms is not None:
        scale = transforms[:, :2, :2]
        shift = transforms[:, :2, 2:3]
        xy = torch.baddbmm(shift, scale, xy)
    xyz = torch.cat([xy, xyz[:, 2:3, :]], 1)
    return xyz
