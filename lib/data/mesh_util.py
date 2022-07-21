import cv2
import trimesh
import numpy as np
import torch
import os
import sys
import os.path as osp
import matplotlib
import matplotlib.cm as cm
from scipy.spatial.ckdtree import cKDTree

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from pytorch3d.structures import Meshes
from pytorch3d.ops.knn import knn_points
from pytorch3d.loss import (
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
# from kaolin.ops.mesh import check_sign
# from kaolin.metrics.trianglemesh import point_to_mesh_distance
from lib.common.render_utils import face_vertices, batch_contains


def scalar_to_color(val, min=None, max=None):
    if min is None:
        min = val.min()
    if max is None:
        max = val.max()

    norm = matplotlib.colors.Normalize(vmin=min, vmax=max, clip=True)
    # use jet colormap
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.jet)

    return mapper.to_rgba(val)[:, :3]


# Losses to smooth / regularize the mesh shape
def update_mesh_shape_prior_losses(mesh, losses):
    # and (b) the edge length of the predicted mesh
    losses["edge"]['value'] = mesh_edge_loss(mesh)
    # mesh normal consistency
    losses["normal"]['value'] = mesh_normal_consistency(mesh)
    # mesh laplacian smoothing
    losses["laplacian"]['value'] = mesh_laplacian_smoothing(mesh, method="uniform")


def make_rotate(rx, ry, rz):
    sinX = np.sin(rx)
    sinY = np.sin(ry)
    sinZ = np.sin(rz)

    cosX = np.cos(rx)
    cosY = np.cos(ry)
    cosZ = np.cos(rz)

    Rx = np.zeros((3, 3))
    Rx[0, 0] = 1.0
    Rx[1, 1] = cosX
    Rx[1, 2] = -sinX
    Rx[2, 1] = sinX
    Rx[2, 2] = cosX

    Ry = np.zeros((3, 3))
    Ry[0, 0] = cosY
    Ry[0, 2] = sinY
    Ry[1, 1] = 1.0
    Ry[2, 0] = -sinY
    Ry[2, 2] = cosY

    Rz = np.zeros((3, 3))
    Rz[0, 0] = cosZ
    Rz[0, 1] = -sinZ
    Rz[1, 0] = sinZ
    Rz[1, 1] = cosZ
    Rz[2, 2] = 1.0

    R = np.matmul(np.matmul(Rz, Ry), Rx)
    return R


def normalize_v3(arr):
    """ Normalize a numpy array of 3 component vectors shape=(n,3) """
    lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
    eps = 0.00000001
    lens[lens < eps] = eps
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr


def compute_normal(vertices, faces):
    # Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    # Calculate the normal for all the triangles,
    # by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
    # the triangles' normal.
    # Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    norm[faces[:, 0]] += n
    norm[faces[:, 1]] += n
    norm[faces[:, 2]] += n
    normalize_v3(norm)

    return norm


def cal_sdf(mesh, points, edge=1.0, only_occ=False):
    pts_occ = mesh.contains(points)
    if only_occ:
        return pts_occ

    verts = torch.from_numpy(mesh.vertices).float()
    points = torch.from_numpy(points).float()

    mesh_tree = cKDTree(verts)
    pts_dist, pts_ind = mesh_tree.query(points)
    pts_dist /= torch.sqrt(torch.tensor(3 * (edge ** 2)))  # p=2

    pts_sdf = (pts_dist * (np.logical_not(pts_occ) - 0.5) * 2)[..., None].float()
    return pts_sdf, pts_occ


def build_mesh_by_poisson(mesh, num_sample=30000):
    from pypoisson import poisson_reconstruction
    """ build a graph from mesh using https://github.com/mmolero/pypoisson """
    normals = mesh.face_normals
    vertices = mesh.vertices
    if num_sample > 0:
        idx = np.random.randint(0, vertices.shape[0], size=(num_sample,))
        vertices = mesh.vertices[idx]
        normals = normals[idx]

    new_faces, new_vertices = poisson_reconstruction(vertices, normals)
    return trimesh.Trimesh(new_vertices, new_faces)


def save_obj_mesh(mesh_path, verts, faces=None):
    file = open(mesh_path, 'w')

    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))

    if faces is not None:
        for f in faces:
            f_plus = f + 1
            file.write('f %d %d %d\n' % (f_plus[0], f_plus[1], f_plus[2]))
    file.close()


def save_obj_mesh_with_color(mesh_path, verts, faces, colors):
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        c = colors[idx]
        file.write('v %.4f %.4f %.4f %.4f %.4f %.4f\n' % (v[0], v[1], v[2], c[0], c[1], c[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()


def save_ply_mesh_with_color(ply_path, points, color):
    """
    Args:
        ply_path: str to save .ply file
        points: [N, 3]
        color: [N, 3]
    """
    assert points.shape == color.shape and points.shape[1] == 3
    to_save = np.concatenate([points, color], axis=-1)
    np.savetxt(ply_path,
               to_save,
               fmt='%.6f %.6f %.6f %d %d %d',
               comments='',
               header=(
                   'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float '
                   'z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
                   points.shape[0])
               )


def save_obj_mesh_with_uv(mesh_path, verts, faces, uvs):
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        vt = uvs[idx]
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
        file.write('vt %.4f %.4f\n' % (vt[0], vt[1]))

    for f in faces:
        f_plus = f + 1
        file.write('f %d/%d %d/%d %d/%d\n' % (f_plus[0], f_plus[0],
                                              f_plus[2], f_plus[2],
                                              f_plus[1], f_plus[1]))
    file.close()


def mesh_clean(mesh, save_path=None):
    """ clean mesh """
    cc = mesh.split(only_watertight=False)
    out_mesh = cc[0]
    bbox = out_mesh.bounds
    height = bbox[1, 0] - bbox[0, 0]
    for c in cc:
        bbox = c.bounds
        if height < bbox[1, 0] - bbox[0, 0]:
            height = bbox[1, 0] - bbox[0, 0]
            out_mesh = c
    if save_path:
        out_mesh.export(save_path)
    return out_mesh


def linear_blend_skinning(points, weight, joint_transform, return_vT=False):
    """
    Args:
         points: FloatTensor [batch, N, 3]
         weight: FloatTensor [batch, N, K]
         joint_transform: FloatTensor [batch, K, 4, 4]
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

    deformed_points = torch.matmul(vT[:, :, :3, :3], points[:, :, :, None])[..., 0] + vT[:, :, :3, 3]
    if return_vT:
        return deformed_points, vT
    return deformed_points


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
    weights = torch.gather(weights, 1, idx.expand(-1, -1, J))
    return weights.squeeze(0)


def barycentric_coordinates_of_projection(points, vertices):
    ''' https://github.com/MPI-IS/mesh/blob/master/mesh/geometry/barycentric_coordinates_of_projection.py
    '''
    """Given a point, gives projected coords of that point to a triangle
    in barycentric coordinates.
    See
        **Heidrich**, Computing the Barycentric Coordinates of a Projected Point, JGT 05
        at http://www.cs.ubc.ca/~heidrich/Papers/JGT.05.pdf

    :param p: point to project. [B, 3]
    :param v0: first vertex of triangles. [B, 3]
    :returns: barycentric coordinates of ``p``'s projection in triangle defined by ``q``, ``u``, ``v``
            vectorized so ``p``, ``q``, ``u``, ``v`` can all be ``3xN``
    """
    # (p, q, u, v)
    v0, v1, v2 = vertices[:, 0], vertices[:, 0], vertices[:, 0]
    p = points

    q = v0
    u = v1 - v0
    v = v2 - v0
    n = torch.cross(u, v)
    s = torch.sum(n * n, dim=1)
    # If the triangle edges are collinear, cross-product is zero,
    # which makes "s" 0, which gives us divide by zero. So we
    # make the arbitrary choice to set s to epsv (=numpy.spacing(1)),
    # the closest thing to zero
    s[s == 0] = 1e-6
    oneOver4ASquared = 1.0 / s
    w = p - q
    b2 = torch.sum(torch.cross(u, w) * n, dim=1) * oneOver4ASquared
    b1 = torch.sum(torch.cross(w, v) * n, dim=1) * oneOver4ASquared
    weights = torch.stack((1 - b1 - b2, b1, b2), dim=-1)
    # check barycenric weights
    # p_n = v0*weights[:,0:1] + v1*weights[:,1:2] + v2*weights[:,2:3]
    return weights


def cal_sdf_batch(verts, faces, points, cmaps=None, vis=None):
    """
    Args:
        verts: [B, N_vert, 3]
        faces: [B, N_face, 3]
        points: [B, N_point, 3]
        cmaps: [B, N_vert, 3]
        vis:

    Returns:
         return_list: [sdf, norm, cmaps, vis]
    """
    Bsize = points.shape[0]
    normals = Meshes(verts, faces).verts_normals_padded()
    triangles = face_vertices(verts, faces)

    # residues, pts_ind, _ = point_to_mesh_distance(points, triangles)
    residues, pts_ind, _ = knn_points(points, verts)

    closest_triangles = torch.gather(triangles, 1, pts_ind[:, :, :, None].expand(-1, -1, 3, 3)).view(-1, 3, 3)
    bary_weights = barycentric_coordinates_of_projection(points.view(-1, 3), closest_triangles)

    # sdf
    pts_dist = torch.sqrt(residues.squeeze(2)) / torch.sqrt(torch.tensor(3.)).float()
    # pts_signs = 2.0 * (check_sign(verts, faces[0], points).float() - 0.5)
    pts_signs = (batch_contains(verts, faces, points)).type_as(verts)

    pts_sdf = (pts_dist * torch.logical_not(pts_signs)).view(Bsize, -1, 1)

    # normal
    normals = face_vertices(normals, faces)
    closest_normals = torch.gather(normals, 1, pts_ind[:, :, :, None].expand(-1, -1, 3, 3)).view(-1, 3, 3)
    pts_norm = (closest_normals * bary_weights[:, :, None]).sum(1).unsqueeze(0) * torch.tensor(
        [-1.0, 1.0, -1.0]).type_as(normals)
    pts_norm = pts_norm.view(Bsize, -1, 3)

    return_list = [pts_sdf, pts_norm]

    if cmaps is not None:
        cmaps = face_vertices(cmaps, faces)
        closest_cmaps = torch.gather(cmaps, 1, pts_ind[:, :, :, None].expand(-1, -1, 3, 3)).view(-1, 3, 3)
        pts_cmap = (closest_cmaps * bary_weights[:, :, None]).sum(1).unsqueeze(0)
        return_list.append(pts_cmap.view(Bsize, -1, 3))

    if vis is not None:
        vis = face_vertices(vis, faces)
        closest_vis = torch.gather(vis, 1, pts_ind[:, :, :, None].expand(-1, -1, 3, 1)).view(-1, 3, 1)
        pts_vis = (closest_vis * bary_weights[:, :, None]).sum(1).unsqueeze(0).ge(1e-1)
        return_list.append(pts_vis.view(Bsize, -1, 1))

    return return_list


def normalize_vertices(vertices, return_params=False):
    """ normalize vertices to [-1, 1]
    Args:
        vertices: FloatTensor [N, 3]
        return_params: return center and scale if True
    Return:
        normalized_v: FloatTensor [N, 3]
    """
    if not torch.is_tensor(vertices):
        vertices = torch.as_tensor(vertices)
    vmax = vertices.max(0)[0]
    vmin = vertices.min(0)[0]
    center = -0.5 * (vmax + vmin)
    scale = (1. / (vmax - vmin).max()).item()
    normalized_v = (vertices + center[None, :]) * scale * 2.
    if return_params:
        return normalized_v, center, scale
    return normalized_v


class SMPLX():
    def __init__(self):
        self.current_dir = osp.join(osp.dirname(__file__), "../../data/smpl_related")
        self.smpl_verts_path = osp.join(self.current_dir, "smpl_data/smpl_verts.npy")
        self.smplx_verts_path = osp.join(self.current_dir, "smpl_data/smplx_verts.npy")
        self.faces_path = osp.join(self.current_dir, "smpl_data/smplx_faces.npy")
        self.cmap_vert_path = osp.join(self.current_dir, "smpl_data/smplx_cmap.npy")

        self.faces = np.load(self.faces_path)
        self.verts = np.load(self.smplx_verts_path)
        self.smpl_verts = np.load(self.smpl_verts_path)

        self.model_dir = osp.join(self.current_dir, "models")
        self.tedra_dir = osp.join(self.current_dir, "../tedra_data")

    def get_smpl_mat(self, vert_ids):

        mat = torch.as_tensor(np.load(self.cmap_vert_path)).float()
        return mat[vert_ids, :]

    def smpl2smplx(self, vert_ids=None):
        """convert vert_ids in smpl to vert_ids in smplx

        Args:
            vert_ids ([int.array]): [n, knn_num]
        """
        smplx_tree = cKDTree(self.verts, leafsize=1)
        _, ind = smplx_tree.query(self.smpl_verts, k=1)  # ind: [smpl_num, 1]

        if vert_ids is not None:
            smplx_vert_ids = ind[vert_ids]
        else:
            smplx_vert_ids = ind

        return smplx_vert_ids

    def smplx2smpl(self, vert_ids=None):
        """convert vert_ids in smplx to vert_ids in smpl

        Args:
            vert_ids ([int.array]): [n, knn_num]
        """
        smpl_tree = cKDTree(self.smpl_verts, leafsize=1)
        _, ind = smpl_tree.query(self.verts, k=1)  # ind: [smplx_num, 1]
        if vert_ids is not None:
            smpl_vert_ids = ind[vert_ids]
        else:
            smpl_vert_ids = ind

        return smpl_vert_ids


def detect_valid_triangle(canon_verts, posed_verts, faces):
    """
    detect valid faces within length
    :param posed_verts: Nx3
    :param faces: NFx3
    :return: triangle mask
    """
    if not torch.is_tensor(canon_verts):
        canon_verts = torch.as_tensor(canon_verts)
    if not torch.is_tensor(posed_verts):
        posed_verts = torch.as_tensor(posed_verts)
    if not torch.is_tensor(faces):
        faces = torch.as_tensor(faces)

    e1 = torch.norm(posed_verts[faces[:, 0]] - posed_verts[faces[:, 1]], p=2, dim=1, keepdim=True)
    e2 = torch.norm(posed_verts[faces[:, 1]] - posed_verts[faces[:, 2]], p=2, dim=1, keepdim=True)
    e3 = torch.norm(posed_verts[faces[:, 2]] - posed_verts[faces[:, 0]], p=2, dim=1, keepdim=True)
    e = torch.cat([e1, e2, e3], 1)

    E1 = torch.norm(canon_verts[faces[:, 0]] - canon_verts[faces[:, 1]], p=2, dim=1, keepdim=True)
    E2 = torch.norm(canon_verts[faces[:, 1]] - canon_verts[faces[:, 2]], p=2, dim=1, keepdim=True)
    E3 = torch.norm(canon_verts[faces[:, 2]] - canon_verts[faces[:, 0]], p=2, dim=1, keepdim=True)
    E = torch.cat([E1, E2, E3], 1)

    max_edge = (E / e).max(1)[0]
    min_edge = (E / e).min(1)[0]

    # mask = 1.0 - (((max_edge > 2.0) & flag_tri) | (max_edge > 3.0) | (min_edge < 0.1)).cpu().float().numpy()
    mask = 1.0 - ((max_edge > 3) | (min_edge < 0.1)).cpu().float().numpy()
    tri_mask = mask > 0.5
    return tri_mask


def get_vitruvian_transform():
    import smpl
    model = smpl.create('./data/smpl_related/models', model_type='smpl_vitruvian')
    with torch.no_grad():
        model.initiate_vitruvian(vitruvian_angle=-45)
        output = model(custom_out=True, pose2rot=True)
        joint_transform = output.joint_transform[:, :24]
    return joint_transform


def remove_self_contact_triangle(mesh, lbs_weights):
    """
    Args:
        mesh: trimesh.Trimesh  canonical mesh in T pose
        lbs_weights: [N, 24],
    Returns:
        new_mesh: trimesh.Trimesh
    """
    assert lbs_weights.shape[0] == mesh.vertices.shape[0]

    origin_verts = torch.from_numpy(mesh.vertices).float()
    origin_faces = torch.from_numpy(mesh.faces).long()

    if not torch.is_tensor(lbs_weights):
        lbs_weights = torch.from_numpy(lbs_weights).float()

    vitruvian_transform = get_vitruvian_transform()
    posed_vertices = linear_blend_skinning(origin_verts[None], lbs_weights[None], vitruvian_transform)[0]
    triangle_mask = detect_valid_triangle(origin_verts, posed_vertices, origin_faces)
    faces = origin_faces[triangle_mask].numpy()
    new_mesh = trimesh.Trimesh(mesh.vertices, faces)
    # re-build water-tight mesh this step is pretty slow and requires
    # import time
    # s = time.time()
    # print(time.time() - s, new_mesh.vertices.shape)
    # new_mesh = build_mesh_by_poisson(new_mesh)
    # print(time.time() - s, new_mesh.vertices.shape)
    return new_mesh


