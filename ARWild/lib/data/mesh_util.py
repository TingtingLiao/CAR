import cv2
import trimesh
import numpy as np
import torch
import os
import sys
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
from pytorch3d.renderer.mesh import rasterize_meshes
from lib.common.render_utils import face_vertices, batch_contains, Pytorch3dRasterizer


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


def projection(points, calib, format='numpy'):
    if format == 'tensor':
        return torch.mm(calib[:3, :3], points.T).T + calib[:3, 3]
    else:
        return np.matmul(calib[:3, :3], points.T).T + calib[:3, 3]


def cal_sdf(mesh, points, edge=1.0, only_occ=False):
    """
    sdf of the points inside a mesh is negative and outside is positive
    Parameters
    ----------
    mesh
    points
    edge
    only_occ

    Returns
    -------

    """
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


def get_visibility(vertices, faces, device):
    """get the visibility of vertices

    Args:
        vertices (torch.tensor): [N,3]
        faces (torch.tensor): [N,3]
    Return:
        vis_mask (torch.tensor): [N,]
    """
    if not torch.is_tensor(vertices):
        vertices = torch.as_tensor(vertices).float()
    if not torch.is_tensor(faces):
        faces = torch.as_tensor(faces).long()
    vertices = vertices.to(device)
    faces = faces.to(device)

    xyz = vertices.clone()
    xyz = (xyz + 1.0) / 2.0

    rasterizer = Pytorch3dRasterizer(image_size=2 ** 12)
    meshes_screen = Meshes(verts=xyz[None, ...], faces=faces[None, ...])

    raster_settings = rasterizer.raster_settings

    pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
        meshes_screen,
        image_size=raster_settings.image_size,
        blur_radius=raster_settings.blur_radius,
        faces_per_pixel=raster_settings.faces_per_pixel,
        bin_size=raster_settings.bin_size,
        max_faces_per_bin=raster_settings.max_faces_per_bin,
        perspective_correct=raster_settings.perspective_correct,
        cull_backfaces=raster_settings.cull_backfaces,
    )

    vis_vertices_id = torch.unique(faces[torch.unique(pix_to_face), :])
    vis_mask = torch.zeros(size=(vertices.shape[0],))
    vis_mask[vis_vertices_id] = 1.

    # print("------------------------\n")
    # print(f"keep points : {vis_mask.sum()/len(vis_mask)}")

    return torch.logical_not(vis_mask.bool())


def build_mesh_by_poisson(mesh, num_sample=30000):
    from pypoisson import poisson_reconstruction
    """ build a graph from mesh using https://github.com/mmolero/pypoisson """
    # normals = mesh.face_normals
    normals = compute_normal(mesh.vertices, mesh.faces)
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

    for v, vt in zip(verts, uvs):
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
        file.write('vt %.4f %.4f\n' % (vt[0], vt[1]))

    for f in faces:
        f_plus = f + 1
        file.write('f %d/%d %d/%d %d/%d\n' % (f_plus[0], f_plus[0],
                                              f_plus[1], f_plus[1],
                                              f_plus[2], f_plus[2]))
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
    if save_path is not None:
        out_mesh.export(save_path)
    return out_mesh


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


def point_mesh_face_distance(meshes, pcls):
    from pytorch3d.loss.point_mesh_distance import point_face_distance
    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")

    # packed representation for pointclouds
    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()

    # point to face distance: shape (P,)
    point_to_face = point_face_distance(points, points_first_idx, tris, tris_first_idx, max_points)
    return point_to_face.view(len(meshes), -1)


def cal_sdf_batch(verts, faces, points):
    Bsize = points.shape[0]

    from pytorch3d.structures import Pointclouds
    meshes = Meshes(verts, faces)
    pcls = Pointclouds(points)
    dist = point_mesh_face_distance(meshes, pcls)
    pts_dist = torch.sqrt(dist) / torch.sqrt(torch.tensor(3.)).float()
    pts_signs = batch_contains(verts, faces, points).type_as(verts)
    pts_sdf = -(pts_dist * pts_signs).view(Bsize, -1, 1)
    return pts_sdf


def _cal_sdf_batch(verts, faces, points, only_sdf=True, cmaps=None, vis=None):
    """
    Args:
        verts: [B, N_vert, 3]
        faces: [B, N_face, 3]
        points: [B, N_point, 3]
        cmaps: [B, N_vert, 3]
        only_sdf: bool

    Returns:
         return_list: [sdf, norm, cmaps, vis]
    """
    Bsize = points.shape[0]
    # residues, pts_ind, _ = point_to_mesh_distance(points, triangles)
    residues, pts_ind, _ = knn_points(points, verts)
    pts_dist = torch.sqrt(residues.squeeze(2)) / torch.sqrt(torch.tensor(3.)).float()
    # pts_signs = 2.0 * (check_sign(verts, faces[0], points).float() - 0.5)
    pts_signs = batch_contains(verts, faces, points).type_as(verts)
    pts_sdf = -(pts_dist * pts_signs).view(Bsize, -1, 1)

    if only_sdf:
        return pts_sdf

    triangles = face_vertices(verts, faces)
    closest_triangles = torch.gather(triangles, 1, pts_ind[:, :, :, None].expand(-1, -1, 3, 3)).view(-1, 3, 3)
    bary_weights = barycentric_coordinates_of_projection(points.view(-1, 3), closest_triangles)

    normals = Meshes(verts, faces).verts_normals_padded()
    normals = face_vertices(normals, faces)
    closest_normals = torch.gather(normals, 1, pts_ind[:, :, :, None].expand(-1, -1, 3, 3)).view(-1, 3, 3)
    pts_norm = (closest_normals * bary_weights[:, :, None]).sum(1).unsqueeze(0) * torch.tensor(
        [-1.0, 1.0, -1.0]).type_as(normals).view(Bsize, -1, 3)

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


def detectBoundary(F):
    '''
    input:
        F: (F, 3) numpy triangle list
    return:
        (F) boundary flag
    '''
    tri_dic = {}
    nV = F.max()
    for i in range(F.shape[0]):
        idx = [F[i, 0], F[i, 1], F[i, 2]]

        if (idx[1], idx[0]) in tri_dic:
            tri_dic[(idx[1], idx[0])].append(i)
        else:
            tri_dic[(idx[0], idx[1])] = [i]

        if (idx[2], idx[1]) in tri_dic:
            tri_dic[(idx[2], idx[1])].append(i)
        else:
            tri_dic[(idx[1], idx[2])] = [i]

        if (idx[0], idx[2]) in tri_dic:
            tri_dic[(idx[0], idx[2])].append(i)
        else:
            tri_dic[(idx[2], idx[0])] = [i]

    v_boundary = np.array((nV + 1) * [False])
    for key in tri_dic:
        if len(tri_dic[key]) != 2:
            v_boundary[key[0]] = True
            v_boundary[key[1]] = True

    boundary = v_boundary[F[:, 0]] | v_boundary[F[:, 1]] | v_boundary[F[:, 2]]

    return boundary


def detect_valid_triangle(verts, verts_deformed, faces):
    """

    Args:
        verts:[N, 3]
        verts_deformed: [N, 3]
        faces: [M, 3]

    Returns:

    """

    def get_edges(vertices, faces):
        e1 = torch.norm(vertices[faces[:, 0]] - vertices[faces[:, 1]], p=2, dim=1, keepdim=True)
        e2 = torch.norm(vertices[faces[:, 1]] - vertices[faces[:, 2]], p=2, dim=1, keepdim=True)
        e3 = torch.norm(vertices[faces[:, 2]] - vertices[faces[:, 0]], p=2, dim=1, keepdim=True)
        e = torch.cat([e1, e2, e3], 1)
        return e

    e = get_edges(verts, faces)
    E = get_edges(verts_deformed, faces)

    max_edge = (E / e).max(1)[0]
    min_edge = (E / e).min(1)[0]
    mask = ((max_edge < 3.0) & (min_edge > 0.1) ).cpu().numpy()

    # boundary = detectBoundary(faces.cpu().numpy()[tri_mask])
    # tri_mask[tri_mask] = np.logical_not(boundary)
    return mask
