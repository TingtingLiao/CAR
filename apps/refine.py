import sys
import os
import torch
import imageio
from tqdm import tqdm
from skimage import img_as_ubyte
import smpl
from pytorch3d.transforms import so3_exponential_map
torch.backends.cudnn.enabled = False
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib.common.train_util import *

# from lib.common.smpl_util import MaxMixturePosePrior


class NormalRefine():
    def __init__(self, opt, render, device):
        super(NormalRefine, self).__init__()
        self.render = render
        self.device = device
        # init smpl model
        self.smpl_model = smpl.create(opt.path,
                                      model_type=opt.model_type,
                                      gender=opt.gender,
                                      use_face_contour=opt.use_face_contour).to(device)
        if opt.model_type == 'smpl_vitruvian':
            self.smpl_model.initiate_vitruvian(vitruvian_angle=opt.vitruvian_angle, device=device)

    @torch.enable_grad()
    def optimize_cloth(self, mesh, normal, n_iter=50, num_neigh=4, gif_path=None):
        device = self.device
        normal = normal.to(device)
        vertices = torch.from_numpy(mesh.vertices).float().to(device)
        N = vertices.shape[0]
        if gif_path is not None:
            writer = imageio.get_writer(gif_path, mode='I', duration=0.1)

        # get neighbors
        neighbors_idx = np.array(num_neigh * list(range(N))).reshape(num_neigh, N).T
        neighbors_idx = torch.from_numpy(neighbors_idx).to(device)
        for v in mesh.edges:
            neighbors_idx[v[0], 0] = neighbors_idx[v[0], 1]
            neighbors_idx[v[0], 1] = neighbors_idx[v[0], 2]
            neighbors_idx[v[0], 2] = v[1]
        neighbors = vertices[neighbors_idx]

        T = torch.full(vertices.shape, 0.0, device=self.device, requires_grad=True)
        R = torch.full(vertices.shape, 0.0, device=self.device, requires_grad=True)
        optimizer = torch.optim.SGD([R, T], lr=0.1, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-3, patience=5)
        losses = {
            "arap": {
                "weight": 0.1,
                "value": 0.0
            },
            "cloth": {
                "weight": 1,
                "value": 0.0
            },
            "edge": {
                "weight": 100.0,
                "value": 0.0
            },
            "normal": {
                "weight": 0.1,
                "value": 0.0
            },
            "laplacian": {
                "weight": 100.0,
                "value": 0.0
            },
            "deform": {
                "weight": 0.1,
                "value": 0.0
            }
        }

        best_loss = float('inf')
        deformed_v_final = None
        loop = tqdm(range(n_iter))
        for i in loop:
            rot = so3_exponential_map(R)
            deformed_v = neighbors - vertices.unsqueeze(1)
            deformed_v = torch.matmul(rot.unsqueeze(1).repeat(1, num_neigh, 1, 1), deformed_v.unsqueeze(3)).squeeze(3)
            deformed_v = deformed_v + vertices.unsqueeze(1) + T.unsqueeze(1)

            arap_loss = (deformed_v - (neighbors + T[neighbors_idx])).pow(2).exp().mean()
            losses['arap']['value'] = arap_loss
            deformed_v = deformed_v.mean(1)

            self.render.load_mesh(deformed_v, mesh.faces, use_normal=True)
            update_mesh_shape_prior_losses(self.render.mesh, losses)

            # optimize normal
            P_normal_F, P_normal_B = self.render.get_clean_image()
            diff_F_cloth = torch.abs(P_normal_F - normal[:, :3])
            losses['cloth']['value'] = (diff_F_cloth.mean())
            losses['deform']['value'] = (deformed_v - vertices).pow(2).exp().mean()

            cloth_loss = torch.tensor(0.0, device=device)
            pbar_desc = ""
            for k in losses.keys():
                cloth_loss_per_cls = losses[k]['value'] * losses[k]["weight"]
                pbar_desc += f"{k}: {cloth_loss_per_cls:.3f} | "
                cloth_loss += cloth_loss_per_cls

            if losses['cloth']['value'].item() < best_loss:
                best_loss = losses['cloth']['value'].item()
                deformed_v_final = deformed_v.clone().detach().cpu()
            loop.set_description(pbar_desc)

            optimizer.zero_grad()
            cloth_loss.backward(retain_graph=True)
            optimizer.step()
            scheduler.step(cloth_loss)

            if gif_path is not None:
                visual_frames = torch.cat(
                    [P_normal_F[0], normal[:3]], 2
                ).permute(1, 2, 0).detach().cpu().numpy() * 0.5 + 0.5
                writer.append_data(img_as_ubyte(visual_frames))
        optim_mesh = trimesh.Trimesh(deformed_v_final.numpy(), mesh.faces)
        return optim_mesh

    @torch.enable_grad()
    def optimize_pose(self, mesh, lbs_weights, calib, init_pose, gt_mask, gif_path=None):
        """
        optimize pose for predicted mesh
        Args:
            mesh:
            lbs_weights:
            calib:
            init_pose:
            gt_mask:
            gif_path
        Returns:

        """
        vertices_canon = torch.from_numpy(mesh.vertices).float().to(self.device)
        lbs_weights = lbs_weights.to(self.device)
        calib = calib.to(self.device)
        gt_mask = gt_mask.to(self.device)

        optimed_pose = init_pose.clone().to(self.device)  # [1, 24, 3, 3]
        optimed_pose.requires_grad_()
        # PosePrior = MaxMixturePosePrior(device=self.device)
        if gif_path is not None:
            writer = imageio.get_writer(gif_path, mode='I', duration=0.1)
        optimizer_smpl = torch.optim.SGD(
            [optimed_pose],
            lr=0.01,
            momentum=0.9)

        scheduler_smpl = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_smpl, min_lr=1e-5, patience=5)

        loop_smpl = tqdm(range(100))

        for i in loop_smpl:
            smpl_out = self.smpl_model(global_orient=optimed_pose[:, :1],
                                       body_pose=optimed_pose[:, 1:],
                                       custom_out=True,
                                       pose2rot=False)
            joint_transform = smpl_out.joint_transform[:, :24]
            _, vertices_posed = warp_and_project_points(vertices_canon, lbs_weights, joint_transform, calib)
            vertices_posed[:, :, 1] *= -1
            self.render.load_mesh(vertices_posed[0], mesh.faces)
            pred_mask = torch.stack(self.render.get_silhouette_image([0]))
            diff = (pred_mask - gt_mask).abs()
            mask_loss = diff.mean()

            # rotation matrix to pose angle
            # pose_prior_loss = PosePrior(optimed_pose) * 0.001
            loss = mask_loss
            optimizer_smpl.zero_grad()
            loss.backward(retain_graph=True)
            optimizer_smpl.step()
            scheduler_smpl.step(loss)

            loop_smpl.set_description(
                f"| mask loss: {loss:.3f} | lr: {optimizer_smpl.param_groups[0]['lr']:.3f}")

            if gif_path is not None:
                visual_frames = diff.detach().cpu().numpy()[0]
                writer.append_data(img_as_ubyte(visual_frames))

        optim_mesh = trimesh.Trimesh(vertices_posed.detach().cpu().numpy()[0], mesh.faces)
        return optim_mesh

    def optimize_mesh(self, mesh, lbs_weights, calib, pose, mask, normal, refine_type='mesh_normal', **kwargs):
        """
          Given a canonical mesh and mask optimize its target pose
        Args:
            mesh:
            lbs_weights: [N, 3]
            calib: [1, 4, 4]
            pose: [1, 24, 3, 3]
            mask: [1, 1, 512, 512]
            normal:  [1, 6, 512, 512]
            refine_type:
            **kwargs:
        Returns:
        """
        posed_mesh = self.optimize_pose(
            mesh, lbs_weights, calib, pose, mask)
        if refine_type == 'mesh_pose':
            return posed_mesh
        elif refine_type == 'mesh_normal':
            return self.optimize_cloth(posed_mesh, normal)
        elif refine_type == 'params':  # todo
            pass
        else:
            raise ValueError('refine_type must be mesh_pose or mesh_normal or params')

    @torch.enable_grad()
    def icon_optimize_cloth(self, mesh, inter, gif_path=None):
        """
        Args:
            mesh: trimesh.Trimesh
            inter: predicted normal image for cloth optimize
            gif_path: gif path
        Returns:
        """
        if gif_path is not None:
            writer = imageio.get_writer(gif_path, mode='I', duration=0.05)

        verts_pr = torch.from_numpy(mesh.vertices).float().to(self.device)
        faces_pr = torch.from_numpy(mesh.faces).long().to(self.device)
        inter = inter.float().to(self.device)

        losses = {
            "cloth": {
                "weight": 5.0,
                "value": 0.0
            },
            "edge": {
                "weight": 100.0,
                "value": 0.0
            },
            "normal": {
                "weight": 0.2,
                "value": 0.0
            },
            "laplacian": {
                "weight": 100.0,
                "value": 0.0
            },
            "smpl": {
                "weight": 1.0,
                "value": 0.0
            },
            "deform": {
                "weight": 20.0,
                "value": 0.0
            }
        }

        deform_verts = torch.full(verts_pr.shape,
                                  0.0,
                                  device=self.device,
                                  requires_grad=True)
        optimizer_cloth = torch.optim.SGD([deform_verts],
                                          lr=1e-1,
                                          momentum=0.9)
        scheduler_cloth = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_cloth, min_lr=1e-3, patience=5)
        # cloth optimization
        loop_cloth = tqdm(range(100))

        for i in loop_cloth:
            offset_v = deform_verts.clone()
            dist, false_ids = torch.topk(deform_verts.abs().sum(dim=1), 30)
            offset_v[false_ids] = offset_v.mean(dim=0)
            deform_verts_pr = verts_pr + offset_v

            self.render.load_mesh(
                deform_verts_pr.unsqueeze(0).to(self.device),
                faces_pr.unsqueeze(0).to(self.device).long(),
                use_normal=True)
            P_normal_F, P_normal_B = self.render.get_clean_image()
            update_mesh_shape_prior_losses(self.render.mesh, losses)
            diff_F_cloth = torch.abs(P_normal_F[0] - inter[:3])
            diff_B_cloth = torch.abs(P_normal_B[0] - inter[3:])
            losses['cloth']['value'] = (diff_F_cloth.mean() + diff_B_cloth.mean()) * 0.5
            losses['deform']['value'] = dist.mean()

            # Weighted sum of the losses
            cloth_loss = torch.tensor(0.0, device=self.device)
            pbar_desc = ""
            for k in losses.keys():
                if k != 'smpl':
                    cloth_loss_per_cls = losses[k]['value'] * losses[k][
                        "weight"]
                    pbar_desc += f"{k}: {cloth_loss_per_cls:.3f} | "
                    cloth_loss += cloth_loss_per_cls

            loop_cloth.set_description(pbar_desc)

            optimizer_cloth.zero_grad()
            cloth_loss.backward(retain_graph=True)
            optimizer_cloth.step()
            scheduler_cloth.step(cloth_loss)

            if gif_path is not None:
                visual_frames = torch.cat(
                    [P_normal_F[0], inter[:3]], -1).permute(1, 2, 0).detach().cpu().numpy() * 0.5 + 0.5
                writer.append_data(img_as_ubyte(visual_frames))

        deform_verts = deform_verts.flatten().detach()
        deform_verts[torch.topk(torch.abs(deform_verts), 30)[1]] = deform_verts.mean()
        deform_verts = deform_verts.view(-1, 3)

        verts_pr += deform_verts
        return verts_pr
