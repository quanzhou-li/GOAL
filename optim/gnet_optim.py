import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from tools.utils import makepath, to_cpu, to_np, to_tensor
from tools.utils import aa2rotmat, rotmat2aa
from bps_torch.bps import bps_torch
import chamfer_distance as chd


class GNetOptim(nn.Module):
    def __init__(self, sbj_m, obj_m, cfg):
        super(GNetOptim, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        self.cfg = cfg
        self.sbj_m = sbj_m
        self.obj_m = obj_m

        self.config_optimizers()

        self.rhand_idx = to_tensor(np.load(cfg.rh_idx)['indices'], dtype=torch.long)

        self.bps_torch = bps_torch()
        self.ch_dist = chd.ChamferDistance()

    def config_optimizers(self):
        self.bs = 1
        bs = self.bs
        device = self.device
        dtype = self.dtype
        self.opt_params = {
            'global_orient': torch.randn(bs, 1 * 3, device=device, dtype=dtype, requires_grad=True),
            'body_pose': torch.randn(bs, 21 * 3, device=device, dtype=dtype, requires_grad=True),
            'left_hand_pose': torch.randn(bs, 15 * 3, device=device, dtype=dtype, requires_grad=True),
            'right_hand_pose': torch.randn(bs, 15 * 3, device=device, dtype=dtype, requires_grad=True),
            'jaw_pose': torch.randn(bs, 1 * 3, device=device, dtype=dtype, requires_grad=True),
            'leye_pose': torch.randn(bs, 1 * 3, device=device, dtype=dtype, requires_grad=True),
            'reye_pose': torch.randn(bs, 1 * 3, device=device, dtype=dtype, requires_grad=True),
            'transl': torch.zeros(bs, 3, device=device, dtype=dtype, requires_grad=True),
        }

        self.opt_s3 = optim.Adam([self.opt_params[k] for k in ['global_orient', 'transl', 'body_pose', 'right_hand_pose']], lr=self.cfg.lr)
        self.optimizer = [self.opt_s3]
        self.num_iters = [self.cfg.n_iters]

        self.LossL1 = nn.L1Loss(reduction='mean')

    def init_params(self, start_params):
        self.sbj_params = start_params
        for k in self.opt_params.keys():
            self.opt_params[k].data = torch.repeat_interleave(start_params[k], 1, dim=0)

    def get_smplx_verts(self, net_output, obj_params):
        obj_output = self.obj_m(**obj_params)
        self.obj_verts = obj_output.vertices
        sbj_params = self.construct_sbj_params(net_output, obj_params['transl'])
        self.init_params(sbj_params)
        with torch.no_grad():
            sbj_output = self.sbj_m(**sbj_params)
            v = sbj_output.vertices.reshape(-1, 10475, 3)
            verts_rh = v[:, self.rhand_idx]
        return verts_rh

    def constrcut_sbj_params(self, net_output, obj_transl):
        fullpose = rotmat2aa(net_output['fullpose_rotmat'])
        fullpose = fullpose.reshape(1, 165)
        sbj_params = {
            'global_orient': fullpose[:, :3].float(),
            'body_pose': fullpose[:, 3:66].float(),
            'jaw_pose': fullpose[:, 66:69].float(),
            'leye_pose': fullpose[:, 69:72].float(),
            'reye_pose': fullpose[:, 72:75].float(),
            'left_hand_pose': fullpose[:, 75:120].float(),
            'right_hand_pose': fullpose[:, 120:165].float(),
            'transl': net_output['body_transl'].reshape(1, 3) + obj_transl
        }
        return sbj_params

    def calc_loss(self, net_output):
        opt_params = {k:aa2rotmat(v) for k,v in self.opt_params.items() if k!='transl'}
        opt_params['transl'] = self.opt_params['transl']

        output = self.sbj_m(**self.opt_params, return_full_pose=True)
        verts = output.vertices
        verts_rh = verts[:, self.rhand_idx]

        rh2obj, _, _, _ = self.ch_dist(torch.tensor(verts_rh), torch.tensor(self.obj_verts))
        rh2obj_w = 1

        losses = {
            'dist_rh2obj': self.LossL1(rh2obj_w*rh2obj, rh2obj_w*net_output['hand_object_dists']),
            'grnd_contact': torch.abs(verts[:,:,2].min() + 0.01),
        }

        body_loss = {k: self.LossL1(self.sbj_params[k], self.opt_params[k]) for k in ['global_orient', 'body_pose', 'left_hand_pose']}
        body_loss['right_hand_pose'] = .3*self.LossL1(self.sbj_params['right_hand_pose'], self.opt_params['right_hand_pose'])
        body_loss['transl'] = self.LossL1(self.sbj_params['transl'], self.opt_params['transl'])

        losses.update(body_loss)
        loss_total = torch.sum(torch.stack([torch.mean(v) for v in losses.values()]))
        losses['loss_total'] = loss_total

        return losses, verts, output

    def get_penetration(self, source_mesh, target_mesh):
        source_verts = source_mesh.verts_packed()
        source_normals = source_mesh.verts_normals_packed()

        target_verts = target_mesh.verts_packed()

        src2trgt, trgt2src, src2trgt_idx, trgt2src_idx = self.ch_dist(source_verts.reshape(1, -1, 3).to(self.device),
                                                                      target_verts.reshape(1, -1, 3).to(self.device))
        source2target_correspond = target_verts[src2trgt_idx.data.view(-1).long()]

        distance_vector = source_verts - source2target_correspond

        in_out = torch.bmm(source_normals.view(-1, 1, 3), distance_vector.view(-1, 3, 1)).view(-1).sign()

        src2trgt_signed = src2trgt * in_out

        return src2trgt_signed

    def fitting(self, net_output, obj_params):

        _ = self.get_smplx_verts(net_output, obj_params)

        for stg, optimizer in enumerate(self.optimizer):
            for itr in range(self.num_iters[stg]):
                optimizer.zero_grad()
                losses, opt_verts, opt_output = self.calc_loss(net_output)
                losses['loss_total'].backward(retain_graph=True)
                optimizer.step()
                if itr % 10 == 0:
                    print(self.create_loss_message(losses, stg, itr))

        opt_results = {k: v for k, v in self.opt_params.items()}
        return opt_results

    @staticmethod
    def create_loss_message(loss_dict, stage=0, itr=0):
        ext_msg = ' | '.join(['%s = %.2e' % (k, v) for k, v in loss_dict.items() if k != 'loss_total'])
        return f'Stage:{stage:02d} - Iter:{itr:04d} - Total Loss: {loss_dict["loss_total"]:02e} | [{ext_msg}]'
