import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from tools.utils import makepath, to_cpu, to_np, to_tensor
from tools.utils import aa2rotmat, rotmat2aa, CRot2rotmat
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
            'global_orient': torch.randn(bs, 1 * 3 * 2, device=device, dtype=dtype, requires_grad=True),
            'body_pose': torch.randn(bs, 21 * 3 * 2, device=device, dtype=dtype, requires_grad=True),
            'left_hand_pose': torch.randn(bs, 15 * 3 * 2, device=device, dtype=dtype, requires_grad=True),
            'right_hand_pose': torch.randn(bs, 15 * 3 * 2, device=device, dtype=dtype, requires_grad=True),
            'jaw_pose': torch.randn(bs, 1 * 3 * 2, device=device, dtype=dtype, requires_grad=True),
            'leye_pose': torch.randn(bs, 1 * 3 * 2, device=device, dtype=dtype, requires_grad=True),
            'reye_pose': torch.randn(bs, 1 * 3 * 2, device=device, dtype=dtype, requires_grad=True),
            'transl': torch.zeros(bs, 3, device=device, dtype=dtype, requires_grad=True),
        }

        self.opt_s3 = optim.Adam([self.opt_params[k] for k in ['global_orient', 'transl', 'body_pose', 'right_hand_pose']], lr=self.cfg.lr)
        self.optimizer = [self.opt_s3]
        self.num_iters = [self.cfg.n_iters]

        self.LossL1 = nn.L1Loss(reduction='mean')

    def init_params(self, start_params):
        self.sbj_params_6D = {}
        for k in self.opt_params.keys():
            self.sbj_params_6D[k] = aa2rotmat(start_params[k]).reshape(-1, 3, 3)[:, :, :2].reshape(-1, 6) if k != 'transl' else start_params[k]
            self.opt_params[k].data = torch.repeat_interleave(self.sbj_params_6D[k], 1, dim=0)

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

    def construct_sbj_params(self, net_output, obj_transl):
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
            'transl': net_output['body_transl'].reshape(1, 3).to(self.device) + obj_transl.to(self.device)
        }
        return sbj_params

    def calc_loss(self, net_output):
        opt_params_rotmat = {}
        opt_params = {}
        for k in self.opt_params.keys():
            if k != 'transl':
                opt_params_rotmat[k] = CRot2rotmat(self.opt_params[k]).reshape(-1, 9)
                opt_params[k] = rotmat2aa(opt_params_rotmat[k]).reshape(1, -1)

        output = self.sbj_m(**opt_params, return_full_pose=True)
        verts = output.vertices
        verts_rh = verts[:, self.rhand_idx]

        rh2obj, _, _, _ = self.ch_dist(torch.tensor(verts_rh).to(self.device), torch.tensor(self.obj_verts).to(self.device))
        rh2obj_w = 10

        losses = {
            'dist_rh2obj': 10 * self.LossL1(rh2obj_w*rh2obj, rh2obj_w*net_output['hand_object_dists']),
            'grnd_contact': torch.pow(verts[:, :, 2].min() + 0.01, 2),
        }

        body_loss = {k: self.LossL1(self.sbj_params_6D[k], self.opt_params[k]) for k in ['global_orient', 'body_pose']}
        body_loss['right_hand_pose'] = .3*self.LossL1(self.sbj_params_6D['right_hand_pose'], self.opt_params['right_hand_pose'])
        body_loss['transl'] = self.LossL1(self.sbj_params['transl'], self.opt_params['transl'])

        # losses.update(body_loss)
        loss_total = torch.sum(torch.stack([torch.mean(v) for v in losses.values()]))
        losses['loss_total'] = loss_total

        return losses, verts, output

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
        fullpose = torch.cat([v for k, v in self.opt_params.items() if k != 'transl'], dim=1)
        return opt_results, fullpose

    @staticmethod
    def create_loss_message(loss_dict, stage=0, itr=0):
        ext_msg = ' | '.join(['%s = %.2e' % (k, v) for k, v in loss_dict.items() if k != 'loss_total'])
        return f'Stage:{stage:02d} - Iter:{itr:04d} - Total Loss: {loss_dict["loss_total"]:02e} | [{ext_msg}]'

