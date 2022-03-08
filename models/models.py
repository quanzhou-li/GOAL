# Adapted from GrabNet

import sys

sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch

from torch import nn
from torch.nn import functional as F
from tools.utils import CRot2rotmat

class ResBlock(nn.Module):

    def __init__(self,
                 Fin,
                 Fout,
                 n_neurons=512):
        super(ResBlock, self).__init__()
        # Feature dimension of input and output
        self.Fin = Fin
        self.Fout = Fout

        self.fc1 = nn.Linear(Fin, n_neurons)
        self.bn1 = nn.BatchNorm1d(n_neurons)

        self.fc2 = nn.Linear(n_neurons, Fout)
        self.bn2 = nn.BatchNorm1d(n_neurons)

        if Fin != Fout:
            self.fc3 = nn.Linear(Fin, Fout)

        self.ll = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, final_nl=True):
        Xin = x if self.Fin == self.Fout else self.ll(self.fc3(x))

        Xout = self.fc1(x)
        Xout = self.bn1(Xout)
        Xout = self.ll(Xout)

        Xout = self.fc2(Xout)
        Xout = self.bn2(Xout)
        Xout = Xin + Xout

        if final_nl:
            return self.ll(Xout)
        return Xout


class GNet(nn.Module):
    def __init__(self,
                 in_condition=1024+3,
                 in_params=1632,
                 n_neurons=512,
                 latentD=16,
                 **kwargs):
        super(GNet, self).__init__()
        self.latentD = latentD

        self.enc_bn0 = nn.BatchNorm1d(in_condition)
        self.enc_bn1 = nn.BatchNorm1d(in_condition + in_params)
        self.enc_rb1 = ResBlock(in_condition + in_params, n_neurons)
        self.enc_rb2 = ResBlock(in_condition + in_params + n_neurons, n_neurons)

        self.enc_mu = nn.Linear(n_neurons, latentD)
        self.enc_var = nn.Linear(n_neurons, latentD)
        self.do = nn.Dropout(p=.1, inplace=False)

        self.dec_bn1 = nn.BatchNorm1d(in_condition)
        self.dec_rb1 = ResBlock(latentD + in_condition, n_neurons)
        self.dec_rb2 = ResBlock(n_neurons + latentD + in_condition, n_neurons)

        self.dec_pose = nn.Linear(n_neurons, 55 * 6)  # Theta
        self.dec_trans = nn.Linear(n_neurons, 3)  # body translation
        self.dec_ver = nn.Linear(n_neurons, 400 * 3)  # vertices locations
        self.dec_dis = nn.Linear(n_neurons, 99)  # hand-object distances

    def encode(self, fullpose_rotmat, body_transl, verts, hand_object_dists, bps_dists, object_transl):
        '''
        :param fullpose_rotmat: N * 1 * 55 * 9
        :param body_transl: N * 3
        :param verts: N * 400 * 3
        :param dists: N * 99
        :param bps_dists: N * 1024
        :param object_transl: N * 3
        :return:
        '''
        bs = fullpose_rotmat.shape[0]

        # Get 6D rotation representation of fullpose_rotmat
        fullpose_6D = (fullpose_rotmat.reshape(bs, 1, 55, 3, 3))[:,:,:,:,:2]
        fullpose_6D = fullpose_6D.reshape(bs, 55*6)

        X = torch.cat([fullpose_6D, body_transl, verts.flatten(start_dim=1), hand_object_dists, bps_dists, object_transl], dim=1)

        X0 = self.enc_bn1(X)
        X = self.enc_rb1(X0)
        X = self.enc_rb2(torch.cat([X0, X], dim=1))

        return torch.distributions.normal.Normal(self.enc_mu(X), F.softplus(self.enc_var(X)))

    def decode(self, Zin, bps_dists, object_transl):
        bs = Zin.shape[0]

        condition = self.dec_bn1(torch.cat([bps_dists, object_transl], dim=1))

        X0 = torch.cat([Zin, condition], dim=1)
        X = self.dec_rb1(X0)
        X = self.dec_rb2(torch.cat([X0, X], dim=1))

        fullpose_6D = self.dec_pose(X)
        body_transl = self.dec_trans(X)
        verts = self.dec_ver(X)
        hand_object_dists = self.dec_dis(X)

        fullpose_rotmat = CRot2rotmat(fullpose_6D).reshape(bs, 1, 55, 9)

        return {'fullpose_rotmat': fullpose_rotmat, 'body_transl': body_transl,
                'verts': verts, 'hand_object_dists': hand_object_dists}

    def forward(self, fullpose_rotmat, body_transl, verts, hand_object_dists, bps_dists, object_transl, **kwargs):
        z = self.encode(fullpose_rotmat, body_transl, verts, hand_object_dists, bps_dists, object_transl)
        z_s = z.rsample()

        params = self.decode(z_s, bps_dists, object_transl)
        results = {'mean': z.mean, 'std': z.scale}
        results.update(params)

        return results


class MNet(nn.Module):

    def __init__(self,
                 Fin=5*55*6+5*3+400*3+400*3+99*3+1024,
                 Fout=10*55*9+10*99*3+10*3+10*400*3,
                 n_neurons=1024,
                 **kwargs):
        super(MNet, self).__init__()
        self.Fin = Fin
        self.Fout = Fout

        self.rb1 = ResBlock(Fin, n_neurons)
        self.rb2 = ResBlock(Fin + n_neurons, n_neurons)

        self.fc_pose = nn.Linear(n_neurons, 10 * 55 * 6)
        self.fc_transl = nn.Linear(n_neurons, 10 * 3)
        self.fc_verts = nn.Linear(n_neurons, 10 * 400 * 3)
        self.fc_dist = nn.Linear(n_neurons, 10 * 99 * 3)

    def forward(self,
                past_pose_rotmat,
                past_transl,
                cur_velocity,
                cur_verts,
                cur_dists_to_goal,
                bps_goal,
                **kwargs):
        '''
        :param past_pose_rotmat: N * 5 * 1 * 55 * 9
        :param past_transl: N * 5 * 3
        :param cur_velocity: N * 400 * 3
        :param cur_verts: N * 400 * 3
        :param cur_dists_to_goal: N * 99 * 3
        :param bps_goal: N * 1 * 1024
        :param kwargs:
        :return:
        '''

        bs = past_pose_rotmat.shape[0]

        pose_6D = (past_pose_rotmat.reshape(bs, 5, 55, 3, 3))[:,:,:,:,:2]
        pose_6D = pose_6D.reshape(bs, 5 * 55 * 6)

        X = torch.cat([pose_6D, past_transl.reshape(bs, 15), cur_verts.reshape(bs, 1200), cur_velocity.reshape(bs, 1200),
                       cur_dists_to_goal.reshape(bs, 99 * 3), bps_goal.reshape(bs, 1024)], dim=1)

        X0 = self.rb1(X)
        X = self.rb2(torch.cat([X, X0], dim=1))

        future_pose_6D = self.fc_pose(X)
        future_transl = self.fc_transl(X)
        future_verts = self.fc_verts(X)
        future_dists = self.fc_dist(X)

        future_pose_rotmat = future_pose_6D.reshape(bs * 10, 55 * 6)
        future_pose_rotmat = CRot2rotmat(future_pose_rotmat).reshape(bs, 10, 1, 55, 9)

        return {'future_pose_delta': future_pose_rotmat,
                'future_transl_delta': future_transl.reshape(bs, 10, 3),
                'future_dists_delta': future_dists.reshape(bs, 10, 99, 3),
                'future_verts_delta': future_verts.reshape(bs, 10, 400, 3)}
