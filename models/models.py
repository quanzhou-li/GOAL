# Adapted from GrabNet

import sys

sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch

from torch import nn
from torch.nn import functional as F
from tools.utils import rotmat2aa, CRot2rotmat


# from tools.train_tools import point2point_signed


class ResBlock(nn.Module):

    def __init__(self,
                 Fin,
                 Fout,
                 n_neurons=256):
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
                 in_condition,
                 in_params,
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

        self.dec_pose = nn.Linear(n_neurons, 55 * 3)  # Theta
        self.dec_trans = nn.Linear(n_neurons, 3)  # body translation
        self.dec_ver = nn.Linear(n_neurons, 400 * 3)  # vertices locations
        self.dec_dis = nn.Linear(n_neurons, 99 * 3)  # hand-object distances
        self.dec_horient = nn.Linear(n_neurons, 3)  # head orientation

    def encode(self, fullpose, trans_subject, vertices, distances, h_orient, trans_object, bps_object):
        '''
        :param fullpose: 55 * 3
        :param trans_subject: 3
        :param vertices: 400 * 3
        :param distances: 99 * 3
        :param h_orient: 3
        :param bps_object: 1024
        :param trans_object: 3
        :return:
        '''
        bs = fullpose.shape[0]

        X = torch.cat([fullpose, trans_subject, vertices.flatten(start_dim=1), distances.flatten(start_dim=1), h_orient, trans_object, bps_object], dim=1)

        X0 = self.enc_bn1(X)
        X = self.enc_rb1(X0)
        X = self.enc_rb2(torch.cat([X0, X], dim=1))

        return torch.distributions.normal.Normal(self.enc_mu(X), F.softplus(self.enc_var(X)))

    def decode(self, Zin, trans_object, bps_object):
        bs = Zin.shape[0]

        condition = self.dec_bn1(torch.cat([trans_object, bps_object], dim=1))

        X0 = torch.cat([Zin, condition], dim=1)
        X = self.dec_rb1(X0)
        X = self.dec_rb2(torch.cat([X0, X], dim=1))

        fullpose = self.dec_pose(X)
        trans_subject = self.dec_trans(X)
        vertices = self.dec_ver(X)
        distances = self.dec_dis(X)
        head_orient = self.dec_horient(X)

        return {'fullpose': fullpose, 'trans_subjecct': trans_subject,
                'vertices': vertices, 'distances': distances, 'head_orient': head_orient}

    def forward(self, fullpose, trans_subject, vertices, distances, h_orient, trans_object, bps_object, **kwargs):
        z = self.encode(fullpose, trans_subject, vertices, distances, h_orient, trans_object, bps_object)
        z_s = z.rsample()

        params = self.decode(z_s, trans_object, bps_object)
        results = {'mean': z.mean, 'std': z.scale}
        results.update(params)

        return results
