import numpy as np
import torch
import os, glob
import cv2
import smplx
import argparse

from models.models import GNet

from tools.objectmodel import ObjectModel
from tools.meshviewer import Mesh, MeshViewer, colors
from tools.utils import params2torch
from tools.utils import to_cpu
from tools.utils import euler
from tools.utils import rotmat2aa
from tools.cfg_parser import Config
from optim.gnet_optim import GNetOptim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def reshape_seq_data(seq_data):
    for k1 in ['body']:
        for k2 in seq_data[k1]['params']:
            seq_data[k1]['params'][k2] = seq_data[k1]['params'][k2].reshape(1, len(seq_data[k1]['params'][k2]))
    return seq_data

def parse_npz(sequence, allow_pickle=True):
    seq_data = np.load(sequence, allow_pickle=allow_pickle)
    data = {}
    for k in seq_data.files:
        if k in ['bps_dists', 'obj_transl']:
            data[k] = seq_data[k]
        else:
            data[k] = seq_data[k].item()
    return data


def construct_sbj_verts(sbj_m, fullpose, sbj_transl, obj_transl):
    sbj_m = sbj_m.to(device)
    fullpose = fullpose.to(device)
    sbj_transl = sbj_transl.to(device)
    obj_transl = obj_transl.to(device)
    sbj_parms = {
        'global_orient': fullpose[:, :3],
        'body_pose': fullpose[:, 3:66],
        'jaw_pose': fullpose[:, 66:69],
        'leye_pose': fullpose[:, 69:72],
        'reye_pose': fullpose[:, 72:75],
        'left_hand_pose': fullpose[:, 75:120],
        'right_hand_pose': fullpose[:, 120:165],
        'transl': sbj_transl.reshape(1, 3) + obj_transl
    }
    verts_sbj = to_cpu(sbj_m(**sbj_parms).vertices)
    return verts_sbj


def render_img(cfg):
    gnet = GNet().to(device)
    gnet.load_state_dict(torch.load(cfg.model_path))
    gnet.eval()

    mv = MeshViewer(offscreen=True)

    # set the camera pose
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = euler([80, 15, 0], 'xzx')
    camera_pose[:3, 3] = np.array([1.2, -1.8, 1.5])
    mv.update_camera_pose(camera_pose)

    test_data = parse_npz(cfg.data_path)

    bps_dists = torch.tensor(test_data['bps_dists'].reshape(1, 1024)).to(device)
    object_transl = torch.tensor(test_data['obj_transl'].reshape(1, 3)).to(device)
    # object_transl += torch.tensor([0, -1.2, 0])
    dist = torch.distributions.normal.Normal(
            loc=torch.tensor(np.zeros([1, 16]), requires_grad=False),
            scale=torch.tensor(np.ones([1, 16]), requires_grad=False)
        )
    Zs = dist.rsample().float().to(device)

    results = gnet.decode(Zs, bps_dists, object_transl)
    fullpose = rotmat2aa(results['fullpose_rotmat'])
    fullpose = fullpose.reshape(1, 165)

    sbj_mesh = os.path.join(cfg.tool_meshes, test_data['body']['vtemp'])
    sbj_vtemp = np.array(Mesh(filename=sbj_mesh).vertices)

    sbj_m = smplx.create(model_path=cfg.smplx_path,
                         model_type='smplx',
                         gender=test_data['gender'],
                         use_pca=False,
                         # num_pca_comps=test_data['n_comps'],
                         flat_hand_mean=True,
                         v_template=sbj_vtemp,
                         batch_size=1)

    verts_sbj = {}
    verts_sbj['predicted'] = construct_sbj_verts(sbj_m, fullpose.float(), results['body_transl'], object_transl)
    verts_sbj['gt'] = construct_sbj_verts(sbj_m, torch.FloatTensor(test_data['body']['params']['fullpose'].reshape(1, -1)),
                                    torch.FloatTensor(test_data['body']['params']['transl']), object_transl)

    obj_mesh = os.path.join(cfg.tool_meshes, test_data['object']['object_mesh'])
    obj_mesh = Mesh(filename=obj_mesh)
    obj_vtemp = np.array(obj_mesh.vertices)
    obj_m = ObjectModel(v_template=obj_vtemp,
                        batch_size=1)
    test_data['object']['params']['transl'] += to_cpu(object_transl)
    obj_parms = params2torch(test_data['object']['params'])
    verts_obj = to_cpu(obj_m(**obj_parms).vertices)

    gnet_optim = GNetOptim(sbj_m, obj_m, cfg)
    optim_results = gnet_optim.fitting(results, obj_parms)
    verts_sbj['optim'] = construct_sbj_verts(sbj_m, optim_results['fullpose'].reshape(1, -1),
                                             optim_results['transl'], object_transl)

    if not os.path.exists(os.path.join(cfg.renderings, test_data['sbj_id'])):
        os.makedirs(os.path.join(cfg.renderings, test_data['sbj_id']))

    for k in verts_sbj.keys():
        s_mesh = Mesh(vertices=verts_sbj[k][0], faces=sbj_m.faces, vc=colors['pink'], smooth=True)
        o_mesh = Mesh(vertices=verts_obj[0], faces=obj_mesh.faces, vc=colors['yellow'])

        mv.set_static_meshes([o_mesh, s_mesh])

        color, depth = mv.viewer.render(mv.scene)
        img = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        img_save_path = os.path.join(cfg.renderings, test_data['sbj_id'])
        img_name = (cfg.data_path.split('/')[-1]).split('.')[-2] + '_' + k + '.jpg'
        cv2.imwrite(os.path.join(img_save_path, img_name), img)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Render GNet Poses')

    parser.add_argument('--model-path', required=True, type=str,
                        help='Path to the saved GNet model')
    parser.add_argument('--data-path', default='datasets_gnet_test/grab/s8/cup_drink_1_frame267.npz', type=str,
                        help='Path to the test data file')
    parser.add_argument('--renderings', default='renderings_optim', type=str,
                        help='Path to the directory saving the renderings')

    args = parser.parse_args()

    model_path = args.model_path
    data_path = args.data_path
    renderings = args.renderings

    cfg = {
        'tool_meshes': 'toolMeshes',
        'smplx_path': 'smplx_models',
        'model_path': model_path,
        'data_path': data_path,
        'renderings': renderings,
        'lr': 5e-4,
        'rh_idx': 'smplx_body_parts_correspondences/rhand_99_indices.npz',
        'n_iters': 100,
    }

    cfg = Config(**cfg)

    if not os.path.exists(cfg.renderings):
        os.makedirs(cfg.renderings)

    render_img(cfg)