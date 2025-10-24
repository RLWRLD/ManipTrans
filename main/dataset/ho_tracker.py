import glob
import os
import pickle
from functools import lru_cache

import numpy as np
import torch
import trimesh
from manotorch.manolayer import ManoLayer
from pytorch3d.structures import Meshes

from main.dataset.transform import aa_to_rotmat, rotmat_to_aa
from .base import ManipData
from .decorators import register_manipdata


def recursive_to(device, data):
    if isinstance(data, torch.Tensor):
        return data.float().to(device)
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data).float().to(device)
    elif isinstance(data, dict):
        return {k: recursive_to(device, v) for k, v in data.items()}
    elif isinstance(data, list):
        return [recursive_to(device, v) for v in data]
    elif isinstance(data, tuple):
        return tuple(recursive_to(device, v) for v in data)
    else:
        return data


def extract_info_from_mano(side, mano_layer, data):
    # Convert numpy inputs to torch tensors, ensuring they are on the same device as the model
    device = mano_layer.th_J_regressor.device
    pose = torch.tensor(data["mano_pose"], device=device, dtype=torch.float32)
    betas = torch.tensor(data["mano_betas"], device=device, dtype=torch.float32)
    tsl = torch.tensor(data["mano_tsl"], device=device, dtype=torch.float32)

    # Perform all calculations in PyTorch
    mano_output = mano_layer(pose.reshape(-1, 16 * 3), betas)

    # Apply translation to vertices and joints
    d_verts = mano_output.verts + tsl[:, None]
    mano_out_joints = mano_output.joints + tsl[:, None]

    # Wrist pose is the 0-th joint/transform
    wrist_pos = mano_out_joints[:, 0]
    wrist_rot = rotmat_to_aa(mano_output.transforms_abs[:, 0, :3, :3])

    # Define joints dictionary using the translated joints and vertices
    mano_joints = {
        "index_proximal": mano_out_joints[:, 1],
        "index_intermediate": mano_out_joints[:, 2],
        "index_distal": mano_out_joints[:, 3],
        "index_tip": d_verts[:, 353],  # reselect tip
        "middle_proximal": mano_out_joints[:, 4],
        "middle_intermediate": mano_out_joints[:, 5],
        "middle_distal": mano_out_joints[:, 6],
        "middle_tip": d_verts[:, 467],  # reselect tip
        "pinky_proximal": mano_out_joints[:, 7],
        "pinky_intermediate": mano_out_joints[:, 8],
        "pinky_distal": mano_out_joints[:, 9],
        "pinky_tip": d_verts[:, 695],  # reselect tip
        "ring_proximal": mano_out_joints[:, 10],
        "ring_intermediate": mano_out_joints[:, 11],
        "ring_distal": mano_out_joints[:, 12],
        "ring_tip": d_verts[:, 576],  # reselect tip
        "thumb_proximal": mano_out_joints[:, 13],
        "thumb_intermediate": mano_out_joints[:, 14],
        "thumb_distal": mano_out_joints[:, 15],
        "thumb_tip": d_verts[:, 766],  # reselect tip
    }

    return wrist_pos, wrist_rot, mano_joints


# @register_manipdata("hotracker_rh")
class HOTracker(ManipData):
    def __init__(
        self,
        *,
        data_dir: str = "data/HO-Tracker",
        split: str = "all",
        device="cuda:0",
        mujoco2gym_transf=None,
        max_seq_len=int(1e10),
        dexhand=None,
        **kwargs,
    ):
        super().__init__(
            data_dir=data_dir,
            split=split,
            skip=1,  # No need to skip for HO-Tracker, which is 60Hz
            device=device,
            mujoco2gym_transf=mujoco2gym_transf,
            max_seq_len=max_seq_len,
            dexhand=dexhand,
            **kwargs,
        )

        # load data pathes
        pathes = {}
        for sp in ["train", "test"] if split == "all" else [split]:
            for data_type in os.listdir(os.path.join(data_dir, "data", f"{sp}_sample")):
                assert data_type in ["h1o1", "h2o1", "h2o2"], (
                    f"data_type {data_type} not recognized"
                )
                for data_item in os.listdir(
                    os.path.join(data_dir, "data", f"{sp}_sample", data_type)
                ):
                    pathes[data_item] = {
                        "data_type": data_type,
                        "path": os.path.join(
                            data_dir, "data", f"{sp}_sample", data_type, data_item
                        ),
                    }

        self.data_pathes = pathes
        self.dataset_fps = 60  # dataset already been downsampled to 60Hz

    @lru_cache(maxsize=None)
    def __getitem__(self, data_key):
        tracking_data_path = self.data_pathes[data_key]["path"]

        hand_data = pickle.load(
            open(os.path.join(tracking_data_path, f"{self.side}_hand.pkl"), "rb")
        )

        # h2o1 data needs this
        if "mano_joints" not in hand_data:
            # NOTE: flat_hand_mean is false for h2o1
            mano_layer = ManoLayer(
                flat_hand_mean=False,
                side=self.side,
                center_idx=0,
                mano_assets_root="data/mano_v1_2",
            )
            wrist_pos, wrist_rot, mano_joints = extract_info_from_mano(
                self.side, mano_layer, hand_data
            )
            hand_data.update(
                {
                    "mano_joints": mano_joints,
                    "wrist_pos": wrist_pos,
                    "wrist_rot": wrist_rot,
                }
            )

        ### Objects are stored differently, depending on the source
        if "@" in data_key:
            # h1o1, h2o2
            obj_prefix = f"{self.side}_"
            mesh_dir = f"{self.side}_urdf"
        else:
            # h2o1
            obj_prefix = ""
            mesh_dir = "urdf"

        obj_data = pickle.load(
            open(os.path.join(tracking_data_path, f"{obj_prefix}obj.pkl"), "rb")
        )

        if isinstance(obj_data["obj_trajectory"], list):
            obj_data["obj_trajectory"] = np.stack(obj_data["obj_trajectory"], axis=0)

        obj_path = (
            glob.glob(os.path.join(tracking_data_path, mesh_dir, "*.ply"))
            + glob.glob(os.path.join(tracking_data_path, mesh_dir, "*.obj"))
        )[0]

        obj_mesh = trimesh.load(obj_path, process=False, force="mesh")
        mesh = Meshes(
            verts=torch.from_numpy(obj_mesh.vertices[None, ...].astype(np.float32)),
            faces=torch.from_numpy(obj_mesh.faces[None, ...].astype(np.float32)),
        )
        rs_verts_obj = self.random_sampling_pc(mesh)

        data = {
            "data_path": tracking_data_path,
            "obj_id": f"{obj_prefix}{os.path.split(obj_path)[-1].split('.')[0]}",
            "obj_mesh_path": obj_path,
            "obj_verts": rs_verts_obj,
            "obj_urdf_path": obj_path.replace(".obj", ".urdf").replace(".ply", ".urdf"),
            "scene_objs": [],
            **obj_data,
            **hand_data,
        }
        data["wrist_rot"] = aa_to_rotmat(data["wrist_rot"])

        # ? caculate the global hand wrist pose based on the dexhand local wrist pose
        data["wrist_rot"] = data["wrist_rot"] @ self.dexhand.relative_rotation
        middle_pos = data["mano_joints"]["middle_proximal"]
        wrist_pos = (
            data["wrist_pos"] - (middle_pos - data["wrist_pos"]) * 0.25
        )  # ? hack for wrist position
        wrist_pos += self.dexhand.relative_translation.astype(np.float32)
        data["wrist_pos"] = wrist_pos

        data = recursive_to(self.device, data)

        self.process_data(data, data_key, rs_verts_obj)

        # todo load retargeted data
        OPT_DEXHAND_PATH = f"data/retargeting/HO-Tracker/mano2{str(self.dexhand)}"
        opt_path = os.path.join(
            OPT_DEXHAND_PATH,
            (tracking_data_path.split("data/"))[-1],
            "opt.pkl",
        )

        self.load_retargeted_data(data, opt_path)

        return data


@register_manipdata("hotracker_rh")
class HOTrackerRH(HOTracker):
    def __init__(
        self,
        *,
        data_dir: str = "data/HO-Tracker",
        split: str = "all",
        device="cuda:0",
        mujoco2gym_transf=None,
        max_seq_len=int(1e10),
        dexhand=None,
        **kwargs,
    ):
        super().__init__(
            data_dir=data_dir,
            split=split,
            device=device,
            mujoco2gym_transf=mujoco2gym_transf,
            max_seq_len=max_seq_len,
            dexhand=dexhand,
            **kwargs,
        )
        self.side = "right"


@register_manipdata("hotracker_lh")
class HOTrackerLH(HOTracker):
    def __init__(
        self,
        *,
        data_dir: str = "data/HO-Tracker",
        split: str = "all",
        device="cuda:0",
        mujoco2gym_transf=None,
        max_seq_len=int(1e10),
        dexhand=None,
        **kwargs,
    ):
        super().__init__(
            data_dir=data_dir,
            split=split,
            device=device,
            mujoco2gym_transf=mujoco2gym_transf,
            max_seq_len=max_seq_len,
            dexhand=dexhand,
            **kwargs,
        )
        self.side = "left"
