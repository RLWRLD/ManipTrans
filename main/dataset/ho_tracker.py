import json
import os
import pickle
from functools import lru_cache

import numpy as np
import smplx
import torch
import trimesh
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from smplx.lbs import batch_rigid_transform, batch_rodrigues
from termcolor import cprint
from torch.utils.data import Dataset
from typing import List
from main.dataset.oakink2_layer.smplx import SMPLXLayer
from main.dataset.transform import aa_to_rotmat, caculate_align_mat, rotmat_to_aa
from .base import ManipData
from .oakink2_dataset_utils import load_obj_map, as_mesh
from .decorators import register_manipdata
import glob


def recursive_to(device, data):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(device)
    elif isinstance(data, dict):
        return {k: recursive_to(device, v) for k, v in data.items()}
    elif isinstance(data, list):
        return [recursive_to(device, v) for v in data]
    elif isinstance(data, tuple):
        return tuple(recursive_to(device, v) for v in data)
    else:
        return data


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
                assert data_type in ["h1o1", "h2o1", "h2o2"], f"data_type {data_type} not recognized"
                for data_item in os.listdir(os.path.join(data_dir, "data", f"{sp}_sample", data_type)):
                    pathes[data_item] = {
                        "data_type": data_type,
                        "path": os.path.join(data_dir, "data", f"{sp}_sample", data_type, data_item),
                    }

        self.data_pathes = pathes
        self.dataset_fps = 60  # dataset already been downsampled to 60Hz

    @lru_cache(maxsize=None)
    def __getitem__(self, index):

        if type(index) == str:
            index = (index.split("@")[0], int(index.split("@")[1]))

        assert (
            type(index) == tuple and len(index) == 2 and type(index[0]) == str and type(index[1]) == int
        ), "index error"

        tracking_data_path = self.data_pathes[f"{index[0]}@{index[1]}"]["path"]

        hand_data = pickle.load(open(os.path.join(tracking_data_path, f"{self.side}_hand.pkl"), "rb"))
        obj_data = pickle.load(open(os.path.join(tracking_data_path, f"{self.side}_obj.pkl"), "rb"))

        obj_path = (
            glob.glob(os.path.join(tracking_data_path, f"{self.side}_urdf", "*.ply"))
            + glob.glob(os.path.join(tracking_data_path, f"{self.side}_urdf", "*.obj"))
        )[0]
        obj_mesh = trimesh.load(obj_path, process=False, force="mesh")
        mesh = Meshes(
            verts=torch.from_numpy(obj_mesh.vertices[None, ...].astype(np.float32)),
            faces=torch.from_numpy(obj_mesh.faces[None, ...].astype(np.float32)),
        )
        rs_verts_obj = self.random_sampling_pc(mesh)

        data = {
            "data_path": tracking_data_path,
            "obj_id": f"{self.side}_{os.path.split(obj_path)[-1].split('.')[0]}",
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
        wrist_pos = data["wrist_pos"] - (middle_pos - data["wrist_pos"]) * 0.25  # ? hack for wrist position
        wrist_pos += self.dexhand.relative_translation
        data["wrist_pos"] = wrist_pos

        data = recursive_to(self.device, data)

        self.process_data(data, f"{index[0]}@{index[1]}", rs_verts_obj)

        # todo load retargeted data
        OPT_DEXHAND_PATH = f"data/retargeting/HO-Tracker/mano2{str(self.dexhand)}"
        opt_path = os.path.join(
            OPT_DEXHAND_PATH,
            (tracking_data_path.split("data/"))[-1],
            f"opt.pkl",
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
