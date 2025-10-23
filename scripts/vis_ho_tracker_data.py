import argparse
import glob
import os
import pickle

import numpy as np
import open3d as o3d
import torch
from manotorch.manolayer import ManoLayer


class HandMesh:
    def __init__(self, parent, idx, side):
        self.parent = parent
        self.side = side
        self.data = self._load(parent, idx, side)
        self.t = 0

        mano_layer = getattr(parent, f"mano_layer_{side}")
        faces = mano_layer.get_mano_closed_faces()

        pose = torch.tensor(self.data["mano_pose"])
        betas = torch.tensor(self.data["mano_betas"])
        tsl = self.data["mano_tsl"]
        self.d_verts = (
            mano_layer(pose.reshape(-1, 16 * 3), betas).verts.numpy() + tsl[:, None]
        )

        self.mesh = o3d.geometry.TriangleMesh()
        self.mesh.vertices = o3d.utility.Vector3dVector(self.d_verts[self.t])
        self.mesh.triangles = o3d.utility.Vector3iVector(faces)
        if self.side == "right":
            self.mesh.paint_uniform_color([0.55, 0.78, 0.78])
        else:
            self.mesh.paint_uniform_color([0.31, 0.55, 0.78])
        self.mesh.compute_vertex_normals()

    def _load(self, parent, idx, side):
        pkl_path = os.path.join(parent.demo_path, parent.dtype, idx, f"{side}_hand.pkl")
        with open(pkl_path, "rb") as f:
            return pickle.load(f)

    def update(self):
        self.t = (self.t + 1) % len(self.d_verts)
        self.mesh.vertices = o3d.utility.Vector3dVector(self.d_verts[self.t])
        self.mesh.compute_vertex_normals()
        return self.mesh


class ObjMesh:
    def __init__(self, parent, idx, side=None):  # None for h2o1
        self.parent = parent
        self.side = side
        self.data = self._load(parent, idx, side)
        self.t = 0

        self.mesh = self._load_mesh(parent, idx, side)
        self.mesh.paint_uniform_color([1.0, 0.42, 0.04])
        self.mesh.transform(self.data["obj_trajectory"][self.t])

    def _load(self, parent, idx, side):
        if side is None:
            pkl_path = os.path.join(parent.demo_path, parent.dtype, idx, "obj.pkl")
        else:
            pkl_path = os.path.join(
                parent.demo_path, parent.dtype, idx, f"{side}_obj.pkl"
            )
        with open(pkl_path, "rb") as f:
            return pickle.load(f)

    def _load_mesh(self, parent, idx, side):
        if side is None:
            base = os.path.join(parent.demo_path, parent.dtype, idx, "urdf")
        else:
            base = os.path.join(parent.demo_path, parent.dtype, idx, f"{side}_urdf")
        mesh_path = (
            glob.glob(os.path.join(base, "*.ply"))
            or glob.glob(os.path.join(base, "*.obj"))
        )[0]
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        mesh.compute_vertex_normals()
        return mesh

    def update(self):
        cur = self.data["obj_trajectory"][self.t]
        self.t = (self.t + 1) % len(self.data["obj_trajectory"])
        self.mesh.transform(self.data["obj_trajectory"][self.t] @ np.linalg.inv(cur))
        return self.mesh


class VisDemo:
    def __init__(self, demo_path, dtype, hand="both"):
        self.demo_path = demo_path
        self.dtype = dtype
        self.hand = hand
        assert dtype in ["h1o1", "h2o1", "h2o2"], f"Unsupported dtype: {dtype}"

        all_demos = sorted(os.listdir(os.path.join(demo_path, dtype)))

        self.demo_list = []
        for idx in all_demos:
            if (
                self.dtype in ["h2o1", "h2o2"]
                or self.hand == "both"
                or self._exist_hand_file(idx, self.hand)
            ):
                self.demo_list.append(idx)

        if not self.demo_list:
            raise ValueError("No valid demos found in the specified directory")

        self.mano_layer_right = ManoLayer(
            flat_hand_mean=(self.dtype != "h2o1"),
            side="right",
            center_idx=0,
            mano_assets_root="data/mano_v1_2",
        )
        self.mano_layer_left = ManoLayer(
            flat_hand_mean=(self.dtype != "h2o1"),
            side="left",
            center_idx=0,
            mano_assets_root="data/mano_v1_2",
        )

    def _exist_hand_file(self, idx, side):
        return os.path.exists(
            os.path.join(self.demo_path, self.dtype, idx, f"{side}_hand.pkl")
        )

    def visualize(self, idx):
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Demo", width=720, height=720)
        vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))

        meshes = []
        for side in ["right", "left", None]:
            if side and self._exist_hand_file(idx, side):
                meshes.append(HandMesh(self, idx, side))

            # NOTE: objects are stored differently depending on dtype
            # So, just try everything within try-except.
            try:
                meshes.append(ObjMesh(self, idx, side))
            except FileNotFoundError:
                pass

        for mesh in meshes:
            vis.add_geometry(mesh.mesh)

        while True:
            should_close = not vis.poll_events()
            if should_close:
                break
            for mesh in meshes:
                # mesh instances have trajectory, and update() shows the next pose
                vis.update_geometry(mesh.update())
            vis.update_renderer()

        vis.destroy_window()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize HO-Tracker data.")
    parser.add_argument(
        "--demo_path",
        type=str,
        default="data/HO-Tracker/data/train_sample",
        help="Path to the demonstration data",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="h1o1",
        choices=["h1o1", "h2o1", "h2o2"],
        help="Type of demonstration",
    )
    parser.add_argument(
        "--hand",
        type=str,
        default="right",
        choices=["right", "left", "both"],
        help="Filter demo by hand in h1o1. Ignored in h2o1 and h2o2.",
    )
    parser.add_argument(
        "--demo_idx",
        type=str,
        default=None,
        help="Specific demo index to visualize. If not provided, all demos will be visualized.",
    )
    args = parser.parse_args()

    vis_demo = VisDemo(args.demo_path, args.dtype, args.hand)

    if args.demo_idx:
        if args.demo_idx in vis_demo.demo_list:
            print(f"Visualizing demo: {args.demo_idx}")
            vis_demo.visualize(args.demo_idx)
        else:
            print(
                f"Error: Demo index '{args.demo_idx}' not found in {os.path.join(args.demo_path, args.dtype)}"
            )
            print(f"Available demos: {vis_demo.demo_list}")
    else:
        for idx in vis_demo.demo_list:
            print(f"Visualizing demo: {idx}")
            vis_demo.visualize(idx)
