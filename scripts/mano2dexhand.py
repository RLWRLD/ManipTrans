# https://github.com/KailinLi/HO-Tracker-Baseline/blob/main/main/dataset/mano2dexhand.py

import math
import os
import pickle
import logging

from isaacgym import gymapi, gymutil
import gymtorch

logging.getLogger("gymapi").setLevel(logging.CRITICAL)
logging.getLogger("gymtorch").setLevel(logging.CRITICAL)
logging.getLogger("gymutil").setLevel(logging.CRITICAL)

import numpy as np
import pytorch_kinematics as pk
import torch
from termcolor import cprint

from main.dataset.factory import ManipDataFactory
from main.dataset.transform import (
    aa_to_quat,
    aa_to_rotmat,
    quat_to_rotmat,
    rot6d_to_aa,
    rot6d_to_quat,
    rot6d_to_rotmat,
    rotmat_to_aa,
    rotmat_to_quat,
    rotmat_to_rot6d,
)
from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory


def pack_data(data, dexhand):
    packed_data = {}
    for k in data[0].keys():
        if k == "mano_joints":
            mano_joints = []
            for d in data:
                mano_joints.append(
                    torch.concat(
                        [
                            d[k][dexhand.to_hand(j_name)[0]]
                            for j_name in dexhand.body_names
                            if dexhand.to_hand(j_name)[0] != "wrist"
                        ],
                        dim=-1,
                    )
                )
            packed_data[k] = torch.stack(mano_joints).squeeze()
        elif type(data[0][k]) == torch.Tensor:
            packed_data[k] = torch.stack([d[k] for d in data]).squeeze()
        elif type(data[0][k]) == np.ndarray:
            packed_data[k] = np.stack([d[k] for d in data]).squeeze()
        else:
            packed_data[k] = [d[k] for d in data]
    return packed_data


def soft_clamp(x, lower, upper):
    return lower + torch.sigmoid(4 / (upper - lower) * (x - (lower + upper) / 2)) * (
        upper - lower
    )


class Mano2Dexhand:
    def __init__(self, args, dexhand, obj_urdf_path):
        self.gym = gymapi.acquire_gym()
        self.sim_params = gymapi.SimParams()
        self.dexhand = dexhand

        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

        self.headless = args.headless
        if self.headless:
            self.graphics_device_id = -1

        assert args.physics_engine == gymapi.SIM_PHYSX

        self.sim_params.substeps = 1
        self.sim_params.physx.solver_type = 1
        self.sim_params.physx.num_position_iterations = 4
        self.sim_params.physx.num_velocity_iterations = 1
        self.sim_params.physx.num_threads = args.num_threads
        self.sim_params.physx.use_gpu = args.use_gpu
        self.sim_params.physx.max_gpu_contact_pairs = int(2 * 4194304)

        self.sim_params.use_gpu_pipeline = args.use_gpu_pipeline
        self.sim_device = args.sim_device if args.use_gpu_pipeline else "cpu"

        self.sim = self.gym.create_sim(
            args.compute_device_id,
            args.graphics_device_id,
            args.physics_engine,
            self.sim_params,
        )

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)
        if not self.headless:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())

        asset_root = os.path.split(self.dexhand.urdf_path)[0]
        asset_file = os.path.split(self.dexhand.urdf_path)[1]

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        # asset_options.use_mesh_materials = True
        dexhand_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options
        )

        self.chain = pk.build_chain_from_urdf(
            open(os.path.join(asset_root, asset_file)).read()
        )
        self.chain = self.chain.to(dtype=torch.float32, device=self.sim_device)

        dexhand_dof_stiffness = torch.tensor(
            [10] * self.dexhand.n_dofs,
            dtype=torch.float,
            device=self.sim_device,
        )
        dexhand_dof_damping = torch.tensor(
            [1] * self.dexhand.n_dofs,
            dtype=torch.float,
            device=self.sim_device,
        )
        self.limit_info = {}
        asset_rh_dof_props = self.gym.get_asset_dof_properties(dexhand_asset)
        self.limit_info["rh"] = {
            "lower": np.asarray(asset_rh_dof_props["lower"]).copy().astype(np.float32),
            "upper": np.asarray(asset_rh_dof_props["upper"]).copy().astype(np.float32),
        }

        self.num_dexhand_bodies = self.gym.get_asset_rigid_body_count(dexhand_asset)
        self.num_dexhand_dofs = self.gym.get_asset_dof_count(dexhand_asset)

        dexhand_dof_props = self.gym.get_asset_dof_properties(dexhand_asset)
        rigid_shape_rh_props_asset = self.gym.get_asset_rigid_shape_properties(
            dexhand_asset
        )
        for element in rigid_shape_rh_props_asset:
            element.friction = 0.0001
            element.rolling_friction = 0.0001
            element.torsion_friction = 0.0001
        self.gym.set_asset_rigid_shape_properties(
            dexhand_asset, rigid_shape_rh_props_asset
        )

        self.dexhand_dof_lower_limits = []
        self.dexhand_dof_upper_limits = []
        self._dexhand_effort_limits = []
        self._dexhand_dof_speed_limits = []
        for i in range(self.num_dexhand_dofs):
            dexhand_dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            dexhand_dof_props["stiffness"][i] = dexhand_dof_stiffness[i]
            dexhand_dof_props["damping"][i] = dexhand_dof_damping[i]

            self.dexhand_dof_lower_limits.append(dexhand_dof_props["lower"][i])
            self.dexhand_dof_upper_limits.append(dexhand_dof_props["upper"][i])
            self._dexhand_effort_limits.append(dexhand_dof_props["effort"][i])
            self._dexhand_dof_speed_limits.append(dexhand_dof_props["velocity"][i])

        self.dexhand_dof_lower_limits = torch.tensor(
            self.dexhand_dof_lower_limits, device=self.sim_device
        )
        self.dexhand_dof_upper_limits = torch.tensor(
            self.dexhand_dof_upper_limits, device=self.sim_device
        )
        self._dexhand_effort_limits = torch.tensor(
            self._dexhand_effort_limits, device=self.sim_device
        )
        self._dexhand_dof_speed_limits = torch.tensor(
            self._dexhand_dof_speed_limits, device=self.sim_device
        )
        default_dof_state = np.ones(self.num_dexhand_dofs, gymapi.DofState.dtype)
        default_dof_state["pos"] *= np.pi / 50
        default_dof_state["pos"][8] = 0.8
        default_dof_state["pos"][9] = 0.05
        self.dexhand_default_dof_pos = default_dof_state
        self.dexhand_default_pose = gymapi.Transform()
        self.dexhand_default_pose.p = gymapi.Vec3(0, 0, 0)
        self.dexhand_default_pose.r = gymapi.Quat(0, 0, 0, 1)

        table_width_offset = 0.2
        mujoco2gym_transf = np.eye(4)
        mujoco2gym_transf[:3, :3] = aa_to_rotmat(
            np.array([0, 0, -np.pi / 2])
        ) @ aa_to_rotmat(np.array([np.pi / 2, 0, 0]))
        table_pos = gymapi.Vec3(-table_width_offset / 2, 0, 0.4)
        self.dexhand_pose = gymapi.Transform()
        table_half_height = 0.015
        self._table_surface_z = table_pos.z + table_half_height
        mujoco2gym_transf[:3, 3] = np.array([0, 0, self._table_surface_z])
        self.mujoco2gym_transf = torch.tensor(
            mujoco2gym_transf, device=self.sim_device, dtype=torch.float32
        )

        self.num_envs = args.num_envs
        num_per_row = int(math.sqrt(self.num_envs))
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_options = gymapi.AssetOptions()
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        asset_options.thickness = 0.001
        asset_options.fix_base_link = True
        asset_options.vhacd_enabled = False
        asset_options.disable_gravity = True
        asset_options.density = 200

        current_asset = self.gym.load_asset(
            self.sim, *os.path.split(obj_urdf_path), asset_options
        )

        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(
            current_asset
        )
        for element in rigid_shape_props_asset:
            element.friction = 0.00001
        self.gym.set_asset_rigid_shape_properties(
            current_asset, rigid_shape_props_asset
        )

        self.envs = []
        self.hand_idxs = []

        for i in range(self.num_envs):
            # Create env
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env)
            dexhand_actor = self.gym.create_actor(
                env,
                dexhand_asset,
                self.dexhand_default_pose,
                "dexhand",
                i,
                (1 if self.dexhand.self_collision else 0),
            )

            # Set initial DOF states
            self.gym.set_actor_dof_states(
                env, dexhand_actor, self.dexhand_default_dof_pos, gymapi.STATE_ALL
            )

            # Set DOF control properties
            self.gym.set_actor_dof_properties(env, dexhand_actor, dexhand_dof_props)

            self.obj_actor = self.gym.create_actor(
                env, current_asset, gymapi.Transform(), "manip_obj", i, 0
            )

            scene_asset_options = gymapi.AssetOptions()
            scene_asset_options.fix_base_link = True
            for joint_vis_id, joint_name in enumerate(self.dexhand.body_names):
                joint_name = self.dexhand.to_hand(joint_name)[0]
                joint_point = self.gym.create_sphere(
                    self.sim, 0.005, scene_asset_options
                )
                a = self.gym.create_actor(
                    env,
                    joint_point,
                    gymapi.Transform(),
                    f"mano_joint_{joint_vis_id}",
                    self.num_envs + 1,
                    0b1,
                )
                if "index" in joint_name:
                    inter_c = 70
                elif "middle" in joint_name:
                    inter_c = 130
                elif "ring" in joint_name:
                    inter_c = 190
                elif "pinky" in joint_name:
                    inter_c = 250
                elif "thumb" in joint_name:
                    inter_c = 10
                else:
                    inter_c = 0
                if "tip" in joint_name:
                    c = gymapi.Vec3(inter_c / 255, 200 / 255, 200 / 255)
                elif "proximal" in joint_name:
                    c = gymapi.Vec3(200 / 255, inter_c / 255, 200 / 255)
                elif "intermediate" in joint_name:
                    c = gymapi.Vec3(200 / 255, 200 / 255, inter_c / 255)
                else:
                    c = gymapi.Vec3(100 / 255, 150 / 255, 200 / 255)
                self.gym.set_rigid_body_color(env, a, 0, gymapi.MESH_VISUAL, c)

        env_ptr = self.envs[0]
        dexhand_handle = 0
        self.dexhand_handles = {
            k: self.gym.find_actor_rigid_body_handle(env_ptr, dexhand_handle, k)
            for k in self.dexhand.body_names
        }
        self.dexhand_dof_handles = {
            k: self.gym.find_actor_dof_handle(env_ptr, dexhand_handle, k)
            for k in self.dexhand.dof_names
        }
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        _net_cf = self.gym.acquire_net_contact_force_tensor(self.sim)

        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(
            self.num_envs, -1, 13
        )
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(
            self.num_envs, -1, 2
        )
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(
            self.num_envs, -1, 13
        )
        self._net_cf = gymtorch.wrap_tensor(_net_cf).view(self.num_envs, -1, 3)
        self.q = self._dof_state[..., 0]
        self.qd = self._dof_state[..., 1]
        self._base_state = self._root_state[:, 0, :]

        self.isaac2chain_order = [
            self.gym.get_actor_dof_names(env_ptr, dexhand_handle).index(j)
            for j in self.chain.get_joint_parameter_names()
        ]

        self.mano_joint_points = [
            self._root_state[
                :, self.gym.find_actor_handle(env_ptr, f"mano_joint_{i}"), :
            ]
            for i in range(len(self.dexhand.body_names))
        ]

        if not self.headless:
            cam_pos = gymapi.Vec3(4, 3, 3)
            cam_target = gymapi.Vec3(-4, -3, 0)
            middle_env = self.envs[self.num_envs // 2 + num_per_row // 2]
            self.gym.viewer_camera_look_at(self.viewer, middle_env, cam_pos, cam_target)

        self.gym.prepare_sim(self.sim)

    def set_force_vis(self, env_ptr, part_k, has_force):
        self.gym.set_rigid_body_color(
            env_ptr,
            0,
            self.dexhand_handles[part_k],
            gymapi.MESH_VISUAL,
            (
                gymapi.Vec3(
                    1.0,
                    0.6,
                    0.6,
                )
                if has_force
                else gymapi.Vec3(1.0, 1.0, 1.0)
            ),
        )

    def fitting(
        self,
        max_iter,
        obj_trajectory,
        target_wrist_pos,
        target_wrist_rot,
        target_mano_joints,
    ):
        assert target_mano_joints.shape[0] == self.num_envs
        target_wrist_pos = (
            self.mujoco2gym_transf[:3, :3] @ target_wrist_pos.T
        ).T + self.mujoco2gym_transf[:3, 3]
        target_wrist_rot = self.mujoco2gym_transf[:3, :3] @ aa_to_rotmat(
            target_wrist_rot
        )
        target_mano_joints = target_mano_joints.view(-1, 3)
        target_mano_joints = (
            self.mujoco2gym_transf[:3, :3] @ target_mano_joints.T
        ).T + self.mujoco2gym_transf[:3, 3]
        target_mano_joints = target_mano_joints.view(self.num_envs, -1, 3)

        obj_trajectory = self.mujoco2gym_transf @ obj_trajectory

        middle_pos = (target_mano_joints[:, 3] + target_wrist_pos) / 2
        obj_pos = obj_trajectory[:, :3, 3]
        offset = middle_pos - obj_pos
        offset = offset / torch.norm(offset, dim=-1, keepdim=True) * 0.2

        opt_wrist_pos = torch.tensor(
            target_wrist_pos + offset,
            device=self.sim_device,
            dtype=torch.float32,
            requires_grad=True,
        )
        opt_wrist_rot = torch.tensor(
            rotmat_to_rot6d(target_wrist_rot),
            device=self.sim_device,
            dtype=torch.float32,
            requires_grad=True,
        )
        opt_dof_pos = torch.tensor(
            self.dexhand_default_dof_pos["pos"][None].repeat(self.num_envs, axis=0),
            device=self.sim_device,
            dtype=torch.float32,
            requires_grad=True,
        )
        opti = torch.optim.Adam(
            [
                {"params": [opt_wrist_pos, opt_wrist_rot], "lr": 0.002},
                {"params": [opt_dof_pos], "lr": 0.001},
            ]
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opti, T_0=1000)
        
        weight = []
        for k in self.dexhand.body_names:
            k = self.dexhand.to_hand(k)[0]
            if "tip" in k:
                if "index" in k:
                    weight.append(20)
                elif "middle" in k:
                    weight.append(10)
                elif "ring" in k:
                    weight.append(7)
                elif "pinky" in k:
                    weight.append(5)
                elif "thumb" in k:
                    weight.append(25)
                else:
                    raise ValueError
            elif "proximal" in k:
                weight.append(1)
            elif "intermediate" in k:
                weight.append(1)
            else:
                weight.append(1)
        weight = torch.tensor(weight, device=self.sim_device, dtype=torch.float32)
        iter = 0
        best_loss = 1e10
        no_improve_iters = 0
        early_stop_patience = 500  # Stop if no improvement for 200 iterations
        while (self.headless and iter < max_iter) or (
            not self.headless and not self.gym.query_viewer_has_closed(self.viewer)
        ):
            iter += 1

            opt_wrist_quat = rot6d_to_quat(opt_wrist_rot)[:, [1, 2, 3, 0]]
            opt_wrist_rotmat = rot6d_to_rotmat(opt_wrist_rot)
            self._root_state[:, 0, :3] = opt_wrist_pos.detach()
            self._root_state[:, 0, 3:7] = opt_wrist_quat.detach()
            self._root_state[:, 0, 7:] = torch.zeros_like(self._root_state[:, 0, 7:])
            self._root_state[:, self.obj_actor, :3] = obj_trajectory[:, :3, 3]
            self._root_state[:, self.obj_actor, 3:7] = rotmat_to_quat(
                obj_trajectory[:, :3, :3]
            )[:, [1, 2, 3, 0]]

            opt_dof_pos_clamped = torch.clamp(
                opt_dof_pos,
                self.dexhand_dof_lower_limits,
                self.dexhand_dof_upper_limits,
            )

            self.gym.set_dof_position_target_tensor(
                self.sim, gymtorch.unwrap_tensor(opt_dof_pos_clamped)
            )
            self.gym.set_actor_root_state_tensor(
                self.sim, gymtorch.unwrap_tensor(self._root_state)
            )

            # Step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            if not self.headless:
                self.gym.step_graphics(self.sim)

            # Update jacobian and mass matrix
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_net_contact_force_tensor(self.sim)
            # Step rendering
            if not self.headless:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, False)
            self.gym.sync_frame_time(self.sim)

            ret = self.chain.forward_kinematics(
                opt_dof_pos_clamped[:, self.isaac2chain_order]
            )
            pk_joints = torch.stack(
                [ret[k].get_matrix()[:, :3, 3] for k in self.dexhand.body_names], dim=1
            )
            pk_joints = (
                rot6d_to_rotmat(opt_wrist_rot) @ pk_joints.transpose(-1, -2)
            ).transpose(-1, -2) + opt_wrist_pos[:, None]

            target_joints = torch.cat(
                [target_wrist_pos[:, None], target_mano_joints], dim=1
            )
            for k in range(len(self.mano_joint_points)):
                self.mano_joint_points[k][:, :3] = target_joints[:, k]

            # Main loss: distance between current and target joint positions
            loss_dist = torch.mean(
                torch.norm(pk_joints - target_joints, dim=-1) * weight[None]
            )

            # --- Contact-based Penalties ---
            # Calculate contact forces on the hand bodies
            hand_contact_forces = torch.norm(self._net_cf[:, : self.num_dexhand_bodies], dim=-1)

            # Penalty for excessive force. The goal here is to generate plausible source traj.
            loss_contact = torch.mean(hand_contact_forces)

            # --- Temporal Regularization ---
            # Penalize high velocity (difference between consecutive frames)
            vel_wrist_pos_loss = torch.mean(
                torch.square(opt_wrist_pos[1:] - opt_wrist_pos[:-1])
            )
            vel_wrist_rot_loss = torch.mean(
                torch.square(opt_wrist_rot[1:] - opt_wrist_rot[:-1])
            )
            vel_dof_loss = torch.mean(
                torch.square(opt_dof_pos_clamped[1:] - opt_dof_pos_clamped[:-1])
            )
            loss_temporal_vel = vel_wrist_pos_loss + vel_wrist_rot_loss + vel_dof_loss

            # Penalize high acceleration (difference between 2-back frames)
            acc_wrist_pos_loss = torch.mean(
                torch.square(opt_wrist_pos[2:] - opt_wrist_pos[:-2])
            )
            acc_wrist_rot_loss = torch.mean(
                torch.square(opt_wrist_rot[2:] - opt_wrist_rot[:-2])
            )
            acc_dof_loss = torch.mean(
                torch.square(opt_dof_pos_clamped[2:] - opt_dof_pos_clamped[:-2])
            )
            loss_temporal_acc = acc_wrist_pos_loss + acc_wrist_rot_loss + acc_dof_loss

            # Combine losses with weights
            weight_contact = 0.01
            weight_temp_vel = 20.0
            weight_temp_acc = 10.0  # Acceleration penalty is usually smaller
            loss = (
                loss_dist
                + weight_contact * loss_contact
                + weight_temp_vel * loss_temporal_vel
                + weight_temp_acc * loss_temporal_acc
            )

            # Update opt_wrist_pos/rot and opt_dof_pos
            opti.zero_grad()
            loss.backward()
            opti.step()
            scheduler.step()

            # Stopping only based on dist loss
            if loss_dist.item() < best_loss - 1e-5:
                best_loss = loss_dist.item()
                no_improve_iters = 0
            else:
                no_improve_iters += 1

            if iter % 100 == 0:
                # Updated print statement to show all loss components
                cprint(
                    f"{iter} | Total: {loss.item():.5f}"
                    f" | Dist: {loss_dist.item():.5f}"
                    f" | Con: {loss_contact.item() * weight_contact:.5f}"
                    f" | Vel: {loss_temporal_vel.item() * weight_temp_vel:.5f}"
                    f" | Acc: {loss_temporal_acc.item() * weight_temp_acc:.5f}"
                    f" | LR: {scheduler.get_last_lr()[0]:.2e}",
                    "green",
                )
            if no_improve_iters >= early_stop_patience:
                cprint(f"Early stopping at iteration {iter} due to no improvement.", "yellow")
                break

        isaac_joints = torch.stack(
            [
                self._rigid_body_state[:, self.dexhand_handles[k], :3]
                for k in self.dexhand.body_names
            ],
            dim=1,
        )

        to_dump = {
            "opt_wrist_pos": opt_wrist_pos.detach().cpu().numpy(),
            "opt_wrist_rot": rot6d_to_aa(opt_wrist_rot).detach().cpu().numpy(),
            "opt_dof_pos": opt_dof_pos_clamped.detach().cpu().numpy(),
            "opt_joints_pos": isaac_joints.detach().cpu().numpy(),
        }

        if not self.headless:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
        return to_dump


if __name__ == "__main__":
    _parser = gymutil.parse_arguments(
        description="Mano to Dexhand",
        headless=True,
        custom_parameters=[
            {
                "name": "--iter",
                "type": int,
                "default": 4000,
            },
            {
                "name": "--data_idx",
                "type": str,
                # "default": "0009",
                "default": "03ac9@0",
            },
            {
                "name": "--dexhand",
                "type": str,
                "default": "inspire",
            },
            {
                "name": "--side",
                "type": str,
                "default": "right",
            },
        ],
    )

    """
    === dtype: h1o1, right hand only ===
    03ac9@0, 05aa2@0, 05d45@0, 07bb1@1, 0fb0e@2, 15642@2, 178ef@1, 1be0e@1,
    20b58@0, 235ac@0, 2979c@0, 2f774@1, 33369@1, 3ae08@10, 3b1e6@10, 60694@2,
    662b0@5, 667dd@1, 6ca00@15, 82851@12, 83207@2, 86fb0@16, 8e463@3, 925aa@5,
    963fd@0, 9f9ee@0, ad924@0, d4ddf@2, d77d9@2, de088@1, e7816@1, e9343@2,
    e9aab@1, ea24d@4, eb877@0
    """

    dexhand = DexHandFactory.create_hand(_parser.dexhand, _parser.side)

    def run(parser, idx):
        dataset_type = ManipDataFactory.dataset_type(idx)
        demo_d = ManipDataFactory.create_data(
            manipdata_type=dataset_type,
            side=parser.side,
            device="cuda:0",
            mujoco2gym_transf=torch.eye(4, device="cuda:0"),
            dexhand=dexhand,
            verbose=False,
        )

        demo_data = pack_data([demo_d[idx]], dexhand)

        parser.num_envs = demo_data["mano_joints"].shape[0]

        mano2inspire = Mano2Dexhand(parser, dexhand, demo_data["obj_urdf_path"][0])

        to_dump = mano2inspire.fitting(
            parser.iter,
            demo_data["obj_trajectory"],
            demo_data["wrist_pos"],
            demo_data["wrist_rot"],
            demo_data["mano_joints"].view(parser.num_envs, -1, 3),
        )

        if dataset_type == "hotracker":
            dump_path = f"data/retargeting/HO-Tracker/mano2{str(dexhand)}/{demo_data['data_path'][0].split('data/')[-1]}/opt.pkl"
        else:
            raise ValueError("Unsupported dataset type")

        os.makedirs(os.path.dirname(dump_path), exist_ok=True)
        with open(dump_path, "wb") as f:
            pickle.dump(to_dump, f)

    run(_parser, _parser.data_idx)
