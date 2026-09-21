from __future__ import annotations
import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

import omni.physics.tensors.impl.api as physx
from isaacsim.core.prims import XFormPrim

import isaaclab.utils.math as math_utils
from isaaclab.sensors.camera import CameraData
from isaaclab.utils.warp import raycast_mesh

from isaaclab.sensors.ray_caster import RayCaster

if TYPE_CHECKING:
    from .sharpa_vbts_cfg import SharpaVBTSCfg

import omni.usd as omni_usd
import isaaclab.sim as sim_utils
from pxr import Gf

# PhysX, get object views
import re
import omni.physics.tensors as physx_tensors
from typing import Sequence, Union

class SimView:
    def __init__(self, target_rigid_expr: str, num_envs: int, *, device: Union[str, torch.device] = None):
        """
        target_rigid_expr: like "/World/envs/env_.*/object" (assumes 'env_.*' segment)
        num_envs: total number of parallel envs
        """
        self.expr = target_rigid_expr
        self.num_envs = int(num_envs)
        self.device = torch.device(device) if device is not None else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # create SimulationView (torch frontend) and set roots under /World/envs/*
        self._sim_view = physx_tensors.create_simulation_view("torch")
        # self._sim_view.set_subspace_roots("/World/envs/*")
        self._sim_view.set_subspace_roots("/")

        # per-env cached RigidBodyView
        self._views: dict[int, object] = {}

        # output tensors (WXYZ quat)
        self.pos_w  = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.quat_w = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float32)

    def update(self, env_ids: Sequence[int]):
        """Create (if needed) per-env RigidBodyView, read world pose, and update pos_w / quat_w for given env_ids."""
        # env_ids -> list[int]
        ids = [int(x) for x in env_ids]

        for env_id in ids:
            # resolve concrete path by replacing env_.* with env_{env_id}
            prim_path = re.sub(r"/envs?/env_[^/]+/", f"/envs/env_{env_id}/", self.expr, count=1)

            view = self._views.get(env_id)
            if view is None:
                # many builds accept a list[str] here; if yours accepts a string, change to prim_paths_expr=prim_path
                view = self._sim_view.create_rigid_body_view([prim_path])
                self._views[env_id] = view
            breakpoint()
            tf = view.get_transforms()  # (K,7): [px,py,pz,qx,qy,qz,qw] (XYZW)
            if tf.shape[0] < 1:
                raise RuntimeError(f"No rigid body found for env {env_id} at '{prim_path}'")

            tf_t = torch.as_tensor(tf, device=self.device, dtype=torch.float32)
            pos = tf_t[0, :3]
            q_xyzw = tf_t[0, 3:7]
            # reorder XYZW -> WXYZ
            q_wxyz = math_utils.convert_quat(q_xyzw, to="wxyz")

            self.pos_w[env_id]  = pos
            self.quat_w[env_id] = q_wxyz

    def get_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (pos_w [num_envs,3], quat_w [num_envs,4 WXYZ])."""
        return self.pos_w, self.quat_w

class SharpaVBTS(RayCaster):
    """Sharpa Vision-Based Tactile Sensor (VBTS) based on ray casting from surface points and normals."""

    cfg: SharpaVBTSCfg
    UNSUPPORTED_TYPES: ClassVar[set[str]] = set()

    # def __init__(self, cfg: SharpaVBTSCfg):
    #     super().__init__(cfg)
    #     self._data = CameraData()

    def __init__(self, cfg: SharpaVBTSCfg):
        # check data_types, no need to write additional functions as ray_caster_camera.py
        for name in cfg.data_types:
            if name not in ["distance_along_normal"]:
                raise ValueError(f"Unsupported data type: {name}")
        # initialize
        super().__init__(cfg)
        # create empty variables for storing output data, also could use CameraData to store
        self._data = CameraData()
        self.mesh_transfer_b2w = None


    def __str__(self) -> str:
        return (
            f"Sharpa VBTS @ '{self.cfg.prim_path}': \n"
            f"\tview type            : {self._view.__class__}\n"
            f"\tupdate period (s)    : {self.cfg.update_period}\n"
            f"\tnumber of meshes     : {len(self.meshes)}\n"
            f"\tnumber of sensors    : {self._view.count}\n"
            f"\tnumber of rays/sensor: {self.num_rays}\n"
            f"\ttotal number of rays : {self.num_rays * self._view.count}"
        )
    """
    Properties
    """

    @property
    def data(self) -> CameraData:
        self._update_outdated_buffers()
        return self._data
    
    """
    Operations.
    """
    
    def reset(self, env_ids: Sequence[int] | None = None):
        # reset the timestamps
        super().reset(env_ids)
        # resolve None
        if env_ids is None:
            env_ids = slice(None)
        # Reset the frame count
        self._frame[env_ids] = 0
        
        # # get mesh initial position and transfer to zero, makesure the mesh is in zero in baked reference frame
        # mesh_path = self.cfg.mesh_prim_paths[0]  # 使用第一个 mesh_prim_paths
        # child = sim_utils.get_first_matching_child_prim(mesh_path, lambda p: p.GetTypeName() == "Mesh") \
        #     or sim_utils.get_first_matching_child_prim(mesh_path, lambda p: p.GetTypeName() == "Plane")

        # if child is not None and child.IsValid():
        #     self.mesh_transfer_b2w = np.linalg.inv(np.array(omni_usd.get_world_transform_matrix(child)))

    """
    Implementation.
    """

    # num_envs: number of environments created
    # N: number of points/norms

    # _w: world reference frame
    # _tar: target reference frame
    # _att: attached body reference frame
    # _baked: baked reference frame

    def _initialize_rays_impl(self):
        # Create all indices buffer and Create frame count buffer
        self._ALL_INDICES = torch.arange(self._view.count, device=self._device, dtype=torch.long)
        self._frame = torch.zeros(self._view.count, device=self._device, dtype=torch.long)

        # read the points and normals on the surface points array (load on CPU then copy to device)
        pts_np = np.load(self.cfg.points_npy)
        nrm_np = np.load(self.cfg.normals_npy)
        # (x,y,3) reshape to (N, 3), N=x*y
        pts_np = np.asarray(pts_np).reshape(-1, 3)
        nrm_np = np.asarray(nrm_np).reshape(-1, 3)

        assert pts_np.shape[-1] == 3 and nrm_np.shape[-1] == 3
        assert pts_np.shape[0] == nrm_np.shape[0]
        # normalize normals just in case, may be uunnecessary
        nrm_norm = np.linalg.norm(nrm_np, axis=-1, keepdims=True) + 1e-12
        nrm_np = nrm_np / nrm_norm
        # to torch tensor and copy to device
        pts = torch.tensor(pts_np, dtype=torch.float32, device=self._device)
        nrms = torch.tensor(nrm_np, dtype=torch.float32, device=self._device)

        # problems, wether flip the nrms is a question????
        # correct scale
        pts = pts * self.cfg.correction_scale  # cm to m
        nrms = -nrms  # flip normals - no need as current direction is already correct
        pts_offset = -0.0016  # meters
        pts = pts + pts_offset * nrms

        # copy to every envs（num_envs, N, 3）
        self.num_rays = pts.shape[0]
        # create buffers after self.num_rays is properly set
        self._create_buffers()

        # ray_starts and ray_directions size become:
        # (num_envs, N, 3)
        # include the value of pts and nrms, in AttachedBody reference frame
        self.ray_starts_att = pts.unsqueeze(0).repeat(self._view.count, 1, 1)
        self.ray_directions_att = nrms.unsqueeze(0).repeat(self._view.count, 1, 1)
        # create buffer for storing ray hits (num_envs, N, 3)
        self.ray_hits_w = torch.zeros(self._view.count, self.num_rays, 3, device=self._device)

        # offset pose (same as camera)
        quat_w = math_utils.convert_camera_frame_orientation_convention(
            torch.tensor([self.cfg.offset.rot], device=self._device),
            origin=self.cfg.offset.convention, target="world"
        )
        self._offset_quat = quat_w.repeat(self._view.count, 1)
        self._offset_pos = torch.tensor(list(self.cfg.offset.pos), device=self._device).repeat(self._view.count, 1)
        
        # visualization of ray hits points (in debug_viz), show at most ~5k points
        self._viz_stride = max(1, self.num_rays // 5000)

        # how many samples per env when we downsample with [::self._viz_stride]
        self._viz_count = (self.num_rays + self._viz_stride - 1) // self._viz_stride

        # Create a per-env buffer for visualization hits in WORLD coordinates
        # Shape: (num_envs, viz_count, 3)
        if not hasattr(self._data, "ray_hits_w"):
            self._data.ray_hits_w = torch.zeros(
                (self._view.count, self._viz_count, 3),
                device=self._device,
                dtype=torch.float32,
            )
        # how many samples we keep per env

        self.target_physx_sim_view = SimView(self.cfg.target_rigid_expr, self._view.count)

        # get mesh initial position and transfer to zero, makesure the mesh is in zero in baked reference frame
        mesh_path = self.cfg.mesh_prim_paths[0]  # 使用第一个 mesh_prim_paths
        child = sim_utils.get_first_matching_child_prim(mesh_path, lambda p: p.GetTypeName() == "Mesh") \
            or sim_utils.get_first_matching_child_prim(mesh_path, lambda p: p.GetTypeName() == "Plane")

        # if child is not None and child.IsValid():
        self.mesh_transfer_b2w = np.array(omni_usd.get_world_transform_matrix(child))
            

    def _create_buffers(self):
        """
        Create buffers for storing data. used in initialization (_initialize_rays_impl).
        Note: different from ray_caster_camera, here num_rays is determined after loading points.npy
        """
        self._data.pos_sensor_w = torch.zeros((self._view.count, 3), device=self._device)
        self._data.quat_sensor_w = torch.zeros((self._view.count, 4), device=self._device)
        self._data.output = {}

        for name in self.cfg.data_types:
            if name == "distance_along_normal":
                shape = (self.num_rays if hasattr(self, "num_rays") else 1, 1)
            else:
                raise ValueError(f"Unknown data type: {name}")
            self._data.output[name] = torch.zeros((self._view.count, *shape), device=self._device)
        
        self.drift = torch.zeros((self._view.count, 3), device=self._device, dtype=torch.float32)
        self.bias  = torch.zeros((self._view.count, 3), device=self._device, dtype=torch.float32)
        self.ray_cast_drift = torch.zeros((self._view.count, 3), device=self._device, dtype=torch.float32)
        self.ray_cast_bias  = torch.zeros((self._view.count, 3), device=self._device, dtype=torch.float32)

        self.target_pos_w = torch.zeros((self._view.count, 3), device=self._device)
        self.target_quat_w = torch.zeros((self._view.count, 4), device=self._device)


    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Fills the buffers of the sensor data. main sensing logic implemented here."""
        self._frame[env_ids] += 1
        self.target_physx_sim_view.update(env_ids)
        self.target_pos_w, self.target_quat_w = self.target_physx_sim_view.get_tensors()
        self.update_ray_hits_w(env_ids)

    # calculate the world pose of the view (parent that the sensor is attached to)
    # same in ray_caster_camera.py, no need to check
    # mainly use physx.RigidBodyView, get_transforms
    def _compute_view_world_poses(self, env_ids: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(self._view, XFormPrim):
            if isinstance(env_ids, slice):
                env_ids = self._ALL_INDICES
            pos_w, quat_w = self._view.get_world_poses(env_ids)
        elif isinstance(self._view, physx.ArticulationView):
            pos_w, quat_w = self._view.get_root_transforms()[env_ids].split([3, 4], dim=-1)
            quat_w = math_utils.convert_quat(quat_w, to="wxyz")
        elif isinstance(self._view, physx.RigidBodyView):
            pos_w, quat_w = self._view.get_transforms()[env_ids].split([3, 4], dim=-1)
            quat_w = math_utils.convert_quat(quat_w, to="wxyz")
        else:
            raise RuntimeError(f"Unsupported view type: {type(self._view)}")
        return pos_w.clone(), quat_w.clone()
    
    """
    Implementation. Additional functions
    """
    def get_world_position(self, prim_path: str):
        """Return (x, y, z) world position of a prim, or None if invalid."""
        try:
            stage = omni_usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                return None
            T = omni_usd.get_world_transform_matrix(prim)  # Gf.Matrix4d
            if T is None:
                return None
            p: Gf.Vec3d = T.ExtractTranslation()
            return float(p[0]), float(p[1]), float(p[2])
        except Exception:
            return None

    # need further test since the currect experiment do not include initial object with orientation
    def get_world_rotation_wxyz(self, prim_path: str):
        stage = omni_usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return None
        T = omni_usd.get_world_transform_matrix(prim)
        if T is None:
            return None
        q = T.ExtractRotation().GetQuat()  # Gf.Quatd
        w = float(q.GetReal()); x, y, z = q.GetImaginary()
        return w, float(x), float(y), float(z)


    
    def update_ray_hits_w(self, env_ids: Sequence[int]):
        """
        Convert attached-frame rays -> world frame (store into cal_ray_starts_w / cal_ray_directions_w),
        then world -> baked(target-local) using cal_target_pos_w / cal_target_quat_w,
        raycast per env (since raycast_mesh expects (N,3)), map hits back to world, and store.
        """
        # normalize env_ids to a python list of ints
        if isinstance(env_ids, slice):
            env_ids = self._ALL_INDICES.tolist()
        elif isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.detach().cpu().tolist()
        else:
            env_ids = list(map(int, env_ids))

        # world pose of attached body (+ sensor offset) for all env_ids
        pos_att_w, quat_att_w = self._compute_view_world_poses(env_ids)
        pos_sensor_w = pos_att_w + math_utils.quat_apply(quat_att_w, self._offset_pos[env_ids])
        quat_sensor_w = math_utils.quat_mul(quat_att_w, self._offset_quat[env_ids])

        # store pose into data
        self._data.pos_sensor_w[env_ids] = pos_sensor_w
        self._data.quat_sensor_w[env_ids] = quat_sensor_w

        # (optional) collect downsampled hits for viz
        viz_hits = []

        # cal_: temperary variable used in calculation for loop
        # per-env loop (raycast_mesh requires (N,3)), easy for debug, may collect to use env_ids in the future
        for i, env_id in enumerate(env_ids):
            # attached-body frame rays for this env
            cal_ray_starts_att = self.ray_starts_att[env_id]       # (N,3)
            cal_ray_directions_att = self.ray_directions_att[env_id]    # (N,3)
            cal_quat_sensor_w = self._data.quat_sensor_w[env_id].unsqueeze(0)  # (1,4 wxyz)
            cal_pos_sensor_w = self._data.pos_sensor_w[env_id]     # (3,)
            cal_target_pos_w = self.target_pos_w[env_id]            # (3,)
            cal_target_quat_w = self.target_quat_w[env_id]           # (4, wxyz)

            # attached -> world
            # cal_quat_sensor_w (1,4) -> repeat (N, 4)
            cal_ray_starts_w = math_utils.quat_apply(cal_quat_sensor_w.repeat(self.num_rays, 1), cal_ray_starts_att) + cal_pos_sensor_w.repeat(self.num_rays, 1)
            cal_ray_directions_w = math_utils.quat_apply(cal_quat_sensor_w.repeat(self.num_rays, 1), cal_ray_directions_att)

            # world -> baked (0,0,0), note the baked mesh may not locate at (0,0,0)
            # Rotation from baked to world is equivlant to current target quat in world frame
            cal_Rotation_b2w = math_utils.matrix_from_quat(cal_target_quat_w)  # (4, wxyz) -> (3,3)
            cal_Rotation_w2b = cal_Rotation_b2w.T

            cal_ray_starts_b = (cal_ray_starts_w - (cal_target_pos_w.unsqueeze(0)).repeat(self.num_rays, 1))@ cal_Rotation_w2b
            # cal_ray_starts_b = (cal_ray_starts_w - cal_target_pos_w)@ cal_Rotation_w2b
            cal_ray_directions_b = torch.nn.functional.normalize(cal_ray_directions_w @ cal_Rotation_w2b)

            # # since the mesh may not at origin at baked frame in DirectRLEnv, provide a compensate
            # update of target mesh current pos and rot
            # Note! mesh_prim_paths[0]!, which is the baked mesh, the baked mesh is not located at origin
            # but at the initial setup position in scene, as a result, need to get its initial pos and rot to provide compensate
            prim_path = self.cfg.mesh_prim_paths[0]
            x,y,z = self.get_world_position(prim_path)
            rw, rx, ry, rz = self.get_world_rotation_wxyz(prim_path)
            tn = torch.tensor([x,y,z], device=self._device, dtype=cal_ray_starts_b.dtype)
            rn = torch.tensor([rw,rx,ry,rz], device=self._device, dtype=cal_ray_starts_b.dtype)
            cal_ray_starts_b_compensate = math_utils.quat_apply(rn.repeat(self.num_rays, 1), cal_ray_starts_b) + tn
            cal_ray_directions_b_compensate = math_utils.quat_apply(rn.repeat(self.num_rays, 1), cal_ray_directions_b)

            # raycast in baked frame (per-env)
            # raycast in baked frame (per-env) -- NEED (B,N,3)
            sensor_output_hits_b, sensor_output_distance, _, _ = raycast_mesh(
                # cal_ray_starts_b.unsqueeze(0),             # (1, N, 3)
                cal_ray_starts_b_compensate.unsqueeze(0),             # (1, N, 3)
                # cal_ray_directions_b.unsqueeze(0),             # (1, N, 3)
                cal_ray_directions_b_compensate.unsqueeze(0),             # (1, N, 3)
                mesh=self.meshes[self.cfg.mesh_prim_paths[0]],
                max_dist=self.cfg.max_distance,
                return_distance=True,
                return_normal=False,
            )

            # # CPD inside test
            # # only perform inside test for the rays that have hit something
            # # torch_bool, inf, -inf, non will return false
            # cpd_torch_bool_has_hit = torch.isfinite(sensor_output_distance)
            # # the core function are written in _compute_inside_mask_baked with input of start points
            # cpd_torch_bool_inside_mask = self._compute_inside_mask_baked(sensor_output_hits_b)
            # cpd_torch_bool_valid_mask = cpd_torch_bool_has_hit & cpd_torch_bool_inside_mask
            # # bitwise NOT operation (~)
            # cpd_torch_bool_outside = ~cpd_torch_bool_valid_mask
            cpd_hits_b_1st = sensor_output_hits_b[0]            # (N,3)
            cpd_distance_1st = sensor_output_distance[0]            # (N,)
            cpd_has_hit_1st = torch.isfinite(sensor_output_distance[0])

            # --- CPD: advance-start second raycast just beyond the first hit ---
            cpd_eps = 1e-4       # tiny step beyond the surface
            cpd_max_dist = 0.1      # short range for CPD ray

            # new start points = start + dir * (dist + eps)
            cpd_advance = (cpd_distance_1st.clamp_min(0.0) + cpd_eps).unsqueeze(-1)            # (N,1)
             # use the same starts as first pass
            cpd_ray_starts_b = cal_ray_starts_b_compensate + cal_ray_directions_b_compensate * cpd_advance              # (N,3)

            # second pass from advanced starts
            _, cpd_output_distance, _, _ = raycast_mesh(
                cpd_ray_starts_b.unsqueeze(0),                         # (1,N,3)
                cal_ray_directions_b_compensate.unsqueeze(0),            # (1,N,3) same direction
                mesh=self.meshes[self.cfg.mesh_prim_paths[0]],
                max_dist=cpd_max_dist,                        # small window right after the first surface
                return_distance=True,
                return_normal=False,
            )
            cpd_has_hit_2nd = torch.isfinite(cpd_output_distance[0])            # (N,) True => contact detected (outside)

            # Invalidate when contact is detected on CPD pass
            # calculation of the torch bool
            cpd_keep = cpd_has_hit_1st & ~cpd_has_hit_2nd

            # filter out the "odd situation" that the points is outside the object mesh
            cpd_dist_final = torch.where(cpd_keep, cpd_distance_1st, torch.zeros_like(cpd_distance_1st))          # (N,)
            cpd_hits_b_final = torch.where(cpd_keep.unsqueeze(-1), cpd_hits_b_1st, torch.zeros_like(cpd_hits_b_1st))  # (N,3)


            # remove batch dim back to (N,3) and (N,)
            # sensor_output_hits_b = sensor_output_hits_b[0]
            # sensor_output_distance = sensor_output_distance[0]
            sensor_output_hits_b = cpd_hits_b_final
            sensor_output_distance = cpd_dist_final
            
            # baked -> world
            sensor_output_hits_w = sensor_output_hits_b @ cal_Rotation_b2w + cal_target_pos_w

            # write full hits
            self.ray_hits_w[env_id] = sensor_output_hits_w

            # distances clipping policy
            if "distance_along_normal" in self._data.output:
                depth = sensor_output_distance.clone()
                # non-finite -> 0
                depth = torch.where(torch.isfinite(depth), depth, torch.zeros_like(depth))
                self._data.output["distance_along_normal"][env_id] = depth.unsqueeze(-1)
            
            # optional: collect viz points
            # if getattr(self.cfg, "debug_vis", False):
            #     viz_hits.append(hits_w[::self._viz_stride, :].contiguous())

            if getattr(self.cfg, "debug_vis", False):
                # 抽样：每 self._viz_stride 取一个点，避免太密集
                # viz_hits.append(cal_ray_starts_b[::self._viz_stride, :].contiguous())
                viz_hits.append(cal_ray_starts_b_compensate[::self._viz_stride, :].contiguous())

        # write viz buffer in one go (if enabled)
        if getattr(self.cfg, "debug_vis", False) and len(viz_hits) > 0:
            self._data.ray_hits_w[env_ids] = torch.stack(viz_hits, dim=0)
            # self._data.ray_hits_w[0] = torch.stack(viz_hits, dim=0)
