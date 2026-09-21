from dataclasses import MISSING
from typing import Literal, Optional, List
from isaaclab.utils import configclass
from isaaclab.sensors.ray_caster.ray_caster_cfg import RayCasterCfg
from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg
# import isaaclab.sim as sim_utils
# from isaaclab.markers.visualization_markers import VisualizationMarkersCfg

@configclass
class SharpaVBTSCfg(RayCasterCfg):
    """配置 Sharpa Vision-Based Tactile Sensor (VBTS)，支持并行环境及目标物体绑定"""

    # vbts, sampled points and normal directions（numpy formate）
    points_npy: str = MISSING  # "/path/to/surface_points.npy"
    normals_npy: str = MISSING # "/path/to/surface_normals.npy"
    
    # offsets and coordinate constrain
    @configclass
    class OffsetCfg:
        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # wxyz
        convention: Literal["opengl", "ros", "world"] = "ros"
        
    offset: OffsetCfg = OffsetCfg()

    # output data type（default is "distance_along_normal"）
    data_types: List[str] = ["distance_along_normal"]

    target_rigid_expr = None

    # numpy points use mm as the unit, may different from the sim world
    correction_scale: float = 1e-3

    # use the pattern config from ray-caster (required but not used)
    pattern_cfg: PinholeCameraPatternCfg = MISSING

    # RAY_CASTER_MARKER_CFG = VisualizationMarkersCfg(
    #     markers={
    #         "hit": sim_utils.SphereCfg(
    #             radius=0.02,
    #             visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
    #         ),
    #     },
    # )
