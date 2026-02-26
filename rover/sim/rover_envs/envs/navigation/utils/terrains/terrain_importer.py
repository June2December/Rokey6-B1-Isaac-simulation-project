from typing import Sequence

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.utils.math import quat_from_euler_xyz, quat_apply_inverse, wrap_to_pi, yaw_quat

from .terrain_utils import TerrainManager

SPHERE_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "sphere": sim_utils.SphereCfg(
            radius=0.2,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0)),
        ),
    }
)


class TerrainBasedPositionCommand(CommandTerm):
    """Command generator that generates position commands based on the terrain."""

    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.terrain: TerrainImporter = env.scene.terrain
        self.pos_command_w = torch.ones(self.num_envs, 3, device=self.device)
        self.heading_command_w = torch.ones(self.num_envs, device=self.device)
        self.pos_command_b = torch.ones_like(self.pos_command_w)
        self.heading_command_b = torch.ones_like(self.heading_command_w)
        self.metrics["error_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_heading"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "TerrainBasedPositionCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        return torch.cat((self.pos_command_b, self.heading_command_b.unsqueeze(1)), dim=1)

    def _resample_command(self, env_ids: Sequence[int]):
        self.pos_command_w[env_ids] = self.terrain.sample_new_targets(env_ids)
        self.pos_command_w[env_ids, 2] += self.robot.data.default_root_state[env_ids, 2]
        r = torch.empty(len(env_ids), device=self.device)
        self.heading_command_w[env_ids] = r.uniform_(*self.cfg.ranges.heading)

    def _update_command(self):
        target_vec = self.pos_command_w - self.robot.data.root_link_pos_w[:, :3]
        self.pos_command_b[:] = quat_apply_inverse(
            yaw_quat(self.robot.data.root_link_quat_w), target_vec)
        self.heading_command_b[:] = wrap_to_pi(
            self.heading_command_w - self.robot.data.heading_w)

    def _update_metrics(self):
        self.metrics["error_pos"] = torch.norm(
            self.pos_command_w - self.robot.data.root_link_pos_w[:, :3], dim=1)
        self.metrics["error_heading"] = torch.abs(wrap_to_pi(
            self.heading_command_w - self.robot.data.heading_w))

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "arrow_goal_visualizer"):
                arrow_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                arrow_cfg.prim_path = "/Visuals/Command/heading_goal"
                arrow_cfg.markers["arrow"].scale = (0.2, 0.2, 0.8)
                self.arrow_goal_visualizer = VisualizationMarkers(arrow_cfg)
            if not hasattr(self, "sphere_goal_visualizer"):
                sphere_cfg = SPHERE_MARKER_CFG.copy()
                sphere_cfg.prim_path = "/Visuals/Command/position_goal"
                sphere_cfg.markers["sphere"].radius = 0.2
                self.sphere_goal_visualizer = VisualizationMarkers(sphere_cfg)
            self.arrow_goal_visualizer.set_visibility(True)
            self.sphere_goal_visualizer.set_visibility(True)
        else:
            if hasattr(self, "arrow_goal_visualizer"):
                self.arrow_goal_visualizer.set_visibility(False)
            if hasattr(self, "sphere_goal_visualizer"):
                self.sphere_goal_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        self.sphere_goal_visualizer.visualize(self.pos_command_w)
        zero_vec = torch.zeros_like(self.heading_command_w)
        quaternion = quat_from_euler_xyz(zero_vec, zero_vec, self.heading_command_w)
        position_arrow_w = self.pos_command_w + torch.tensor([0.0, 0.0, 0.25], device=self.device)
        self.arrow_goal_visualizer.visualize(position_arrow_w, quaternion)


class RoverTerrainImporter(TerrainImporter):
    def __init__(self, cfg: TerrainImporterCfg):
        super().__init__(cfg)
        self._cfg = cfg
        self._terrainManager = TerrainManager(
            num_envs=self._cfg.num_envs, device=self.device)
        self.target_distance = 9.0
        # walls 제거 — scene_utils.py의 spawn_basecamp_marker()가 rocket으로 대체

    def sample_new_targets(self, env_ids):
        original_env_ids = env_ids

        hm = self._terrainManager._heightmap_manager
        _bcamp_half = 10.0 / 2.0
        _cx = (hm.min_x + hm.max_x) / 2.0
        _cy = (hm.min_y + hm.max_y) / 2.0

        target_position = torch.zeros(self._cfg.num_envs, 3, device=self.device)

        reset_buf_len = len(env_ids)
        while reset_buf_len > 0:
            target_position[env_ids] = self.generate_random_targets(env_ids, target_position)
            current_ids = env_ids
            env_ids, reset_buf_len = self._terrainManager.check_if_target_is_valid(
                current_ids, target_position[current_ids, 0:2], device=self.device
            )
            tx = target_position[current_ids, 0]
            ty = target_position[current_ids, 1]
            in_basecamp = (torch.abs(tx - _cx) < _bcamp_half) & \
                          (torch.abs(ty - _cy) < _bcamp_half)
            basecamp_ids = current_ids[in_basecamp]
            if len(basecamp_ids) > 0:
                env_ids = torch.unique(torch.cat([env_ids, basecamp_ids]))
                reset_buf_len = len(env_ids)

        target_position[original_env_ids, 2] = self._terrainManager._heightmap_manager.get_height_at(
            target_position[original_env_ids, 0:2]
        )
        return target_position[original_env_ids]

    def generate_random_targets(self, env_ids, target_position):
        radius = self.target_distance
        theta = torch.rand(len(env_ids), device=self.device) * 2 * torch.pi
        target_position[env_ids, 0] = torch.cos(theta) * radius + self.env_origins[env_ids, 0]
        target_position[env_ids, 1] = torch.sin(theta) * radius + self.env_origins[env_ids, 1]
        return target_position[env_ids]

    def get_spawn_locations(self):
        return self._terrainManager.spawn_locations