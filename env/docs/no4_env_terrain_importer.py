from typing import Sequence

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
# TODO (anton): Remove the following import since they were changed in the Orbit API
# from isaaclab.envs.mdp.commands.commands_cfg import TerrainBasedPositionCommandCfg
# from isaaclab.envs.mdp.commands.position_command import TerrainBasedPositionCommand
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

# TODO: THIS IS A TEMPORARY FIX, since terrain based command is changed in the Orbit API


class TerrainBasedPositionCommand(CommandTerm):
    """Command generator that generates position commands based on the terrain.

    The position commands are sampled from the terrain mesh and the heading commands are either set
    to point towards the target or are sampled uniformly.
    """

    """Configuration for the command generator."""

    def __init__(self, cfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # obtain the robot and terrain assets
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]
        # -- terrain
        self.terrain: TerrainImporter = env.scene.terrain

        # crete buffers to store the command
        # -- commands: (x, y, z, heading)
        self.pos_command_w = torch.ones(self.num_envs, 3, device=self.device)
        self.heading_command_w = torch.ones(self.num_envs, device=self.device)
        self.pos_command_b = torch.ones_like(self.pos_command_w)
        self.heading_command_b = torch.ones_like(self.heading_command_w)
        # -- metrics
        self.metrics["error_pos"] = torch.zeros(
            self.num_envs, device=self.device)
        self.metrics["error_heading"] = torch.zeros(
            self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "TerrainBasedPositionCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base position in base frame. Shape is (num_envs, 3)."""
        return torch.cat((self.pos_command_b, self.heading_command_b.unsqueeze(1)), dim=1)

    """
    Implementation specific functions.
    """

    def _resample_command(self, env_ids: Sequence[int]):
        # sample new position targets from the terrain
        self.pos_command_w[env_ids] = self.terrain.sample_new_targets(env_ids)
        # offset the position command by the current root position
        self.pos_command_w[env_ids,
                           2] += self.robot.data.default_root_state[env_ids, 2]
        # random heading command
        r = torch.empty(len(env_ids), device=self.device)
        self.heading_command_w[env_ids] = r.uniform_(*self.cfg.ranges.heading)

    def _update_command(self):
        """Re-target the position command to the current root position and heading."""
        target_vec = self.pos_command_w - \
            self.robot.data.root_link_pos_w[:, :3]
        self.pos_command_b[:] = quat_apply_inverse(
            yaw_quat(self.robot.data.root_link_quat_w), target_vec)
        self.heading_command_b[:] = wrap_to_pi(
            self.heading_command_w - self.robot.data.heading_w)

    def _update_metrics(self):
        # logs data
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

            # set their visibility to true
            self.arrow_goal_visualizer.set_visibility(True)
            self.sphere_goal_visualizer.set_visibility(True)
        else:
            if hasattr(self, "arrow_goal_visualizer"):
                self.arrow_goal_visualizer.set_visibility(False)
            if hasattr(self, "sphere_goal_visualizer"):
                self.sphere_goal_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the sphere marker
        self.sphere_goal_visualizer.visualize(self.pos_command_w)

        # update the arrow marker
        zero_vec = torch.zeros_like(self.heading_command_w)
        quaternion = quat_from_euler_xyz(
            zero_vec, zero_vec, self.heading_command_w)
        position_arrow_w = self.pos_command_w + \
            torch.tensor([0.0, 0.0, 0.25], device=self.device)
        self.arrow_goal_visualizer.visualize(position_arrow_w, quaternion)


class RoverTerrainImporter(TerrainImporter):
    def __init__(self, cfg: TerrainImporterCfg):
        super().__init__(cfg)
        self._cfg = cfg
        self._terrainManager = TerrainManager(
            num_envs=self._cfg.num_envs, device=self.device)
        self.target_distance = 9.0
        self._spawn_basecamp_walls()

    def _spawn_basecamp_walls(self, basecamp_size_m: float = 30.0, wall_height: float = 5.0, wall_thickness: float = 0.1):
        """Spawn 4 visual-only (no collision/physics) cuboid prims as a basecamp boundary marker.

        The walls form a closed square frame of ``basecamp_size_m × basecamp_size_m``
        centered at the terrain's geographic center. They are rendered as blue
        semi-transparent boxes but have **no** PhysicsCollisionAPI, so rovers pass
        straight through them.

        Args:
            basecamp_size_m: Side length of the basecamp zone in meters.
            wall_height:     Height of each wall prim in meters.
            wall_thickness:  Thickness (depth) of each wall prim in meters.
        """
        try:
            from pxr import UsdGeom, UsdShade, Sdf, Gf
            from isaacsim.core.utils.stage import get_current_stage
        except ImportError as e:
            print(f"[BasecampWalls] Warning: USD API not available ({e}). Skipping wall creation.")
            return

        hm  = self._terrainManager._heightmap_manager
        res = self._terrainManager.resolution_in_m

        # Terrain geographic center in world coordinates
        center_x = (hm.min_x + hm.max_x) / 2.0
        center_y = (hm.min_y + hm.max_y) / 2.0

        # Terrain height at center via direct heightmap lookup (avoids any offset issues)
        cx_grid = int((hm.max_x - hm.min_x) / 2.0 / res)
        cy_grid = int((hm.max_y - hm.min_y) / 2.0 / res)
        center_z = float(hm.heightmap[cy_grid, cx_grid]) + wall_height / 2.0

        half = basecamp_size_m / 2.0
        wl   = 20.0   # wall length

        stage = get_current_stage()

        # ── Parent Xform to group all wall prims ─────────────────────────────────
        stage.DefinePrim("/World/Basecamp", "Xform")

        # ── Blue semi-transparent material ────────────────────────────────────────
        mat_path = "/World/Basecamp/BasecampMaterial"
        material = UsdShade.Material.Define(stage, mat_path)
        shader   = UsdShade.Shader.Define(stage, mat_path + "/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.05, 0.25, 1.0))
        shader.CreateInput("roughness",    Sdf.ValueTypeNames.Float).Set(0.8)
        shader.CreateInput("metallic",     Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput("opacity",      Sdf.ValueTypeNames.Float).Set(1.0)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        # ── Wall definitions: (prim_name, translate_xyz, scale_xyz) ──────────────
        # Corners are covered by making N/S walls slightly longer (wl = size + thickness).
        wall_specs = [
            ("north_wall", (center_x,        center_y + half, center_z), (wl,              wall_thickness, wall_height)),
            ("south_wall", (center_x,        center_y - half, center_z), (wl,              wall_thickness, wall_height)),
            ("east_wall",  (center_x + half, center_y,        center_z), (wall_thickness,  wl,             wall_height)),
            ("west_wall",  (center_x - half, center_y,        center_z), (wall_thickness,  wl,             wall_height)),
        ]

        for name, translate, scale in wall_specs:
            prim_path = f"/World/Basecamp/{name}"
            cube  = UsdGeom.Cube.Define(stage, prim_path)
            prim  = cube.GetPrim()

            # Position and scale via xformOps
            xformable = UsdGeom.Xformable(prim)
            xformable.AddTranslateOp().Set(Gf.Vec3d(*translate))
            xformable.AddScaleOp().Set(Gf.Vec3d(*scale))

            # Bind material
            UsdShade.MaterialBindingAPI(prim).Bind(material)
            # NOTE: No PhysicsCollisionAPI is added → purely visual, rovers pass through

        print(
            f"[BasecampWalls] 4 visual walls spawned — "
            f"center world ({center_x:.1f}, {center_y:.1f}, {center_z:.1f}), "
            f"zone {basecamp_size_m:.0f}m × {basecamp_size_m:.0f}m, "
            f"height {wall_height:.1f}m"
        )

    def sample_new_targets(self, env_ids):
        # We need to keep track of the original env_ids, because we need to resample some of them
        original_env_ids = env_ids

        # Basecamp exclusion zone: 30×30m centered at terrain geographic center
        hm = self._terrainManager._heightmap_manager
        _bcamp_half = 30.0 / 2.0
        _cx = (hm.min_x + hm.max_x) / 2.0
        _cy = (hm.min_y + hm.max_y) / 2.0

        # Initialize the target position
        target_position = torch.zeros(
            self._cfg.num_envs, 3, device=self.device)

        # Sample new targets
        reset_buf_len = len(env_ids)
        while reset_buf_len > 0:
            # sample new random targets
            target_position[env_ids] = self.generate_random_targets(
                env_ids, target_position)

            # Save current env_ids before check_if_target_is_valid overwrites them
            current_ids = env_ids

            # Standard validity check (rocks + boundary distance)
            env_ids, reset_buf_len = self._terrainManager.check_if_target_is_valid(
                current_ids, target_position[current_ids, 0:2], device=self.device
            )

            # Basecamp exclusion: reject targets inside the 30×30m zone at terrain center
            tx = target_position[current_ids, 0]
            ty = target_position[current_ids, 1]
            in_basecamp = (torch.abs(tx - _cx) < _bcamp_half) & \
                          (torch.abs(ty - _cy) < _bcamp_half)
            basecamp_ids = current_ids[in_basecamp]
            if len(basecamp_ids) > 0:
                env_ids = torch.unique(torch.cat([env_ids, basecamp_ids]))
                reset_buf_len = len(env_ids)

        # Adjust the height of the target, so that it matches the terrain
        target_position[original_env_ids, 2] = self._terrainManager._heightmap_manager.get_height_at(
            target_position[original_env_ids, 0:2]
        )

        return target_position[original_env_ids]

    def generate_random_targets(self, env_ids, target_position):
        """
        This function generates random targets for the rover to navigate to.
        The targets are generated in a circle around the environment origin.

        Args:
            env_ids: The ids of the environments for which we need to generate targets.
            target_position: The target position buffer.
        """
        radius = self.target_distance
        theta = torch.rand(len(env_ids), device=self.device) * 2 * torch.pi

        # set the target x and y positions
        target_position[env_ids, 0] = torch.cos(
            theta) * radius + self.env_origins[env_ids, 0]
        target_position[env_ids, 1] = torch.sin(
            theta) * radius + self.env_origins[env_ids, 1]

        return target_position[env_ids]

    def get_spawn_locations(self):
        """
        This function returns valid spawn locations, that avoids spawning the rover on top of obstacles.

        Returns:
            spawn_locations: The spawn locations buffer. Shape (num_env, 3).
        """
        return self._terrainManager.spawn_locations


# class TerrainBasedPositionCommandCustom(TerrainBasedPositionCommand):

#     cfg: TerrainBasedPositionCommandCfg

#     def __init__(self, cfg: TerrainBasedPositionCommandCfg, env):
#         super().__init__(cfg, env)

#     def _set_debug_vis_impl(self, debug_vis: bool):
#         # create markers if necessary for the first tome
#         if debug_vis:
#             if not hasattr(self, "box_goal_visualizer"):
#                 marker_cfg = VisualizationMarkersCfg(
#                     prim_path="/Visuals/Command/position_goal",
#                     markers={
#                         "sphere": sim_utils.SphereCfg(
#                             radius=0.2,
#                             visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
#                         ),
#                     },
#                 )
#                 self.box_goal_visualizer = VisualizationMarkers(marker_cfg)
#             # set their visibility to true
#             self.box_goal_visualizer.set_visibility(True)
#         else:
#             if hasattr(self, "box_goal_visualizer"):
#                 self.box_goal_visualizer.set_visibility(False)

#     def _debug_vis_callback(self, event):
#         self.box_goal_visualizer.visualize(translations=self.pos_command_w, marker_indices=[0])
