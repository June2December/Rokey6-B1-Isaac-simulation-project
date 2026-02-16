#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from omni.isaac.kit import SimulationApp

# ROBOT_USD   = "/home/june/cobot3/rover/sim/assets/robots/aau_rover/Mars_Rover.usd"
ROBOT_USD   = "/home/june/cobot3/rover/sim/assets/robots/aau_rover/AAU_Rover_With_Arm.usd"
TERRAIN_USD = "/home/june/cobot3/rover/sim/assets/terrains/debug/debug1/terrain_merged.usd"

simulation_app = SimulationApp({"headless": False})

import time
import numpy as np
import omni.usd
from pxr import UsdGeom, UsdPhysics, UsdLux, Gf
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.articulations import Articulation


def find_articulation_root(stage, prefix="/World/Rover"):
    for prim in stage.Traverse():
        if not prim.IsValid():
            continue
        p = str(prim.GetPath())
        if p.startswith(prefix) and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            return p
    return prefix


def add_basic_lights(stage):
    # DomeLight
    dome_path = "/World/Lights/DomeLight"
    if not stage.GetPrimAtPath(dome_path).IsValid():
        dome = UsdLux.DomeLight.Define(stage, dome_path)
        dome.CreateIntensityAttr(800.0)

    # DistantLight
    sun_path = "/World/Lights/SunLight"
    if not stage.GetPrimAtPath(sun_path).IsValid():
        sun = UsdLux.DistantLight.Define(stage, sun_path)
        sun.CreateIntensityAttr(2500.0)
        xform = UsdGeom.Xformable(stage.GetPrimAtPath(sun_path))
        xform.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 45.0))


def apply_collision_to_terrain_meshes(stage, terrain_prefix="/World/Terrain"):
    """
    버전 차이 때문에 PhysxSchema.*는 건드리지 않고,
    Mesh에 USD CollisionAPI만 적용합니다. (일단 관통 방지 목적)
    """
    applied = 0
    for prim in stage.Traverse():
        if not prim.IsValid():
            continue
        path = str(prim.GetPath())
        if not path.startswith(terrain_prefix):
            continue

        if prim.IsA(UsdGeom.Mesh):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(prim)
                applied += 1

    print(f"[INFO] terrain collision applied (UsdPhysics.CollisionAPI): {applied}")
    return applied


# 1) World
world = World(stage_units_in_meters=1.0, physics_dt=1.0/60.0, rendering_dt=1.0/60.0)

# 2) Load USDs
add_reference_to_stage(TERRAIN_USD, "/World/Terrain")
add_reference_to_stage(ROBOT_USD,   "/World/Rover")

# 3) Stage warm-up
ctx = omni.usd.get_context()
for _ in range(120):
    simulation_app.update()
    time.sleep(0.01)
stage = ctx.get_stage()

from pxr import UsdGeom, Gf

rover_prim = stage.GetPrimAtPath("/World/Rover")

xform = UsdGeom.Xformable(rover_prim)

# 기존 transform이 있으면 덮어쓰기
ops = xform.GetOrderedXformOps()
if len(ops) == 0:
    translate_op = xform.AddTranslateOp()
else:
    translate_op = ops[0]

translate_op.Set(Gf.Vec3d(20.0, 20.0, 1.0))


# 4) Lights
add_basic_lights(stage)

# 5) Terrain collision (USD Collision only)
apply_collision_to_terrain_meshes(stage, "/World/Terrain")

# 6) Articulation path
arti_root = find_articulation_root(stage, "/World/Rover")
print("[INFO] articulation prim_path =", arti_root)

# 7) Articulation
rover = Articulation(prim_path=arti_root, name="rover")
world.scene.add(rover)

world.reset()

# 8) DOFs
dof_names = rover.dof_names
print("\n=== DOF NAMES ===")
for i, n in enumerate(dof_names):
    print(f"{i:02d}: {n}")

wheel_ids = [i for i, n in enumerate(dof_names) if "drive" in n.lower()]
print("\n[INFO] wheel_ids =", wheel_ids)
if len(wheel_ids) == 0:
    raise RuntimeError("Drive DOF를 못 찾았습니다.")

wheel_vel = np.full((len(wheel_ids),), 6.0, dtype=np.float32)

# 9) Run
for step in range(1200):
    rover.set_joint_velocities(wheel_vel, joint_indices=np.array(wheel_ids, dtype=np.int32))
    world.step(render=True)

simulation_app.close()
