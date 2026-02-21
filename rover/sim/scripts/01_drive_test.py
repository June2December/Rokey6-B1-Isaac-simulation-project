from pathlib import Path
from omni.isaac.kit import SimulationApp


SCRIPT_DIR = Path(__file__).resolve().parent
SIM_DIR    = SCRIPT_DIR.parent
ROBOT_USD = SIM_DIR / "assets" / "robots" / "aau_rover_simple" / "mobile_manipulator_instance" / "rover_instance.usd"
TERRAIN_USD = SIM_DIR / "assets" / "terrains" / "debug" / "debug1" / "terrain_merged.usd"
ROBOT_CONTAINER = "/World/Robot"
ROVER_PATH = f"{ROBOT_CONTAINER}/mobile_manipulator/rover"

simulation_app = SimulationApp({"headless": False})

import time
import numpy as np
import omni.usd
from pxr import UsdGeom, UsdPhysics, UsdLux, Gf
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.articulations import Articulation


def find_articulation_root(stage, prefix):
    candidates = []
    for prim in stage.Traverse():
        if not prim.IsValid():
            continue
        p = str(prim.GetPath())
        if p.startswith(prefix) and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            candidates.append(p)

    if not candidates:
        raise RuntimeError(f"No ArticulationRootAPI found under {prefix}")

    return min(candidates, key=len)


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
add_reference_to_stage(str(TERRAIN_USD), "/World/Terrain")
add_reference_to_stage(str(ROBOT_USD), ROBOT_CONTAINER)

# 3) Stage warm-up
ctx = omni.usd.get_context()
for _ in range(120):
    simulation_app.update()
    time.sleep(0.01)
stage = ctx.get_stage()
# --- DEBUG: print actual prim paths under /World/Robot ---
def print_tree(stage, root_path, max_depth=4):
    root = stage.GetPrimAtPath(root_path)
    print(f"[DEBUG] root valid? {root.IsValid()}  path={root_path}")
    if not root.IsValid():
        return
    def rec(prim, depth):
        if depth > max_depth:
            return
        print("  " * depth + f"- {prim.GetPath()} ({prim.GetTypeName()})")
        for c in prim.GetChildren():
            rec(c, depth + 1)
    rec(root, 0)

print_tree(stage, ROBOT_CONTAINER, max_depth=6)
# --- DEBUG end ---
print("[DEBUG] Does ROVER_PATH exist? ->", stage.GetPrimAtPath(ROVER_PATH).IsValid(), ROVER_PATH)

container_prim = stage.GetPrimAtPath(ROBOT_CONTAINER)
print("[DEBUG] Container valid? ->", container_prim.IsValid(), ROBOT_CONTAINER)
if container_prim.IsValid():
    print("[DEBUG] Children of container:")
    for c in container_prim.GetChildren():
        print(" -", c.GetPath())

from pxr import UsdGeom, Gf

rover_prim = stage.GetPrimAtPath(ROVER_PATH)
if not rover_prim.IsValid():
    raise RuntimeError(f"ROVER_PATH invalid: {ROVER_PATH} (check printed children paths above)")
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
arti_root = find_articulation_root(stage, ROVER_PATH)
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