"""
scene_utils.py — USD 씬 마커 생성 유틸리티
"""
import os
import omni.kit.commands
from pxr import UsdGeom, UsdShade, Gf, Sdf


def _apply_color(stage, prim_path: str, color: tuple):
    mat_path = prim_path + "/Material"
    material = UsdShade.Material.Define(stage, mat_path)
    shader   = UsdShade.Shader.Define(stage, mat_path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness",    Sdf.ValueTypeNames.Float).Set(0.3)
    shader.CreateInput("metallic",     Sdf.ValueTypeNames.Float).Set(0.8)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(prim_path)).Bind(material)


def _safe_xform(stage, prim_path: str, translate: tuple, scale: tuple):
    prim = stage.GetPrimAtPath(prim_path)
    xf   = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()

    t_attr = prim.GetAttribute("xformOp:translate")
    if t_attr.IsValid():
        t_attr.Set(Gf.Vec3d(*translate)); t_op = UsdGeom.XformOp(t_attr)
    else:
        t_op = xf.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
        t_op.Set(Gf.Vec3d(*translate))

    s_attr = prim.GetAttribute("xformOp:scale")
    if s_attr.IsValid():
        s_attr.Set(Gf.Vec3d(*scale)); s_op = UsdGeom.XformOp(s_attr)
    else:
        s_op = xf.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble)
        s_op.Set(Gf.Vec3d(*scale))

    xf.SetXformOpOrder([t_op, s_op])


_ROCKET_USD = os.path.join(
    os.path.dirname(__file__),
    "..", "rover_envs", "assets", "terrains", "debug", "debug1", "Rocket_default.usd"
)


def spawn_basecamp_marker(stage, cx: float, cy: float, visual_radius: float = 10.0):
    """
    terrain_importer가 만든 /World/Basecamp 아래에 rocket을 추가.
    파일 없으면 실린더 폴백.
    """
    rocket_path = os.path.abspath(_ROCKET_USD)
    prim_path   = "/World/Basecamp/Rocket"

    if os.path.exists(rocket_path):
        # /World/Basecamp 는 terrain_importer가 이미 만들었으므로 그 아래에 추가
        omni.kit.commands.execute(
            "CreatePrimWithDefaultXform",
            prim_type="Xform",
            prim_path=prim_path,
        )
        prim = stage.GetPrimAtPath(prim_path)
        prim.GetReferences().AddReference(rocket_path, "/Rocket")
        _safe_xform(stage, prim_path,
                    translate=(cx, cy, -15.0), scale=(7.0, 7.0, 5.5))
        print(f"[Scene] 로켓 배치: ({cx:.1f}, {cy:.1f})  경로: {prim_path}")
    else:
        print(f"[Scene] Rocket_default.usd 없음 → 실린더 폴백: {rocket_path}")
        omni.kit.commands.execute(
            "CreatePrimWithDefaultXform",
            prim_type="Cylinder",
            prim_path=prim_path,
        )
        _safe_xform(stage, prim_path,
                    translate=(cx, cy, 0.01), scale=(visual_radius, visual_radius, 0.01))
        _apply_color(stage, prim_path, (0.05, 0.9, 0.15))
        print(f"[Scene] 실린더 마커 생성: ({cx:.1f}, {cy:.1f})")


def get_basecamp_center(raw_env) -> tuple:
    terrain = raw_env.scene.terrain
    hm = terrain._terrainManager._heightmap_manager
    return (hm.min_x + hm.max_x) / 2.0, (hm.min_y + hm.max_y) / 2.0