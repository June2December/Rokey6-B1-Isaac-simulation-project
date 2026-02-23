"""
scene_utils.py — USD 씬 마커 생성 유틸리티
"""
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


def spawn_basecamp_marker(stage, cx: float, cy: float, visual_radius: float = 10.0):
    """
    베이스캠프 바닥 표시용 녹색 납작 실린더 생성.

    terrain_importer.py의 _spawn_basecamp_walls()는 공중 1.7m에
    얇은 막대기 4개를 띄우는 방식이라 바닥 표시가 없음.
    이 함수가 지형 위에 녹색 원형 바닥 마커를 추가해서 베이스캠프 위치를 표시.
    """
    omni.kit.commands.execute("CreatePrimWithDefaultXform", prim_type="Xform",    prim_path="/World/Mission")
    omni.kit.commands.execute("CreatePrimWithDefaultXform", prim_type="Cylinder", prim_path="/World/Mission/Basecamp")
    _safe_xform(stage, "/World/Mission/Basecamp",
                translate=(cx, cy, 0.01), scale=(visual_radius, visual_radius, 0.01))
    _apply_color(stage, "/World/Mission/Basecamp", (0.05, 0.9, 0.15))
    print(f"[Scene] 베이스캠프 마커 생성: ({cx:.1f}, {cy:.1f})")


def get_basecamp_center(raw_env) -> tuple:
    """지형 높이맵 중심 좌표를 베이스캠프 위치로 반환."""
    terrain = raw_env.scene.terrain
    hm = terrain._terrainManager._heightmap_manager
    return (hm.min_x + hm.max_x) / 2.0, (hm.min_y + hm.max_y) / 2.0