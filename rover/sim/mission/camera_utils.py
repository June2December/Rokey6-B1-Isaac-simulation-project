"""
camera_utils.py — 멀티 뷰포트 카메라 설정 유틸리티
각 로봇에 카메라를 부착하고 별도 뷰포트 창을 띄웁니다.
"""
import omni.kit.commands
from pxr import UsdGeom, Gf


def attach_custom_camera(stage, env_idx: int, view_type: str):
    """
    로봇 몸체에 카메라 부착.

    Args:
        view_type: "1P" (1인칭) 또는 "3P" (3인칭)
    """
    cam_path = f"/World/envs/env_{env_idx}/Robot/Body/Camera_{view_type}"
    omni.kit.commands.execute("CreatePrimWithDefaultXform", prim_type="Camera", prim_path=cam_path)

    cam_prim = stage.GetPrimAtPath(cam_path)
    if not cam_prim.IsValid():
        return None

    if view_type == "1P":
        translate, rotate, focal = (-1.5, 0.0, 0.4),  (90.0, 0.0, -90.0), 15.0
    else:
        translate, rotate, focal = (-2.5, 0.0, 1.2), (90.0, 0.0,  90.0), 24.0

    xf = UsdGeom.Xformable(cam_prim)
    xf.ClearXformOpOrder()
    t_op = xf.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
    t_op.Set(Gf.Vec3d(*translate))
    r_op = xf.AddRotateXYZOp(precision=UsdGeom.XformOp.PrecisionDouble)
    r_op.Set(Gf.Vec3d(*rotate))
    xf.SetXformOpOrder([t_op, r_op])

    cam_geom = UsdGeom.Camera(cam_prim)
    cam_geom.GetHorizontalApertureAttr().Set(36.0)
    cam_geom.GetFocalLengthAttr().Set(focal)
    return cam_path


def setup_dual_viewports(stage):
    """
    뷰포트 창 2개를 띄워 Robot0(1인칭), Robot1(1인칭) 시점 연결.
    메인 뷰포트는 전체 관전 모드로 유지.
    """
    cam0 = attach_custom_camera(stage, env_idx=0, view_type="1P")
    cam1 = attach_custom_camera(stage, env_idx=1, view_type="1P")

    try:
        import omni.kit.viewport.utility as vp_util
        main_vp = vp_util.get_active_viewport()
        if main_vp:
            main_vp.set_active_camera("/OmniverseKit_Persp")

        vp0 = vp_util.create_viewport_window("Robot 0 (1인칭)", width=500, height=300)
        if vp0 and cam0:
            vp0.viewport_api.set_active_camera(cam0)

        vp1 = vp_util.create_viewport_window("Robot 1 (1인칭)", width=500, height=300)
        if vp1 and cam1:
            vp1.viewport_api.set_active_camera(cam1)

        print("[Camera] 멀티 뷰포트 생성 완료!")
    except Exception as e:
        print(f"[Camera] 멀티 뷰포트 생성 실패 (무시 가능): {e}")