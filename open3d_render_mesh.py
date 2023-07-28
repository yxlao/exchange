import open3d as o3d
import numpy as np

def render_geometries_mesh_open3d_K_T(geometries,
                                      K,
                                      T,
                                      height,
                                      width,
                                      visible=False):
    """
    Render a mesh using Open3D legacy visualizer. This requires a display. 
    It might not work on a server.

    Args:
        geometries: a list of Open3d geometries.
        K: (3, 3) np.ndarray camera intrinsic.
        T: (4, 4) np.ndarray camera extrinsic.
        height: int image height.
        width: int image width.
        visible: bool whether to show the window. Your machine must have a monitor.

    Returns:
        image: (H, W, 3) float32 np.ndarray image.
    """
    o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=width,
        height=height,
        fx=K[0, 0],
        fy=K[1, 1],
        cx=K[0, 2],
        cy=K[1, 2],
    )

    o3d_extrinsic = T
    o3d_camera = o3d.camera.PinholeCameraParameters()
    o3d_camera.intrinsic = o3d_intrinsic
    o3d_camera.extrinsic = o3d_extrinsic

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=visible)
    for geometry in geometries:
        vis.add_geometry(geometry)
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(o3d_camera, allow_arbitrary=True)
    vis.get_render_option().point_size = 1.0
    for geometry in geometries:
        vis.update_geometry(geometry)
    vis.poll_events()
    vis.update_renderer()
    buffer = vis.capture_screen_float_buffer()
    vis.destroy_window()

    return np.array(buffer)

def main():
    # Example
    # mesh = o3d.io.read_triangle_mesh("xxx")
    # K = np.array(xxx)
    # T = np.array(xxx)
    # image = render_geometries_mesh_open3d_K_T([mesh], K, T, height, width)
    
