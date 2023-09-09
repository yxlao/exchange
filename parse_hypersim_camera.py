import os
from pathlib import Path

import camtools as ct
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.set_printoptions(suppress=True, precision=4)

_pwd = Path(__file__).parent.absolute()
_project_root = _pwd
_data_root = _project_root / "data"


def main():
    metadata_camera_parameters_csv_file = (
        _project_root / "contrib/mikeroberts3000/metadata_camera_parameters.csv"
    )

    scene_name = "ai_001_001"
    camera_name = "cam_00"
    frame_id = 0

    # Project all "sofa" vertices into our image
    camera_dir = _data_root / "scenes" / scene_name / "_detail" / camera_name
    img_dir = _data_root / "scenes" / scene_name / "images"

    # Read parameters from csv file
    df_camera_parameters = pd.read_csv(
        metadata_camera_parameters_csv_file, index_col="scene_name"
    )

    df_ = df_camera_parameters.loc[scene_name]

    width_pixels = int(df_["settings_output_img_width"])
    height_pixels = int(df_["settings_output_img_height"])

    M_proj = np.array(
        [
            [df_["M_proj_00"], df_["M_proj_01"], df_["M_proj_02"], df_["M_proj_03"]],
            [df_["M_proj_10"], df_["M_proj_11"], df_["M_proj_12"], df_["M_proj_13"]],
            [df_["M_proj_20"], df_["M_proj_21"], df_["M_proj_22"], df_["M_proj_23"]],
            [df_["M_proj_30"], df_["M_proj_31"], df_["M_proj_32"], df_["M_proj_33"]],
        ]
    )
    mesh_positions_world = np.array(np.random.rand(10000, 3) * 50.0)

    # Verify that we are projecting points to their correct screen-space positions
    camera_positions_hdf5_file = os.path.join(
        camera_dir, "camera_keyframe_positions.hdf5"
    )
    camera_orientations_hdf5_file = os.path.join(
        camera_dir, "camera_keyframe_orientations.hdf5"
    )

    depth_meters_hdf5_file = os.path.join(
        img_dir,
        "scene_" + camera_name + "_geometry_hdf5",
        "frame.%04d.depth_meters.hdf5" % frame_id,
    )

    with h5py.File(camera_positions_hdf5_file, "r") as f:
        camera_positions = f["dataset"][:]
    with h5py.File(camera_orientations_hdf5_file, "r") as f:
        camera_orientations = f["dataset"][:]

    with h5py.File(depth_meters_hdf5_file, "r") as f:
        depth_meters = f["dataset"][:].astype(np.float32)

    # matrix to map to integer screen coordinates from normalized device coordinates
    M_screen_from_ndc = np.array(
        [
            [0.5 * (width_pixels - 1), 0, 0, 0.5 * (width_pixels - 1)],
            [0, -0.5 * (height_pixels - 1), 0, 0.5 * (height_pixels - 1)],
            [0, 0, 0.5, 0.5],
            [0, 0, 0, 1.0],
        ]
    )

    # get position and rotation matrix for Hypersim image
    camera_position_world = camera_positions[frame_id]
    R_world_from_cam = camera_orientations[frame_id]

    t_world_from_cam = np.array(camera_position_world).T
    R_cam_from_world = np.array(R_world_from_cam).T
    t_cam_from_world = -R_cam_from_world @ t_world_from_cam

    M_cam_from_world = np.eye(4)
    M_cam_from_world[:3, :3] = R_cam_from_world
    M_cam_from_world[:3, 3] = t_cam_from_world

    num_points = mesh_positions_world.shape[0]

    T = ct.convert.T_blender_to_pinhole(M_cam_from_world)
    K_opengl = M_screen_from_ndc @ M_proj
    fx = K_opengl[0, 0]
    fy = -K_opengl[1, 1]
    cx = -K_opengl[0, 2]
    cy = -K_opengl[1, 2]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    P_screen = ct.project.points_to_pixel(mesh_positions_world, K, T)

    plt.figure(figsize=(18, 18))

    plt.subplot(211)
    plt.imshow(depth_meters)
    plt.title("Depth image without projected points")

    plt.subplot(212)
    plt.imshow(depth_meters)
    plt.title("Depth image with projected points")
    plt.scatter(P_screen[:, 0], P_screen[:, 1], color="red", s=0.05)
    plt.show()


if __name__ == "__main__":
    main()
