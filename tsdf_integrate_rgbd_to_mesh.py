import open3d as o3d
from tqdm import tqdm


def integrate_rgbd_to_mesh(
    Ks,
    Ts,
    im_depths,
    im_colors,
    voxel_size,
):
    """
    Integrate RGBD images into a TSDF volume and extract a mesh.

    Args:
        Ks: (N, 3, 3) camera intrinsics.
        Ts: (N, 4, 4) camera extrinsics.
        im_depths: (N, H, W) depth images, already in world scale.
        im_colors: (N, H, W, 3) color images, float range in [0, 1].
        voxel_size: TSDF voxel size, in meters, e.g. 3 / 512.
    """
    num_images = len(Ks)
    if (
        len(Ts) != num_images
        or len(im_depths) != num_images
        or len(im_colors) != num_images
    ):
        raise ValueError("Ks, Ts, im_depths, im_colors must have the same length.")

    # Constants.
    trunc_voxel_multiplier = 8.0
    sdf_trunc = trunc_voxel_multiplier * voxel_size

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    for K, T, im_depth, im_color in tqdm(
        zip(Ks, Ts, im_depths, im_colors),
        total=len(Ks),
        desc="Integrating RGBD frames",
    ):
        im_color_uint8 = (im_color * 255).astype(np.uint8)
        im_depth_uint16 = (im_depth * 1000).astype(np.uint16)
        im_color_o3d = o3d.geometry.Image(im_color_uint8)
        im_depth_o3d = o3d.geometry.Image(im_depth_uint16)
        im_rgbd_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(
            im_color_o3d,
            im_depth_o3d,
            depth_scale=1000.0,
            depth_trunc=10.0,
            convert_rgb_to_intensity=False,
        )
        o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=im_depth.shape[1],
            height=im_depth.shape[0],
            fx=K[0, 0],
            fy=K[1, 1],
            cx=K[0, 2],
            cy=K[1, 2],
        )
        o3d_extrinsic = T
        volume.integrate(
            im_rgbd_o3d,
            o3d_intrinsic,
            o3d_extrinsic,
        )

    mesh = volume.extract_triangle_mesh()
    return mesh
