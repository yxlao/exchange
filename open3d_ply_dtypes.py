import open3d as o3d
import numpy as np


def main():
    dataset = o3d.data.PLYPointCloud()
    print(f"PLY path: {dataset.path}")

    # Read with legacy. Legacy point cloud alway has double color.
    pcd = o3d.io.read_point_cloud(dataset.path)
    colors = np.array(pcd.colors)
    print("Read using legacy point cloud")
    print(f"dtype: {colors.dtype}")  # float64
    print(f"min, max: {colors.min()}, {colors.max()}")  # 0.0, 1.0

    # Write with legacy. Legacy point cloud is alway written as uint8 color.
    # Run `head -n 15 pcd_legacy.ply` to see the header
    print("Write using legacy point cloud")
    o3d.io.write_point_cloud("pcd_legacy.ply", pcd)  # saved as uint8

    # Read with tensor
    pcd = o3d.t.io.read_point_cloud(dataset.path)
    colors = pcd.point.colors.numpy()
    print("Read using tensor point cloud")
    print(f"dtype: {colors.dtype}")  # uint8
    print(f"min, max: {colors.min()}, {colors.max()}")  # 0, 255

    # Write with tensor, uint8
    # Run `head -n 15 pcd_tensor_uint8.ply` to see the header
    o3d.t.io.write_point_cloud("pcd_tensor_uint8.ply", pcd)  # saved as uint8
    pcd_read = o3d.t.io.read_point_cloud("pcd_tensor_uint8.ply")
    colors = pcd_read.point.colors.numpy()
    print("Read using tensor point cloud uint8")
    print(f"dtype: {colors.dtype}")  # uint8
    print(f"min, max: {colors.min()}, {colors.max()}")  # 0, 255

    # Write with tensor, float32
    # Run `head -n 15 pcd_tensor_float32.ply` to see the header
    pcd.point.colors = colors.astype(np.float32) / 255.0
    o3d.t.io.write_point_cloud("pcd_tensor_float32.ply", pcd)  # float32
    pcd_read = o3d.t.io.read_point_cloud("pcd_tensor_float32.ply")
    colors = pcd_read.point.colors.numpy()
    print("Read using tensor point cloud float32")
    print(f"dtype: {colors.dtype}")  # float32
    print(f"min, max: {colors.min()}, {colors.max()}")  # 0.0, 1.0


if __name__ == "__main__":
    main()
