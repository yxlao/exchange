from pathlib import Path
import numpy as np
import open3d as o3d
from pathlib import Path


def main():
    # Create source mesh
    pwd = Path(__file__).parent
    mesh_src = o3d.io.read_triangle_mesh(str(pwd / "cube.obj"),
                                         enable_post_processing=True)

    # Get all elements from src mesh
    vertices = np.asarray(mesh_src.vertices)
    triangles = np.asarray(mesh_src.triangles)
    textures = np.asarray(mesh_src.textures[0])
    triangle_uvs = np.asarray(mesh_src.triangle_uvs)
    triangle_material_ids = np.zeros((len(triangles), ), dtype=np.int32)

    # Create dst mesh
    mesh_dst = o3d.geometry.TriangleMesh()
    mesh_dst.vertices = o3d.utility.Vector3dVector(vertices)
    mesh_dst.triangles = o3d.utility.Vector3iVector(triangles)
    mesh_dst.textures = [o3d.geometry.Image(textures)]
    mesh_dst.triangle_uvs = o3d.utility.Vector2dVector(triangle_uvs)
    mesh_dst.triangle_material_ids = o3d.utility.IntVector(
        triangle_material_ids)

    # Visualize
    mesh_dst.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh_dst])


if __name__ == '__main__':
    main()
