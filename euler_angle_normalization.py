from pathlib import Path
import open3d as o3d
from scipy.spatial.transform import Rotation
import numpy as np

import json
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation
from typing import Tuple, List
import copy


def main():
    # rotmat = Rotation.from_matrix(R_bbox2cam)
    # euler_angle = rotmat.as_euler('xyz', degrees=True).tolist()
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])

    # Cube
    cube = o3d.geometry.TriangleMesh.create_box(width=1, height=2, depth=3)
    cube = o3d.geometry.LineSet.create_from_triangle_mesh(cube)
    cube.paint_uniform_color([1, 0, 0])

    # Apply translation, z axis, 5 units
    mat_translation = np.eye(4)
    mat_translation[:3, 3] = [0, 0, 5]
    cube.transform(mat_translation)

    o3d.visualization.draw_geometries([cube, axes])


def normalize_euler_angles(a, b, c) -> Tuple[float, float, float]:
    """
    Returns normalized euler angles such that -90 < a <= 90, -90 < b <= 90,
    -90 < c <= 90.
    Return the normalized euler angles such that rectangular box after the
    new rotation can overlap with the original box. Note that they are not
    equivalent, only the bbox is overlapping.
    This is due to the fact that for rectangles, if we rotate around one
    axis for 180 degrees, the rectangle still overlaps with the original one,
    although the euler angles are different.
    Note that naively plus minus 180 will not work, as the euler angles are
    tightly coupled. We need to be very careful for each case.
    """
    # Create a rotation matrix from the input Euler angles
    rotation = Rotation.from_euler("XYZ", [a, b, c], degrees=True)

    # Generate all possible equivalent rotations by applying 180° rotations
    # around principal axes

    # Original rotation matrix
    rot_original = rotation.as_matrix()

    # Define 180° rotation matrices around each axis
    rot_x_180 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    rot_y_180 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    rot_z_180 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])

    # Generate all possible equivalent rotation matrices
    equivalent_matrices = [
        rot_original,
        rot_original @ rot_x_180,
        rot_original @ rot_y_180,
        rot_original @ rot_z_180,
        rot_original @ rot_x_180 @ rot_y_180,
        rot_original @ rot_x_180 @ rot_z_180,
        rot_original @ rot_y_180 @ rot_z_180,
        rot_original @ rot_x_180 @ rot_y_180 @ rot_z_180
    ]

    # Try different rotation sequences to find one that gives angles in the desired range
    rotation_sequences = ["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"]

    best_angles = None

    # First try all combinations of matrices and sequences
    for matrix in equivalent_matrices:
        for seq in rotation_sequences:
            try:
                rot = Rotation.from_matrix(matrix)
                angles = rot.as_euler(seq, degrees=True)

                # Check if all angles are already in the target range
                if all(-90 < angle <= 90 for angle in angles):
                    # Convert back to XYZ for consistency
                    if seq != "XYZ":
                        rot_consistent = Rotation.from_euler(seq, angles, degrees=True)
                        angles = rot_consistent.as_euler("XYZ", degrees=True)

                    # Verify all angles are still in the target range
                    if all(-90 < angle <= 90 for angle in angles):
                        best_angles = tuple(angles)
                        break
            except:
                continue

        if best_angles is not None:
            break

    # If we found a solution, return it
    if best_angles is not None:
        return best_angles

    # If we couldn't find a solution with the direct approach, try manually normalizing
    # Create a copy of the original rotation
    rot_copy = Rotation.from_euler("XYZ", [a, b, c], degrees=True)

    # Try all possible equivalent rotations and manually adjust angles
    for matrix in equivalent_matrices:
        try:
            rot = Rotation.from_matrix(matrix)
            a_eq, b_eq, c_eq = rot.as_euler("XYZ", degrees=True)

            # Normalize each angle into (-90, 90] range
            adjustments = []

            # Try different adjustments
            for a_adj in [0, 180, -180]:
                for b_adj in [0, 180, -180]:
                    for c_adj in [0, 180, -180]:
                        new_a = a_eq + a_adj
                        new_b = b_eq + b_adj
                        new_c = c_eq + c_adj

                        # Ensure angles are in the primary range
                        while new_a <= -180: new_a += 360
                        while new_a > 180: new_a -= 360
                        while new_b <= -180: new_b += 360
                        while new_b > 180: new_b -= 360
                        while new_c <= -180: new_c += 360
                        while new_c > 180: new_c -= 360

                        # Check if in target range
                        if -90 < new_a <= 90 and -90 < new_b <= 90 and -90 < new_c <= 90:
                            # Verify this is an equivalent rotation
                            test_rot = Rotation.from_euler("XYZ", [new_a, new_b, new_c], degrees=True)
                            orig_points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])

                            rotated_1 = rot.apply(orig_points)
                            rotated_2 = test_rot.apply(orig_points)

                            # Check if the rotations are approximately equivalent for box corners
                            if np.allclose(np.abs(rotated_1), np.abs(rotated_2), atol=1e-6):
                                adjustments.append((new_a, new_b, new_c))

            if adjustments:
                # Use the first valid adjustment
                best_angles = adjustments[0]
                break
        except:
            continue

    # If still no solution found, try using the original angles but with normalization
    if best_angles is None:
        # Start with original angles
        a_norm, b_norm, c_norm = a, b, c

        # Normalize to primary range first
        while a_norm <= -180: a_norm += 360
        while a_norm > 180: a_norm -= 360
        while b_norm <= -180: b_norm += 360
        while b_norm > 180: b_norm -= 360
        while c_norm <= -180: c_norm += 360
        while c_norm > 180: c_norm -= 360

        # Try to get all angles in (-90, 90] range by applying 180° rotations
        if a_norm > 90:
            a_norm -= 180
            b_norm = -b_norm
            c_norm = c_norm + 180 if c_norm <= 0 else c_norm - 180
        elif a_norm <= -90:
            a_norm += 180
            b_norm = -b_norm
            c_norm = c_norm + 180 if c_norm <= 0 else c_norm - 180

        if b_norm > 90:
            b_norm -= 180
            a_norm = -a_norm
            c_norm = c_norm + 180 if c_norm <= 0 else c_norm - 180
        elif b_norm <= -90:
            b_norm += 180
            a_norm = -a_norm
            c_norm = c_norm + 180 if c_norm <= 0 else c_norm - 180

        if c_norm > 90:
            c_norm -= 180
            a_norm = -a_norm
            b_norm = -b_norm
        elif c_norm <= -90:
            c_norm += 180
            a_norm = -a_norm
            b_norm = -b_norm

        # Ensure angles are in primary range again
        while a_norm <= -180: a_norm += 360
        while a_norm > 180: a_norm -= 360
        while b_norm <= -180: b_norm += 360
        while b_norm > 180: b_norm -= 360
        while c_norm <= -180: c_norm += 360
        while c_norm > 180: c_norm -= 360

        best_angles = (a_norm, b_norm, c_norm)

    # Validate angles
    a_result, b_result, c_result = best_angles

    # Ensure all angles are in the desired range
    if not -90 < a_result <= 90:
        raise ValueError(f"a_result is not in the range (-90, 90]: {a_result}")
    if not -90 < b_result <= 90:
        raise ValueError(f"b_result is not in the range (-90, 90]: {b_result}")
    if not -90 < c_result <= 90:
        raise ValueError(f"c_result is not in the range (-90, 90]: {c_result}")

    return best_angles


def normalize_bbox(bbox: np.ndarray) -> np.ndarray:
    """
    Normalize the bounding box elment of the bbox.
    bbox: [x, y, z, w, h, l, euler_a, euler_b, euler_c]
    """
    assert bbox.ndim == 1 and len(bbox) == 9
    normalized_bbox = copy.deepcopy(bbox)
    normalized_bbox[-3:] = normalize_euler_angles(*bbox[-3:])
    return normalized_bbox


def test_normalize_bbox(bbox_a: np.ndarray, visualize=False):
    # - euler rotate around x: rotation from y to z
    # - euler rotate around y: rotation from z to x
    # - euler rotate around z: rotation from x to y
    bbox_b = normalize_bbox(bbox_a)
    print(f"bbox_a: {bbox_a}")
    print(f"bbox_b: {bbox_b}")

    # Bounding box a
    xyz_a = bbox_a[:3]
    extent_a = bbox_a[3:6]
    rotation_a = Rotation.from_euler("XYZ", bbox_a[6:], degrees=True).as_matrix()
    box_a = o3d.geometry.OrientedBoundingBox(
        center=xyz_a, R=np.array(rotation_a), extent=extent_a
    )
    box_lines_a = o3d.geometry.LineSet.create_from_oriented_bounding_box(box_a)
    box_lines_a.paint_uniform_color([0.5, 0, 0])

    # Bounding box b
    xyz_b = bbox_b[:3]
    whl_b = bbox_b[3:6]
    rotation_b = Rotation.from_euler("XYZ", bbox_b[6:], degrees=True).as_matrix()
    box_b = o3d.geometry.OrientedBoundingBox(
        center=xyz_b, R=np.array(rotation_b), extent=whl_b
    )
    box_lines_b = o3d.geometry.LineSet.create_from_oriented_bounding_box(box_b)
    box_lines_b.paint_uniform_color([0, 0.5, 0])

    # Check if the bbox_a's points overlap with bbox_b's points
    points_a = np.asarray(box_lines_a.points)
    points_b = np.asarray(box_lines_b.points)
    points_a_sorted = np.array(sorted(points_a.tolist()))
    points_b_sorted = np.array(sorted(points_b.tolist()))
    if not np.allclose(points_a_sorted, points_b_sorted, atol=1e-3, rtol=1e-3):
        raise ValueError("bbox_a's points do not overlap with bbox_b's points")
    else:
        print("bbox_a's points overlap with bbox_b's points")

    # Visulized for comparision
    if visualize:
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        o3d.visualization.draw_geometries([axes, box_lines_a, box_lines_b])


def generate_test_cases() -> List[np.ndarray]:
    """
    Generate test cases for euler angle normalization.
    For each angle a, b, c:
    - below -90
    - within [-90, 90]
    - above 90
    Returns:
        List of test cases, each case is a 9D array:
        [x, y, z, w, h, l, euler_a, euler_b, euler_c]
    """
    # Base bbox parameters (arbitrary but reasonable values)
    x, y, z = 1.0, 1.0, 1.0
    w, h, l = 0.5, 0.3, 0.7

    # Angle ranges for testing
    angle_ranges = {
        'below_90': -100,  # below -90
        'within': 20,      # within [-90, 90]
        'above_90': 115    # above 90
    }

    test_cases = []
    for a_range in ['below_90', 'within', 'above_90']:
        for b_range in ['below_90', 'within', 'above_90']:
            for c_range in ['below_90', 'within', 'above_90']:
                a = angle_ranges[a_range]
                b = angle_ranges[b_range]
                c = angle_ranges[c_range]

                test_case = np.array([x, y, z, w, h, l, a, b, c])
                test_cases.append(test_case)

    return test_cases


def main():
    # Original test cases
    print("Running original test cases:")
    print("-" * 50)
    bbox_a = np.array([2.0, 0.0, 0.0, 0.2, 0.4, 0.8, 95, 100, 105])
    test_normalize_bbox(bbox_a=bbox_a, visualize=False)
    print()

    bbox_a = np.array([0.94, 0.59, 1.78, 0.88, 0.5, 0.88, -179.94, -68.27, 179.94])
    test_normalize_bbox(bbox_a=bbox_a, visualize=False)
    print()

    bbox_a = np.array([5.0, 0.0, 0.0, 0.2, 0.4, 0.8, -179.94, -68.27, 179.94])
    test_normalize_bbox(bbox_a=bbox_a, visualize=False)
    print()

    # Comprehensive test cases
    print("\nRunning comprehensive test cases:")
    print("-" * 50)
    test_cases = generate_test_cases()
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest case {i}/27:")
        print(f"Input angles (a, b, c): {test_case[6:]}")
        test_normalize_bbox(test_case, visualize=False)
        print("-" * 30)


if __name__ == "__main__":
    main()
