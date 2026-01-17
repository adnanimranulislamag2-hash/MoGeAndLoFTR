import open3d as o3d
import sys

ply_path = sys.argv[1] if len(sys.argv) > 1 else "output/ test101_pointcloud.ply"

print(f"Loading {ply_path}...")
pcd = o3d.io.read_point_cloud(ply_path)
print(f"Point cloud has {len(pcd.points)} points")

# Export to different formats
base_name = ply_path.replace('.ply', '')

# 1. Export as PLY (copy to a location you can download)
print(f"Saving to {base_name}_export.ply")
o3d.io.write_point_cloud(f"{base_name}_export.ply", pcd)

# 2. Export point cloud info
print(f"\nPoint Cloud Information:")
print(f"  Total points: {len(pcd.points)}")
print(f"  Has colors: {pcd.has_colors()}")
print(f"  Has normals: {pcd.has_normals()}")

import numpy as np
points = np.asarray(pcd.points)
print(f"  Bounding box:")
print(f"    X: [{points[:,0].min():.3f}, {points[:,0].max():.3f}]")
print(f"    Y: [{points[:,1].min():.3f}, {points[:,1].max():.3f}]")
print(f"    Z: [{points[:,2].min():.3f}, {points[:,2].max():.3f}]")

print("\n✅ Done! Download the PLY file to view on your local machine.")