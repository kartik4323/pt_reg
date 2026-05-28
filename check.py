import open3d as o3d
import numpy as np

points = []

with open(r"C:\Users\KARTIK PANDEY\OneDrive\Desktop\hero_python\fragment_compact\outputs\test_reconstruction\reconstructed.obj", "r") as f:
    for line in f:
        if line.startswith("v "):
            _, x, y, z = line.split()
            points.append([float(x), float(y), float(z)])

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.array(points))

o3d.visualization.draw_geometries([pcd])