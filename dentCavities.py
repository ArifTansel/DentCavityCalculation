import trimesh
import open3d as o3d
import numpy as np
from sklearn.cluster import KMeans
# Load the tooth STL model using Trimesh
mesh_trimesh = trimesh.load_mesh("sinifBirNumaraAlti.stl")

# Get vertices, faces, and normals
vertices = np.array(mesh_trimesh.vertices)
faces = np.array(mesh_trimesh.faces)
normals = np.array(mesh_trimesh.vertex_normals)

# Compute Mean Curvature using Trimesh
mean_curvature = trimesh.curvature.discrete_mean_curvature_measure(mesh_trimesh, mesh_trimesh.vertices, radius=2)

# Focus on negative curvature for cavity detection
# Negative curvature indicates concave regions


cavity_indices = np.where(mean_curvature < 0.4)[0]  # Select all vertices with negative curvature
outline_indices = np.where((mean_curvature > 3.0))[0]
cavity_faces = faces[cavity_indices]

cavity_vertices = vertices[np.unique(cavity_indices)]


mesh_o3d = o3d.geometry.TriangleMesh()
mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
mesh_o3d.compute_vertex_normals()

colors = np.ones((vertices.shape[0], 3)) * 0.7  # Light gray for normal surface
colors[outline_indices] = [1, 0, 0]  # Red for cavities
colors[cavity_indices] = [0, 1, 0]  # Red for cavities
mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(colors)

# cavity mesh 
mesh_cavity = o3d.geometry.TriangleMesh()
mesh_cavity.vertices = o3d.utility.Vector3dVector(cavity_vertices)
mesh_cavity.triangles = o3d.utility.Vector3iVector(cavity_faces)



# min_z = np.min(cavity_vertices[:, 2])
# max_z = np.max(cavity_vertices[:, 2])
# cavity_depth = max_z - min_z  # Derinlik (Z eksenindeki fark)

# cavity_centroid = np.mean(cavity_vertices, axis=0)


# min_z_point = [cavity_centroid[0], cavity_centroid[1], min_z]
# max_z_point = [cavity_centroid[0], cavity_centroid[1], max_z]

# # **Çizgiyi tanımlama**
# line_set = o3d.geometry.LineSet()
# line_set.points = o3d.utility.Vector3dVector([min_z_point, max_z_point])
# line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
# line_set.colors = o3d.utility.Vector3dVector([[0, 0, 1]])  # Mavi çizgi


# Visualize tooth with cavity detection
o3d.visualization.draw_geometries([mesh_cavity])