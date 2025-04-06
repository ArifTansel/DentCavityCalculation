import trimesh
import open3d as o3d
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RANSACRegressor
from scipy.spatial import ConvexHull

# Load the tooth STL model
mesh_trimesh = trimesh.load_mesh("heh.stl")

# Get vertices, faces, and normals
vertices = np.array(mesh_trimesh.vertices)
faces = np.array(mesh_trimesh.faces)
normals = np.array(mesh_trimesh.vertex_normals)

# Compute Mean Curvature
mean_curvature = trimesh.curvature.discrete_mean_curvature_measure(mesh_trimesh, mesh_trimesh.vertices, radius=2)

# Define cavity detection threshold (14th percentile)
curvature_threshold = np.percentile(mean_curvature, 14)
cavity_indices = np.where(mean_curvature < curvature_threshold)[0]  # Indices of cavity

# Convert to Open3D for visualization
mesh_o3d = o3d.geometry.TriangleMesh()
mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
mesh_o3d.compute_vertex_normals()

# Color assignment: normal surface = gray, cavity = red
colors = np.ones((vertices.shape[0], 3)) * 0.7  # Light gray
colors[cavity_indices] = [1, 0, 0]  # Mark cavity region as RED
mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(colors)

### STEP 1: Extract ROI and Ensure Isolation
roi_vertices = vertices[cavity_indices]  # Extract cavity region
roi_pcd = o3d.geometry.PointCloud()
roi_pcd.points = o3d.utility.Vector3dVector(roi_vertices)

# Ensure the cavity is a separate cluster (if needed)
labels = np.array(roi_pcd.cluster_dbscan(eps=0.5, min_points=5, print_progress=False))
largest_cluster = np.argmax(np.bincount(labels[labels >= 0]))  # Find the biggest cluster
roi_vertices = roi_vertices[labels == largest_cluster]  # Keep only the main cavity points

### STEP 2: Compute Convex Hull to Remove Outliers
hull = ConvexHull(roi_vertices)
roi_vertices = roi_vertices[hull.vertices]  # Keep only convex hull points

### STEP 3: Compute PCA for Better Orientation
pca = PCA(n_components=3)
pca.fit(roi_vertices)
principal_axes = pca.components_  # New coordinate system
obb_center = np.mean(roi_vertices, axis=0)  # Cavity center

# Transform points into PCA space
aligned_points = (roi_vertices - obb_center) @ principal_axes.T
min_bound = np.min(aligned_points, axis=0)
max_bound = np.max(aligned_points, axis=0)
obb_extent = max_bound - min_bound  # Get (length, height, depth)

### STEP 4: Surface Smoothness Analysis using RANSAC Plane Fitting
X = roi_vertices[:, :2]  # Use x and y coordinates
y = roi_vertices[:, 2]   # Use z coordinate as dependent variable

ransac = RANSACRegressor()
ransac.fit(X, y)

# Predict the expected Z values (smooth plane)
y_pred = ransac.predict(X)

# Compute residuals (differences between actual and predicted Z)
residuals = np.abs(y - y_pred)

# Measure surface roughness using standard deviation and RMSE
std_dev = np.std(residuals)
rmse = np.sqrt(np.mean(residuals**2))

print(f"Surface Roughness (Std Dev): {std_dev:.5f}")
print(f"Surface Roughness (RMSE): {rmse:.5f}")

# Visualize the roughness analysis by coloring residuals
smoothness_colors = np.zeros((roi_vertices.shape[0], 3))  # Default: black
max_residual = np.max(residuals)

for i in range(len(residuals)):
    color_intensity = residuals[i] / max_residual  # Normalize residuals
    smoothness_colors[i] = [color_intensity, 1 - color_intensity, 0]  # Gradient from green to red

smoothness_pcd = o3d.geometry.PointCloud()
smoothness_pcd.points = o3d.utility.Vector3dVector(roi_vertices)
smoothness_pcd.colors = o3d.utility.Vector3dVector(smoothness_colors)

### STEP 5: Create Visualization Vectors 
def create_vector(center, direction, length, color, offset_ratio=0.5):
    """ Creates a 3D vector (line) that starts inside the OBB and extends outward. """
    start = center - (direction * (length * offset_ratio))  # Start inside OBB
    end = start + (direction * length)  # Extend outward
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector([start, end])
    line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
    line_set.colors = o3d.utility.Vector3dVector([color])
    
    return line_set

# Principal axes
axis_x, axis_y, axis_z = principal_axes  # Align with cavity

# Create dimension vectors
vector_length = create_vector(obb_center, axis_x, obb_extent[0], [1, 0, 0])  # Red
vector_height = create_vector(obb_center, axis_y, obb_extent[1], [0, 1, 0])  # Green
vector_depth = create_vector(obb_center, axis_z, obb_extent[2], [0, 0, 1])  # Blue

### STEP 6: Visualize
o3d.visualization.draw_geometries([mesh_o3d, smoothness_pcd, vector_length, vector_height, vector_depth])
o3d.visualization.draw_geometries([mesh_o3d, smoothness_pcd])
o3d.visualization.draw_geometries([smoothness_pcd])