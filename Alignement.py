import trimesh
import open3d as o3d
import numpy as np
import copy

# Load the STL files using Trimesh
def load_with_trimesh(file_path):
    # Load the mesh with trimesh
    tri_mesh = trimesh.load(file_path)
    return tri_mesh

# Convert Trimesh to Open3D mesh
def trimesh_to_open3d(tri_mesh):
    # Extract vertices and faces
    vertices = np.array(tri_mesh.vertices)
    faces = np.array(tri_mesh.faces)
    
    # Create Open3D mesh
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
    
    # Compute normals
    o3d_mesh.compute_vertex_normals()
    o3d_mesh.compute_triangle_normals()
    
    return o3d_mesh

# Load the STL files with Trimesh
source_trimesh = load_with_trimesh("input/Master.stl")
target_trimesh = load_with_trimesh("input/Master.stl")

# Convert to Open3D meshes for visualization and alignment
source_mesh = trimesh_to_open3d(source_trimesh)
target_mesh = trimesh_to_open3d(target_trimesh)

# Create point clouds from the meshes for alignment
source_pcd = source_mesh.sample_points_uniformly(number_of_points=5000)
target_pcd = target_mesh.sample_points_uniformly(number_of_points=5000)

# Initial alignment using point-to-point ICP
def initial_alignment(source, target):
    trans_init = np.identity(4)
    
    # Apply ICP
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, 
        max_correspondence_distance=10,
        init=trans_init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )
    
    return reg_p2p.transformation

# Fine alignment using point-to-plane ICP for better accuracy
def fine_alignment(source, target, initial_transform):
    # Apply the initial transformation
    source_temp = copy.deepcopy(source)
    source_temp.transform(initial_transform)
    
    # Estimate normals if they don't exist
    if not source_temp.has_normals():
        source_temp.estimate_normals()
    if not target.has_normals():
        target.estimate_normals()
    
    # Apply point-to-plane ICP
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source_temp, target, 
        max_correspondence_distance=1,
        init=np.identity(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )
    
    # Combine transformations
    combined_transform = np.matmul(reg_p2l.transformation, initial_transform)
    
    return combined_transform

# Get the initial alignment
initial_transform = initial_alignment(source_pcd, target_pcd)
# Get the fine alignment
final_transform = fine_alignment(source_pcd, target_pcd, initial_transform)

# Apply the transformation to the source trimesh
aligned_source_trimesh = source_trimesh.copy()
aligned_source_trimesh.apply_transform(final_transform)

# Convert the aligned trimesh back to Open3D for visualization
aligned_source_mesh = trimesh_to_open3d(aligned_source_trimesh)

# Create point clouds with more points for accurate comparison
aligned_source_pcd = aligned_source_mesh.sample_points_uniformly(number_of_points=50000)
target_pcd_dense = target_mesh.sample_points_uniformly(number_of_points=50000)

# Function to compute distances between point clouds using Open3D's distance computation
def compute_distance_between_point_cloud_and_mesh(source_pcd, target_mesh):
    # Sample points from the target mesh to create a point cloud
    target_pcd = target_mesh.sample_points_uniformly(number_of_points=100000)
    
    # Build KD-tree for the target point cloud
    target_tree = o3d.geometry.KDTreeFlann(target_pcd)
    
    # Compute distances from source points to target mesh
    source_points = np.asarray(source_pcd.points)
    distances = []
    
    for point in source_points:
        # Find the nearest neighbor in the target point cloud
        k, idx, dist = target_tree.search_knn_vector_3d(point, 1)
        distances.append(np.sqrt(dist[0]))  # Take the square root to get the actual distance
    
    return np.array(distances)

# Compute distances from aligned source to target
distances_source_to_target = compute_distance_between_point_cloud_and_mesh(aligned_source_pcd, target_mesh)

# Compute distances from target to aligned source
distances_target_to_source = compute_distance_between_point_cloud_and_mesh(target_pcd_dense, aligned_source_mesh)

# Set threshold for determining matching areas
distance_threshold = 0.5  # Adjust this value based on your models

# Color the aligned source mesh
aligned_source_vertices = np.asarray(aligned_source_mesh.vertices)
aligned_source_mesh.vertex_colors = o3d.utility.Vector3dVector(
    np.zeros((len(aligned_source_vertices), 3))
)

# Sample points from the aligned source mesh to get vertices to color
# Create a kd-tree for the aligned source point cloud
pcd_tree = o3d.geometry.KDTreeFlann(aligned_source_pcd)

# For each vertex in the aligned source mesh, find the closest point in the point cloud
# and assign its distance-based color
vertex_colors = np.zeros((len(aligned_source_vertices), 3))
for i, vertex in enumerate(aligned_source_vertices):
    # Convert vertex to a 3D point
    query = np.array([vertex[0], vertex[1], vertex[2]])
    
    # Find the nearest point in the point cloud
    _, idx, _ = pcd_tree.search_knn_vector_3d(query, 1)
    
    # Get the distance associated with that point
    distance = distances_source_to_target[idx[0]]
    
    # Assign color based on distance (green for match, red for non-match)
    if distance < distance_threshold:
        vertex_colors[i] = [0, 1, 0]  # Green for matching areas
    else:
        vertex_colors[i] = [1, 0, 0]  # Red for non-matching areas

aligned_source_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

# Color the target mesh
target_vertices = np.asarray(target_mesh.vertices)
target_mesh.vertex_colors = o3d.utility.Vector3dVector(
    np.zeros((len(target_vertices), 3))
)

# Sample points from the target mesh to get vertices to color
# Create a kd-tree for the target point cloud
target_pcd_tree = o3d.geometry.KDTreeFlann(target_pcd_dense)

# For each vertex in the target mesh, find the closest point in the point cloud
# and assign its distance-based color
target_vertex_colors = np.zeros((len(target_vertices), 3))
for i, vertex in enumerate(target_vertices):
    # Convert vertex to a 3D point
    query = np.array([vertex[0], vertex[1], vertex[2]])
    
    # Find the nearest point in the point cloud
    _, idx, _ = target_pcd_tree.search_knn_vector_3d(query, 1)
    
    # Get the distance associated with that point
    distance = distances_target_to_source[idx[0]]
    
    # Assign color based on distance (green for match, red for non-match)
    if distance < distance_threshold:
        target_vertex_colors[i] = [0, 1, 0]  # Green for matching areas
    else:
        target_vertex_colors[i] = [1, 0, 0]  # Red for non-matching areas

target_mesh.vertex_colors = o3d.utility.Vector3dVector(target_vertex_colors)

# Visualize the results using Open3D
def visualize_results():
    # Create a visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Add the colored meshes
    vis.add_geometry(aligned_source_mesh)
    vis.add_geometry(target_mesh)
    
    # Set render options
    opt = vis.get_render_option()
    opt.mesh_show_wireframe = False
    opt.mesh_show_back_face = True
    
    # Run the visualization
    vis.run()
    vis.destroy_window()

# Visualize the aligned and colored meshes
visualize_results()

# Optionally save the colored meshes as PLY for visualization
o3d.io.write_triangle_mesh("plyFiles/aligned_source_colored.ply", aligned_source_mesh)
o3d.io.write_triangle_mesh("plyFiles/target_colored.ply", target_mesh)

# Optional: Save the aligned mesh back to STL format using Trimesh
aligned_source_trimesh.export("stlFiles/aligned_source.stl")

print("Alignment and comparison complete. Green areas match, red areas don't match.")