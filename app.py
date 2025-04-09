import trimesh
import open3d as o3d
import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from open3d.visualization import gui
from open3d.visualization import rendering
# Load the tooth STL model using Trimesh
mesh_trimesh = trimesh.load_mesh("stlFiles/Rough.stl")

# Get vertices, faces, and normals
vertices = np.array(mesh_trimesh.vertices)
faces = np.array(mesh_trimesh.faces)
normals = np.array(mesh_trimesh.vertex_normals)

print("calculating mean_cuvature....")
# Compute Mean Curvature using Trimesh
mean_curvature = trimesh.curvature.discrete_mean_curvature_measure(mesh_trimesh, mesh_trimesh.vertices, radius=2)

print("calculated mean_cuvature")

def extract_largest_cavity(vertices, faces, cavity_indices):
    # Get unique cavity indices
    unique_cavity_indices = np.unique(cavity_indices)
    
    # Find faces that have all vertices in cavity_indices
    cavity_face_mask = np.isin(faces.ravel(), unique_cavity_indices).reshape(faces.shape)
    cavity_face_indices = np.where(np.all(cavity_face_mask, axis=1))[0]
    cavity_faces = faces[cavity_face_indices]
    
    # Create adjacency matrix for connected component analysis
    edges = set()
    for face in cavity_faces:
        edges.add((face[0], face[1]))
        edges.add((face[1], face[2]))
        edges.add((face[2], face[0]))
    
    # Create sparse adjacency matrix
    row, col = zip(*edges)
    row = np.array(row)
    col = np.array(col)
    data = np.ones_like(row)
    
    # Create sparse matrix with size equal to total vertices 
    # (will be pruned to cavity vertices later)
    adj_matrix = csr_matrix((data, (row, col)), shape=(vertices.shape[0], vertices.shape[0]))
    
    # Find connected components
    n_components, labels = connected_components(csgraph=adj_matrix, directed=False)
    
    # Count vertices in each component
    component_sizes = np.zeros(n_components, dtype=int)
    for label in labels[unique_cavity_indices]:
        component_sizes[label] += 1
    
    # Find the largest component
    largest_component = np.argmax(component_sizes)
    
    # Get vertices from the largest component
    largest_cavity_indices = np.where(labels == largest_component)[0]
    largest_cavity_indices = np.intersect1d(largest_cavity_indices, unique_cavity_indices)
    
    # Create index mapping for new mesh
    index_map = np.zeros(len(vertices), dtype=int)
    for i, idx in enumerate(largest_cavity_indices):
        index_map[idx] = i
    
    # Get faces for largest component
    largest_face_mask = np.isin(cavity_faces.ravel(), largest_cavity_indices).reshape(cavity_faces.shape)
    largest_face_indices = np.where(np.all(largest_face_mask, axis=1))[0]
    largest_cavity_faces = cavity_faces[largest_face_indices]
    
    # Remap face indices
    remapped_faces = np.zeros_like(largest_cavity_faces)
    for i in range(largest_cavity_faces.shape[0]):
        for j in range(3):
            remapped_faces[i, j] = index_map[largest_cavity_faces[i, j]]
    
    # Create and return the largest cavity mesh
    largest_cavity_mesh = o3d.geometry.TriangleMesh()
    largest_cavity_mesh.vertices = o3d.utility.Vector3dVector(vertices[largest_cavity_indices])
    largest_cavity_mesh.triangles = o3d.utility.Vector3iVector(remapped_faces)
    largest_cavity_mesh.compute_vertex_normals()
    
    # Set color for visualization
    cavity_colors = np.ones((len(largest_cavity_indices), 3)) * [0, 1, 0]  # Green
    largest_cavity_mesh.vertex_colors = o3d.utility.Vector3dVector(cavity_colors)
    
    return largest_cavity_mesh, largest_cavity_indices
def extract_cavity_bottom(largest_cavity_mesh, threshold_percentage=0.1):
    # Get vertices and triangles from the mesh
    cavity_vertices = np.asarray(largest_cavity_mesh.vertices)
    cavity_triangles = np.asarray(largest_cavity_mesh.triangles)
    
    # Calculate z-range
    min_z = np.min(cavity_vertices[:, 2])
    max_z = np.max(cavity_vertices[:, 2])
    z_range = max_z - min_z
    
    # Define a threshold for what constitutes the "bottom"
    # Here we're considering the bottom 10% of the cavity's depth
    z_threshold = min_z + z_range * threshold_percentage
    
    # Find vertices that are in the bottom region
    bottom_vertex_mask = cavity_vertices[:, 2] <= z_threshold
    bottom_vertex_indices = np.where(bottom_vertex_mask)[0]
    
    # Find triangles where all three vertices are in the bottom region
    bottom_triangles_mask = np.isin(cavity_triangles.ravel(), bottom_vertex_indices).reshape(cavity_triangles.shape)
    bottom_triangle_indices = np.where(np.all(bottom_triangles_mask, axis=1))[0]
    
    if len(bottom_triangle_indices) == 0:
        print("No triangles found in the bottom region. Try adjusting the threshold.")
        return None
    
    # Create a new mesh for the bottom surface
    bottom_triangles = cavity_triangles[bottom_triangle_indices]
    
    # Get unique vertices used in the bottom triangles
    unique_vertices = np.unique(bottom_triangles.ravel())
    
    # Create index mapping
    index_map = np.zeros(len(cavity_vertices), dtype=int)
    for i, idx in enumerate(unique_vertices):
        index_map[idx] = i
    
    # Remap triangle indices
    remapped_triangles = np.zeros_like(bottom_triangles)
    for i in range(bottom_triangles.shape[0]):
        for j in range(3):
            remapped_triangles[i, j] = index_map[bottom_triangles[i, j]]
    
    # Create bottom surface mesh
    bottom_mesh = o3d.geometry.TriangleMesh()
    bottom_mesh.vertices = o3d.utility.Vector3dVector(cavity_vertices[unique_vertices])
    bottom_mesh.triangles = o3d.utility.Vector3iVector(remapped_triangles)
    bottom_mesh.compute_vertex_normals()
    
    # Set color for visualization
    bottom_colors = np.ones((len(unique_vertices), 3)) * [0, 1, 0]  # Blue for bottom surface
    bottom_mesh.vertex_colors = o3d.utility.Vector3dVector(bottom_colors)
    
    return bottom_mesh

cavity_indices = np.where(mean_curvature < 0.4)[0]  # Select all vertices with negative curvature
outline_indices = np.where((mean_curvature > 3.0))[0]
largest_cavity_mesh, largest_cavity_indices = extract_largest_cavity(vertices, faces, cavity_indices)

cavity_vertices= np.asarray(largest_cavity_mesh.vertices)


cavity_bottom = extract_cavity_bottom(largest_cavity_mesh, threshold_percentage=0.4)
# 153'ünücü satırın altına
cavity_bottom.compute_vertex_normals()

######################### convert a open3d Triangle mesh to a trimesh mesh

# Convert to numpy arrays
cav_bot_vertices = np.asarray(cavity_bottom.vertices)
cav_bot_faces = np.asarray(cavity_bottom.triangles)

# Create a Trimesh object
tri_mesh = trimesh.Trimesh(vertices=cav_bot_vertices, faces=cav_bot_faces, process=False)

######################### calculate smoothness

normals = tri_mesh.face_normals
adj = tri_mesh.face_adjacency

# Compute angle between adjacent face normals
dot = np.einsum('ij,ij->i', normals[adj[:, 0]], normals[adj[:, 1]])
angles = np.arccos(np.clip(dot, -1.0, 1.0))
angles_deg = np.degrees(angles)

print("Mean angle between adjacent faces:", np.mean(angles_deg))
print("Standard deviation (roughness):", np.std(angles_deg))

#########################

# kavite altının z eksenindeki ortalamasını al

bottom_vertices = np.asarray(cavity_bottom.vertices)

bottom_z_values = bottom_vertices[:, 2]
min_z = np.mean(bottom_z_values) 

max_z = np.max(cavity_vertices[:, 2])
cavity_depth = max_z - min_z  # Derinlik (Z eksenindeki fark)

cavity_centroid = np.mean(cavity_vertices, axis=0)


min_z_point = [cavity_centroid[0], cavity_centroid[1], min_z]
max_z_point = [cavity_centroid[0], cavity_centroid[1], max_z]

# **Çizgiyi tanımlama**
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector([min_z_point, max_z_point])
line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # Mavi çizgi

print(cavity_depth)

o3d.visualization.draw_geometries([line_set,cavity_bottom])

app = gui.Application.instance
app.initialize()

window = app.create_window("Cavity Depth Visualization", 1024, 768)
widget3d = gui.SceneWidget()
widget3d.scene = rendering.Open3DScene(window.renderer)
window.add_child(widget3d)

mesh = o3d.geometry.TriangleMesh.create_sphere()
mesh.compute_vertex_normals()

mat_cavity = rendering.MaterialRecord()
mat_cavity.shader = 'defaultLit'
# mat_cavity.base_color = np.array([0.5, 0.0, 0.0, 1.0], dtype=np.float32)  # RGBA
widget3d.scene.add_geometry("cavity_bottom", cavity_bottom, mat_cavity)

mat_line = rendering.MaterialRecord()
mat_line.shader = 'defaultLit'
mat_line.base_color = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)  # RGBA
widget3d.scene.add_geometry("depth_line", line_set,mat_line)

# Create a material for the label
mat = rendering.MaterialRecord()
mat.shader = "unlitText"

midpoint = [(min_z_point[0] + max_z_point[0])/2, 
            (min_z_point[1] + max_z_point[1])/2, 
            (min_z_point[2] + max_z_point[2])/2]
# Add a 3D text label at the midpoint
depth_str = f"{cavity_depth:.4f}"
# widget3d.add_3d_label(midpoint, depth_str) 



bounds = widget3d.scene.bounding_box
widget3d.setup_camera(60, bounds, bounds.get_center())
widget3d.look_at(bounds.get_center(), bounds.get_center() + [0, 0, 5], [0, 1, 0])

# Run the app
app.run()




# # Visualize
# # Get unique cavity vertices and their indices
# unique_cavity_indices = np.unique(cavity_indices)
# cavity_vertices = vertices[unique_cavity_indices]
# # Create a mapping from original vertex indices to new vertex indices
# index_map = np.zeros(len(vertices), dtype=int)
# for i, idx in enumerate(unique_cavity_indices):
#     index_map[idx] = i


# mesh_o3d = o3d.geometry.TriangleMesh()
# mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
# mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
# mesh_o3d.compute_vertex_normals()

# colors = np.ones((vertices.shape[0], 3)) * 0.7  # Light gray for normal surface
# colors[outline_indices] = [1, 0, 0]  # Red for cavities
# colors[cavity_indices] = [0, 1, 0]  # Red for cavities
# mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(colors)

# # Create a new mesh for just the cavity portion
# cavity_mesh = o3d.geometry.TriangleMesh()


# # Find faces that have all vertices in cavity_indices
# cavity_face_mask = np.isin(faces.ravel(), unique_cavity_indices).reshape(faces.shape)
# cavity_face_indices = np.where(np.all(cavity_face_mask, axis=1))[0]
# cavity_faces = faces[cavity_face_indices]

# # Remap the face indices to use the new vertex indices
# remapped_cavity_faces = np.zeros_like(cavity_faces)
# for i in range(cavity_faces.shape[0]):
#     for j in range(3):
#         remapped_cavity_faces[i, j] = index_map[cavity_faces[i, j]]

# # Set the vertices and faces for the cavity mesh
# cavity_mesh.vertices = o3d.utility.Vector3dVector(cavity_vertices)
# cavity_mesh.triangles = o3d.utility.Vector3iVector(remapped_cavity_faces)

# # Compute normals for visualization
# cavity_mesh.compute_vertex_normals()

# # Optional: Set color for cavity mesh
# cavity_colors = np.ones((len(cavity_vertices), 3)) * [0, 1, 0]  # Green color
# cavity_mesh.vertex_colors = o3d.utility.Vector3dVector(cavity_colors)

# # Visualize both meshes if needed
# o3d.visualization.draw_geometries([cavity_mesh])







# # Visualize tooth with cavity detection
