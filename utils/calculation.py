import numpy as np
import open3d as o3d
import trimesh
def calculate_oklidian_length_point(point1 , point2 ): 
    x1 = np.asarray(point1.points)[0]
    x2 = np.asarray(point2.points)[0]

    # Öklidyen mesafe hesapla
    distance = np.linalg.norm(x1 - x2)
    return distance

def split_side_and_get_normal_means(side_mesh):
    """
    Splits a side mesh into right and left walls and returns their mean normal vectors.
    
    Args:
        side_mesh: Open3D triangle mesh representing side walls
        
    Returns:
        right_mesh: The right side wall mesh
        left_mesh: The left side wall mesh
        right_normal_mean: Mean normal vector of the right mesh
        left_normal_mean: Mean normal vector of the left mesh
    """
    # Ensure we have triangle normals
    side_mesh.compute_triangle_normals()
    triangle_normals = np.asarray(side_mesh.triangle_normals)
    
    # Project normals onto the x-y plane
    xy_normals = triangle_normals[:, :2]
    
    # Calculate angles in the x-y plane (in radians)
    angles = np.arctan2(xy_normals[:, 1], xy_normals[:, 0])
    
    # Convert to degrees and shift to 0-360 range
    angles_deg = np.degrees(angles) % 360
    
    # Define right (0-180) and left (180-360) indices
    right_indices = np.where((angles_deg >= 0) & (angles_deg < 180))[0]
    left_indices = np.where((angles_deg >= 180) & (angles_deg < 360))[0]
    
    # Calculate mean normal for right side
    right_normals = triangle_normals[right_indices]
    right_normal_mean = np.mean(right_normals, axis=0)
    # Normalize the mean normal vector
    right_normal_mean = right_normal_mean / np.linalg.norm(right_normal_mean)
    
    # Calculate mean normal for left side
    left_normals = triangle_normals[left_indices]
    left_normal_mean = np.mean(left_normals, axis=0)
    # Normalize the mean normal vector
    left_normal_mean = left_normal_mean / np.linalg.norm(left_normal_mean)
    
    # Create the right mesh
    vertices = np.asarray(side_mesh.vertices)
    triangles = np.asarray(side_mesh.triangles)
    
    right_triangles = triangles[right_indices]
    unique_right_vertices = np.unique(right_triangles.ravel())
    
    vertex_map_right = np.zeros(len(vertices), dtype=int)
    for i, idx in enumerate(unique_right_vertices):
        vertex_map_right[idx] = i
        
    remapped_right_triangles = np.zeros_like(right_triangles)
    for i in range(right_triangles.shape[0]):
        for j in range(3):
            remapped_right_triangles[i, j] = vertex_map_right[right_triangles[i, j]]
            
    right_mesh = o3d.geometry.TriangleMesh()
    right_mesh.vertices = o3d.utility.Vector3dVector(vertices[unique_right_vertices])
    right_mesh.triangles = o3d.utility.Vector3iVector(remapped_right_triangles)
    
    # Create the left mesh
    left_triangles = triangles[left_indices]
    unique_left_vertices = np.unique(left_triangles.ravel())
    
    vertex_map_left = np.zeros(len(vertices), dtype=int)
    for i, idx in enumerate(unique_left_vertices):
        vertex_map_left[idx] = i
        
    remapped_left_triangles = np.zeros_like(left_triangles)
    for i in range(left_triangles.shape[0]):
        for j in range(3):
            remapped_left_triangles[i, j] = vertex_map_left[left_triangles[i, j]]
            
    left_mesh = o3d.geometry.TriangleMesh()
    left_mesh.vertices = o3d.utility.Vector3dVector(vertices[unique_left_vertices])
    left_mesh.triangles = o3d.utility.Vector3iVector(remapped_left_triangles)
    
    return right_mesh, left_mesh, right_normal_mean, left_normal_mean

def calculate_roughness(mesh):
    mesh_vertices = np.asarray(mesh.vertices)
    mesh_faces = np.asarray(mesh.triangles)
    tri_mesh = trimesh.Trimesh(vertices=mesh_vertices, faces=mesh_faces, process=False)
    normals = tri_mesh.face_normals
    adj = tri_mesh.face_adjacency

    # Compute angle between adjacent face normals
    dot = np.einsum('ij,ij->i', normals[adj[:, 0]], normals[adj[:, 1]])
    angles = np.arccos(np.clip(dot, -1.0, 1.0)) # Mean angle
    angles_deg = np.degrees(angles) # roughness
    return angles_deg

def calculate_distal_mesial_marginal_ridge_width():
    pass